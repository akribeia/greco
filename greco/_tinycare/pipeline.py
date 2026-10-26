"""Production TinyCARE train/reuse/predict flow, independent of the web CLI."""
from greco.volume_storage import float16_data
from dataclasses import asdict
from pathlib import Path
from ..project_layout import join as storage_path
import time

import mrcfile
import numpy as np
import torch

from .config import ARCHITECTURE, FORMAT, TrainingOptions, model_metadata, write_json
from .model import TinyCARE, load_weights
from .predict import predict_volume
from .train import train


def read_halves(paths):
    volumes = []
    header = None
    pixel = None
    for path in paths:
        with mrcfile.mmap(path) as source:
            sampling = np.asarray(source.voxel_size.tolist(), dtype=float)
            if source.data.ndim != 3 or not all(source.data.shape):
                raise ValueError('TinyCARE inputs must be nonempty 3D volumes.')
            if not np.isfinite(sampling).all() or (sampling <= 0).any() or not np.allclose(sampling, sampling[0], rtol=1e-6):
                raise ValueError('TinyCARE inputs require positive isotropic sampling.')
            if header is not None:
                if source.data.shape != volumes[0].shape or not np.isclose(sampling[0], pixel, rtol=1e-6):
                    raise ValueError('TinyCARE even/odd grids differ.')
                fields = ('origin', 'nxstart', 'nystart', 'nzstart', 'mapc', 'mapr', 'maps')
                for field in fields:
                    first = np.asarray(header[field].tolist())
                    second = np.asarray(source.header[field].tolist())
                    if not np.allclose(first, second, rtol=0., atol=1e-5):
                        raise ValueError('TinyCARE even/odd geometry differs: ' + field + '.')
            else:
                header = source.header.copy()
            pixel = float(sampling[0])
            volume = np.array(source.data, dtype=np.float32)
            if not np.isfinite(volume).all():
                raise ValueError('Nonfinite TinyCARE inputs.')
            volumes.append(volume)
    return volumes, pixel, header


def run(config):
    from .device import check_device
    check_device()
    options = TrainingOptions.from_mapping(config)
    output = Path(config['output'])
    target = output / 'tomogram_tinycare.mrc'
    if target.exists():
        raise ValueError('TinyCARE output already exists; create or clear the job first.')
    model_dir = Path(config.get('model_from') or storage_path(output,'tinycare_model'))
    if not config.get('model_from') and model_dir.exists():
        raise ValueError('TinyCARE model directory already exists; create or clear the job first.')
    # Fail before creating outputs or allocating a model when metadata is invalid.
    if config.get('model_from'):
        model_metadata(model_dir)
    volumes, pixel, header = read_halves([config['even'], config['odd']])
    if config.get('model_from'):
        model_metadata(model_dir, pixel)
    if not torch.cuda.is_available():
        raise RuntimeError('TinyCARE requires a CUDA GPU.')
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = options.tf32
    torch.backends.cuda.matmul.allow_tf32 = options.tf32
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)

    def progress(phase, **extra):
        write_json(storage_path(output,'progress.json'), dict(phase=phase, **extra))

    model = TinyCARE().to(device='cuda', memory_format=torch.channels_last_3d)
    training_seconds = 0.
    if not config.get('model_from'):
        model_dir.mkdir()
        progress('TinyCARE: extracting even/odd training patches')
        trained = train(model, volumes, model_dir, asdict(options), progress)
        training_seconds = trained['training_seconds']
        norm = dict(mean=trained['mean'], std=trained['std'], pixel_size_A=pixel,
                    architecture=ARCHITECTURE, format=FORMAT, selected_weights='weights_last.pt',
                    training_steps=trained['training_steps'], training_precision=trained['precision'])
        write_json(model_dir / 'config.json', dict(format=FORMAT, architecture=ARCHITECTURE,
            train_loss='mse', train_epochs=options.epochs, train_steps_per_epoch=options.steps_per_epoch,
            train_batch_size=options.batch_size, train_learning_rate=4e-4,
            patch_size=trained['patch_size'], seed=options.seed, tf32=options.tf32,
            training_precision=trained['precision']))
        write_json(model_dir / 'norm.json', norm)
        write_json(model_dir / 'training.json', dict(history=trained['history'], seconds=training_seconds,
            extraction_seconds=trained['extraction_seconds'], volume_cache=trained['volume_cache'], backend='torch',
            training_precision=trained['precision'], overflow_retries=trained['overflow_retries'],
            final_grad_scale=trained['final_grad_scale']))
    norm, checkpoint = model_metadata(model_dir, pixel)
    load_weights(model, checkpoint)
    model.eval()
    start = time.monotonic()
    result = np.zeros_like(volumes[0])
    tiles = None
    for name, volume in zip(('even', 'odd'), volumes):
        progress('TinyCARE: predicting ' + name)
        prediction, tiles = predict_volume(model, volume, norm['mean'], norm['std'],
                                           precision=options.precision)
        result += prediction * .5
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite TinyCARE prediction.')
    with mrcfile.new(target, overwrite=False) as destination:
        destination.set_data(float16_data(result))
        destination.voxel_size = pixel
        for field in ('origin', 'nxstart', 'nystart', 'nzstart', 'mapc', 'mapr', 'maps'):
            destination.header[field] = header[field]
        destination.update_header_stats()
    diagnostic = dict(method='TinyCARE', architecture=ARCHITECTURE,
        training_steps=norm['training_steps'], model_dir=str(model_dir),
        training_seconds=training_seconds, prediction_seconds=time.monotonic() - start,
        reused_model=bool(config.get('model_from')), shape=list(result.shape),
        pixel_size_A=pixel, tiles=tiles, output=str(target),
        inference='single_forward_block_aligned_halo_tiling', backend='torch',
        gpu=torch.cuda.get_device_name(), peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        merge='mean_of_separately_denoised_even_and_odd',
        training_precision=norm.get('training_precision',
            ('bf16' if norm['bf16'] else 'fp32') if isinstance(norm.get('bf16'), bool) else None),
        inference_precision=options.precision, storage_precision='fp16', mrc_mode=12)
    write_json(storage_path(output,'tinycare.json'), diagnostic)
    progress('TinyCARE: completed', training_steps=norm['training_steps'])
    return diagnostic
