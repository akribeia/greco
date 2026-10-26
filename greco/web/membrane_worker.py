"""Greco membranes standard inference: bs8, 50% overlap, eight mirror TTA folds."""
from greco.project_layout import join as storage_path
import argparse
import hashlib
import json
from pathlib import Path
import time


def run(config):
    import mrcfile
    import numpy as np
    import torch
    from monai.inferers import SlidingWindowInferer
    from greco._membrane.preprocessing import load_data_for_inference, get_mirrored_img, get_prediction_transforms
    from greco._membrane.checkpoint import load_model, FORMAT

    start = time.monotonic()
    output = Path(config['output'])
    def progress(phase, **extra):
        temporary = storage_path(output, 'progress.tmp')
        temporary.write_text(json.dumps(dict(phase=phase, **extra)))
        temporary.replace(storage_path(output, 'progress.json'))
    if not torch.cuda.is_available():
        raise RuntimeError('Membrane segmentation requires a CUDA GPU.')
    with mrcfile.mmap(config['input']) as m:
        shape = tuple(m.data.shape)
        pixel = np.array(m.voxel_size.tolist(), dtype=float)
        if m.data.ndim != 3 or not np.isfinite(m.data).all() or float(m.data.std()) <= 0:
            raise ValueError('Membrane segmentation requires a finite, nonconstant 3D tomogram.')
        if not np.isfinite(pixel).all() or (pixel <= 0).any() or not np.allclose(pixel, pixel[0], rtol=1e-5):
            raise ValueError('Membrane segmentation requires a valid isotropic MRC pixel size.')
        header = m.header.copy()
    progress('Membranes: loading model (batch size 8)')
    checkpoint = Path(config['checkpoint'])
    load_start = time.monotonic()
    model = load_model(checkpoint, device='cuda')
    loading_seconds = time.monotonic() - load_start
    volume, _, _ = load_data_for_inference(config['input'], get_prediction_transforms(), device=torch.device('cpu'))
    volume = volume.to(torch.float32)
    inferer = SlidingWindowInferer((160, 160, 160), 8, overlap=.5, progress=False,
                                  mode='gaussian', device=torch.device('cpu'))
    predictions = torch.zeros_like(volume)
    inference_start = time.monotonic()
    for fold in range(8):
        progress('Membranes: mirror inference', epoch=fold + 1, epochs=8)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            mirrored = get_mirrored_img(volume.clone(), fold).to('cuda')
            predicted = inferer(mirrored, model)
            if isinstance(predicted, (tuple, list)):
                predicted = predicted[0]
            predictions += get_mirrored_img(predicted, fold).detach().cpu()
        del mirrored, predicted
    predictions /= 8
    inference_seconds = time.monotonic() - inference_start
    # Upstream loads MRC ZYX as network XYZ; invert this on export.
    logits = predictions.squeeze(0).squeeze(0).numpy().transpose(2, 1, 0)
    if logits.shape != shape or not np.isfinite(logits).all():
        raise RuntimeError('Invalid membrane output grid or logits.')
    mask = (logits > 0).astype(np.int8)
    target = storage_path(output, 'membranes.mrc')
    # Keep source axes, physical origin and sampling; no rotation or inversion.
    with mrcfile.new(target, overwrite=False) as m:
        m.set_data(mask)
        m.voxel_size = tuple(pixel)
        for field in ('origin', 'nxstart', 'nystart', 'nzstart', 'mapc', 'mapr', 'maps'):
            m.header[field] = header[field]
        m.update_header_stats()
    with checkpoint.open('rb') as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = dict(membrane_layer=str(target), shape=list(shape), voxel_size_A=float(pixel[0]),
                  membrane_voxels=int(mask.sum()), membrane_fraction=float(mask.mean()),
                  architecture='Greco membranes U-Net v10', batch_size=8, overlap=.5, tta_folds=8,
                  threshold_logit=0., input=str(config['input']), rescaled=False,
                  inference_seconds=inference_seconds, runtime_seconds=time.monotonic() - start,
                  gpu=torch.cuda.get_device_name(), checkpoint_sha256=checkpoint_sha256,
                  checkpoint_format=getattr(model, "checkpoint_format", FORMAT), checkpoint_bytes=checkpoint.stat().st_size,
                  loading_seconds=loading_seconds)
    (storage_path(output, 'membranes.json')).write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
