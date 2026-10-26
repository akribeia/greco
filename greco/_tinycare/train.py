# Third-party references and adapted components
# Greco PyTorch training following cryoCARE Noise2Noise training and CSBDeep/CARE optimizer conventions. Mixed precision and Greco training integration are Greco contributions.
#
# Original Greco contributions use the root LICENSE; upstream portions retain
# their notices and applicable terms. Attribution updated 2026-10-23.
# Retained notices:
# licenses/cryocare/LICENSE.txt
# licenses/csbdeep/LICENSE.txt
#
# Copyright (c) 2020, juglab
# Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
# Full license conditions and disclaimers are retained at the paths above.
#

"""Noise2Noise training with the validated TinyCARE objective and schedule."""
import math
import time

import torch
from torch.nn import functional as F

from .data import PatchData
from .model import save_weights
from .config import TrainingOptions


class TinyAdam(torch.optim.Adam):
    """Adam with epsilon applied to the biased second moment, as before."""
    def __init__(self, params, lr=4e-4, *, fused=False):
        super().__init__(params, lr=lr, betas=(.9, .999), eps=1e-7, fused=fused)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        for group in self.param_groups:
            # PyTorch divides v by its bias correction before adding epsilon.
            # Scale epsilon to reproduce the previous biased-v denominator.
            group['eps'] = 1e-7 / math.sqrt(1 - group['betas'][1] ** self.updates)
        return super().step(closure)


def scaled_step(optimizer, scaler):
    """Keep TinyAdam's epsilon schedule at successful updates on AMP overflow."""
    updates, scale = optimizer.updates, scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= scale
    if not success:
        optimizer.updates = updates
    return success


def train(model, volumes, model_dir, config, progress):
    options = TrainingOptions.from_mapping(config)
    epochs, steps, batch_size = options.epochs, options.steps_per_epoch, options.batch_size
    patch = min(72, 4 * ((min(volumes[0].shape) // 2 - 1) // 4))
    if patch < 16:
        raise ValueError('TinyCARE training needs a thicker volume (minimum dimension 34 voxels).')
    device = next(model.parameters()).device
    amp = device.type == 'cuda' and options.precision == 'fp16'
    precision = 'fp16' if amp else 'fp32'
    start = time.monotonic()
    data = PatchData(*volumes, patch, device=device, seed=options.seed)
    extraction_seconds = time.monotonic() - start
    optimizer = TinyAdam(model.parameters(), fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', init_scale=256., enabled=amp)
    retries = 0
    # The previous scheduler reduces on the tenth non-improving epoch.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=9, threshold=0., threshold_mode='abs', eps=0.)
    history = {key: [] for key in ('loss', 'val_loss', 'lr')}
    best = float('inf')
    progress('TinyCARE: training', epoch=0, epochs=epochs, training_precision=precision)
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        training_loss = torch.zeros((), device=device)
        for _ in range(steps):
            x, target = data.next_train(batch_size)
            for attempt in range(16):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    value = model(x)
                loss = F.mse_loss(value.float(), target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite TinyCARE forward/loss; gradient scaling cannot fix activation overflow.')
                scaler.scale(loss).backward()
                if scaled_step(optimizer, scaler):
                    break
                retries += 1
            else:
                raise ValueError('TinyCARE gradients repeatedly overflowed.')
            training_loss += loss.detach()
        model.eval()
        validation_loss = torch.zeros((), device=device)
        with torch.inference_mode():
            for offset in range(0, len(data.val), batch_size):
                points = data.val[offset:offset + batch_size]
                x, target = data.batch(points, augment=False)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    value = model(x)
                validation_loss += F.mse_loss(value.float(), target) * len(points)
        training_loss = float(training_loss / steps)
        validation_loss = float(validation_loss / len(data.val))
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
            raise ValueError('Nonfinite TinyCARE training loss.')
        history['loss'].append(training_loss)
        history['val_loss'].append(validation_loss)
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        scheduler.step(validation_loss)
        if validation_loss < best:
            best = validation_loss
            save_weights(model, model_dir / 'weights_best.pt')
        save_weights(model, model_dir / 'weights_last.pt')
        progress('TinyCARE: training', epoch=epoch + 1, epochs=epochs,
                 loss=training_loss, val_loss=validation_loss, training_precision=precision)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    if optimizer.updates != epochs * steps:
        raise RuntimeError('TinyCARE optimizer update count differs from the requested schedule.')
    return dict(mean=data.mean, std=data.std, training_steps=epochs * steps,
                patch_size=patch, extraction_seconds=extraction_seconds,
                training_seconds=time.monotonic() - start, history=history,
                volume_cache=data.volume_cache, precision=precision,
                overflow_retries=retries, final_grad_scale=scaler.get_scale())
