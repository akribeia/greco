"""Strict, self-contained mixed BF16/FP8/INT4 membrane weights.

This is storage compression, not low-precision inference. Decode on CPU to
FP32, then use the unchanged network and inference autocast policy.
"""
from pathlib import Path
import torch

FORMAT = 'greco-membranes-v1'
ARCHITECTURE = 'greco-membranes-unet'




def load_model(path, device='cuda'):
    from .network import MembraneNetwork
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = MembraneNetwork()
    state = decode(payload, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.checkpoint_format = payload['format']
    del state, payload
    model.target_shape = (160, 160, 160)
    model.rescale_patches = False
    return model.to(device).eval()


def decode(payload, template):
    if not isinstance(payload, dict) or payload.get('format') != FORMAT or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported membrane architecture.')
    packed, meta, aliases, remaining = (payload.get(k) for k in ('packed', 'meta', 'canon', 'remaining'))
    if not all(isinstance(x, dict) for x in (packed, meta, aliases, remaining)):
        raise ValueError('Incomplete quantized MemBrain-seg checkpoint.')
    if any(not isinstance(r, dict) for r in meta.values()):
        raise ValueError('Invalid membrane metadata.')
    if set(packed) != {k for k, r in meta.items() if r.get('dtype') != 'int4'} or set(meta) & set(aliases) or (set(meta) | set(aliases)) & set(remaining):
        raise ValueError('Invalid membrane tensor groups.')
    if set(template) != set(meta) | set(aliases) | set(remaining):
        raise ValueError('Membrane checkpoint has missing or unexpected tensors.')
    state = {}
    for key, record in meta.items():
        expected = template[key]
        shape = tuple(expected.shape)
        kind = record.get('dtype')
        if tuple(record.get('shape', ())) != shape or expected.ndim < 2 or kind not in ('bf16', 'fp8', 'int4'):
            raise ValueError('Invalid membrane shape or encoding: ' + key)
        value = record.get('q2') if kind == 'int4' else packed[key]
        packed_shape = (*shape[:-1], (shape[-1] + 1) // 2) if kind == 'int4' else shape
        dtype = torch.bfloat16 if kind == 'bf16' else torch.uint8
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != packed_shape:
            raise ValueError('Invalid packed membrane tensor: ' + key)
        if kind == 'bf16':
            tensor = value.float()
        else:
            scale = record.get('s')
            if not isinstance(scale, torch.Tensor) or not scale.is_floating_point() or scale.shape != (shape[0],) or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError('Invalid membrane channel scale: ' + key)
            scale = scale.float().reshape(shape[0], *([1] * (len(shape) - 1)))
            if kind == 'fp8':
                tensor = value.contiguous().view(torch.float8_e4m3fn).float() * scale
            else:
                unpacked = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=torch.int8)
                unpacked[..., 0::2] = (value & 15).to(torch.int8)
                unpacked[..., 1::2] = (value >> 4).to(torch.int8)
                tensor = (unpacked[..., :shape[-1]].float() - 8.) * scale
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite membrane tensor: ' + key)
        state[key] = tensor.to(expected.dtype)
    for key, canonical in aliases.items():
        if not isinstance(canonical, str) or canonical not in meta or template[key].shape != state[canonical].shape or template[key].dtype != state[canonical].dtype:
            raise ValueError('Invalid membrane tensor alias: ' + key)
        state[key] = state[canonical]
    for key, value in remaining.items():
        if not isinstance(value, torch.Tensor) or value.shape != template[key].shape or value.dtype != template[key].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid uncompressed membrane tensor: ' + key)
        state[key] = value
    return state
