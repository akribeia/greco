"""TinyCARE protocol and model metadata; safe to import in the web service."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

ARCHITECTURE = 'residual_3d_UNet_12x2'
FORMAT = 'greco-tinycare-torch-v1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 15
    steps_per_epoch: int = 200
    batch_size: int = 16
    seed: int = 17
    tf32: bool = True
    precision: str = 'fp16'

    @classmethod
    def from_mapping(cls, config):
        values = {key: config.get(key, default) for key, default in vars(cls()).items()}
        for key in ('epochs', 'steps_per_epoch', 'batch_size', 'seed'):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError('TinyCARE ' + key + ' must be an integer.')
            if key != 'seed' and value < 1:
                raise ValueError('TinyCARE ' + key + ' must be positive.')
        if not 0 <= values['seed'] < 2**63:
            raise ValueError('TinyCARE seed must be between zero and 2^63-1.')
        if not isinstance(values['tf32'], bool):
            raise ValueError('TinyCARE tf32 must be true or false.')
        if values['precision'] not in ('fp16', 'fp32'):
            raise ValueError('TinyCARE precision must be fp16 or fp32.')
        return cls(**values)


def model_metadata(directory, pixel_size=None):
    """One model contract for GUI submission and training/inference workers."""
    directory = Path(directory)
    try:
        norm = json.loads((directory / 'norm.json').read_text())
        architecture = json.loads((directory / 'config.json').read_text())
    except (OSError, ValueError) as error:
        raise ValueError('TinyCARE model metadata is missing or unreadable.') from error
    if not isinstance(norm, dict) or not isinstance(architecture, dict):
        raise ValueError('Invalid TinyCARE model metadata.')
    if architecture.get('format') != FORMAT or architecture.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported TinyCARE architecture or format.')
    if norm.get('architecture') != ARCHITECTURE:
        raise ValueError('Incompatible TinyCARE normalization architecture.')
    for key in ('mean', 'std', 'pixel_size_A'):
        value = norm.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid TinyCARE model ' + key + '.')
    if norm['std'] <= 0 or norm['pixel_size_A'] <= 0:
        raise ValueError('TinyCARE model variance and sampling must be positive.')
    steps = norm.get('training_steps')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('Invalid TinyCARE model training step count.')
    if pixel_size is not None and not math.isclose(norm['pixel_size_A'], pixel_size, rel_tol=1e-6):
        raise ValueError('TinyCARE model pixel size must match the target tomogram.')
    filename = norm.get('selected_weights')
    if not isinstance(filename, str) or not filename or '\\' in filename:
        raise ValueError('Invalid TinyCARE checkpoint filename.')
    checkpoint = directory / filename
    if Path(filename).name != filename or checkpoint.suffix.lower() != '.pt':
        raise ValueError('Invalid TinyCARE checkpoint filename.')
    if checkpoint.resolve().parent != directory.resolve() or not checkpoint.is_file():
        raise ValueError('TinyCARE checkpoint is missing or outside its model directory.')
    return norm, checkpoint
