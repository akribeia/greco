"""Float16 MRC storage for tomograms; reconstruction arithmetic stays float32."""
import numpy as np


def float16_data(value):
    """Reject invalid/out-of-range values instead of silently saving infinities."""
    value = np.asarray(value)
    if not np.isfinite(value).all() or (value.size and np.max(np.abs(value)) > 65504.):
        raise ValueError('Tomogram contains nonfinite values or exceeds the float16 storage range.')
    return np.asarray(value, dtype=np.float16)
