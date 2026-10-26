"""Remove only regenerable even/odd TinyCARE observations, never model/outputs."""
from pathlib import Path
import shutil
from ..project_layout import join as storage_path


def cleanup_intermediates(output):
    output=Path(output)
    candidates=[storage_path(output,'half_sources'),storage_path(output,'tomogram_even.mrc'),storage_path(output,'tomogram_odd.mrc')]
    removed=[];size=0
    for path in candidates:
        if path.is_symlink():raise ValueError('Refusing linked TinyCARE intermediate: '+str(path))
        if not path.exists():continue
        if path.is_dir():
            size+=sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
            shutil.rmtree(path)
        else:size+=path.stat().st_size;path.unlink()
        removed.append(str(path))
    return dict(removed=removed,bytes_removed=size)
