"""Runtime paths for source bundles and installed Greco packages."""
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
KINDS = ('main', 'vision', 'membranes', 'gpu', 'tinycare')


def _path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


RUNTIME = _path('GRECO_RUNTIME_ROOT', ROOT / 'runtime')
ASSETS = _path('GRECO_ASSETS_ROOT', ROOT / 'assets' if (ROOT / 'assets').is_dir() else PACKAGE / 'assets')
# Preserve the tested bundle. A normal environment uses its own CUDA packages.
def _cuda_root():
    bundled = RUNTIME / 'cuda'
    if bundled.is_dir():
        return bundled
    # Keep runtime headers and NVRTC libraries on the same wheel version.
    for entry in sys.path:
        if entry:
            candidate = Path(entry) / 'nvidia' / 'cuda_runtime'
            if (candidate / 'include' / 'cuda_runtime.h').is_file():
                return candidate
    return bundled


CUDA = _path('GRECO_CUDA_PATH', _cuda_root())


def python(kind='main'):
    if kind not in KINDS:
        raise ValueError('Unknown Greco runtime: ' + kind)
    override = os.environ.get('GRECO_' + kind.upper() + '_PYTHON') or os.environ.get('GRECO_PYTHON')
    if override:
        return Path(override).expanduser().absolute()
    bundled = RUNTIME / 'gpu/bin/python'
    return bundled if bundled.is_file() else Path(sys.executable)


def libraries():
    paths = [RUNTIME / 'gpu/lib', RUNTIME / 'native-libs', CUDA / 'lib64']
    # NVIDIA wheel libraries in the interpreter's environment, including NVRTC.
    for entry in sys.path:
        if not entry:
            continue
        nvidia = Path(entry) / 'nvidia'
        if nvidia.is_dir():
            paths.extend(sorted(nvidia.glob('*/lib')))
    return list(dict.fromkeys(p for p in paths if p.is_dir()))


def worker_environment(kind='main'):
    if kind not in KINDS:
        raise ValueError('Unknown Greco runtime: ' + kind)
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Avoid shared-filesystem reads of compiled NVRTC kernels.
    env['CUPY_CACHE_IN_MEMORY'] = '1'
    env['GRECO_RUNTIME_ROOT'] = str(RUNTIME)
    env['GRECO_ASSETS_ROOT'] = str(ASSETS)
    paths = [ROOT]
    overlay = RUNTIME / 'cupy-overlay'
    if overlay.is_dir():
        paths.append(overlay)
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    if CUDA.is_dir():
        env['CUDA_PATH'] = env['CUDA_HOME'] = str(CUDA)
        if (CUDA / 'bin').is_dir():
            env['PATH'] = str(CUDA / 'bin') + os.pathsep + env.get('PATH', '')
    native = list(map(str, libraries()))
    inherited = env.get('LD_LIBRARY_PATH', '')
    if inherited:
        native.extend(inherited.split(os.pathsep))
    if native:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(native))
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '4'
    return env


def main():
    """Re-exec once so native loaders see the configured library paths."""
    executable = str(python())
    os.execve(executable, [executable, '-m', 'greco.launcher', *sys.argv[1:]], worker_environment())


if __name__ == '__main__':
    main()
