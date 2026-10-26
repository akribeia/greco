"""Check the installed PyTorch runtime on the allocated GPU."""


def check_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('TinyCARE requires a CUDA GPU.')
    try:
        # Execute a kernel: architecture lists alone do not account for PTX JIT.
        torch.ones(1, device='cuda').add_(1)
        torch.cuda.synchronize()
    except RuntimeError as error:
        name = torch.cuda.get_device_name()
        raise RuntimeError(
            f'TinyCARE cannot run on {name} with installed PyTorch {torch.__version__}. '
            'Request a GPU supported by this PyTorch build in the Slurm template. '
            f'CUDA reported: {error}'
        ) from error
