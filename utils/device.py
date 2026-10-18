import torch


def initialize_device():
    # 1. NVIDIA CUDA / AMD ROCm
    if torch.cuda.is_available():
        return torch.device('cuda')

    # 2. Intel Arc / Data Center GPUs (Native PyTorch 2.4+ XPU or IPEX)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device('xpu')

    # 3. Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')

    # 4. Fallback CPU
    return torch.device('cpu')


def initialize_openvino_device(device_name='AUTO', core=None):
    if core is None:
        import openvino as ov
        core = ov.Core()

    requested = device_name.strip().upper()
    available = core.available_devices
    if requested == 'AUTO':
        gpu_device = next(
            (device for device in available if device == 'GPU' or device.startswith('GPU.')),
            None,
        )
        if gpu_device:
            return gpu_device
        raise RuntimeError(
            "OpenVINO device AUTO requires an available GPU; CPU fallback is "
            f"disabled. Available devices: {available}"
        )
    if requested == 'CPU':
        raise RuntimeError(
            "CPU inference is disabled for the OpenVINO backend. "
            "Select GPU or AUTO."
        )
    if not requested.startswith('GPU'):
        raise RuntimeError(
            f"OpenVINO device '{device_name}' is not a supported GPU target. "
            "CPU inference is disabled."
        )
    if requested in available:
        return requested
    if '.' not in requested and any(
        device == requested or device.startswith(f'{requested}.')
        for device in available
    ):
        return requested
    raise RuntimeError(
        f"OpenVINO device '{device_name}' is unavailable. "
        f"Available devices: {available}"
    )


def inference_device_type(device):
    device_name = str(device).lower()
    if device_name.startswith(('cuda', 'gpu', 'mps')):
        return 'GPU'
    if device_name.startswith(('xpu', 'intel_gpu')):
        return 'XPU'
    return 'CPU'