import hashlib
import os
import reprlib

import numpy as np
import torch

from utils.device import initialize_openvino_device
from utils.device import inference_device_type
from utils.logger import logger


class OpenVINOModel:
    def __init__(self, compiled_model, device_name):
        self.compiled_model = compiled_model
        self.device_name = device_name
        self.is_openvino = True
        self.dtype = torch.float32

    def __call__(self, *inputs):
        values = {
            self.compiled_model.input(index): _to_numpy(value)
            for index, value in enumerate(inputs)
        }
        result = self.compiled_model(values)
        outputs = tuple(
            torch.from_numpy(np.array(result[port], copy=True))
            for port in self.compiled_model.outputs
        )
        return outputs[0] if len(outputs) == 1 else outputs


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _source_signature(path):
    if not path:
        return ""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        stat = os.stat(path)
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    if os.path.isdir(path):
        entries = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                stat = os.stat(full_path)
                entries.append(
                    f"{os.path.relpath(full_path, path)}:{stat.st_size}:{stat.st_mtime_ns}"
                )
        return ";".join(sorted(entries))
    return path


def compile_torch_model(
    model,
    example_inputs,
    device_name,
    cache_name,
    source_paths=(),
    static_input_shapes=None,
    dynamic_input_shapes=None,
):
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO inference was requested, but the 'openvino' package is "
            "not installed. Install project requirements to use this backend."
        ) from exc

    core = ov.Core()
    requested_device = device_name.strip().upper()
    device_name = initialize_openvino_device(device_name, core)
    cache_metadata = "|".join(
        [
            cache_name,
            model.__class__.__qualname__,
            ov.__version__,
            reprlib.repr(tuple(tuple(value.shape) for value in example_inputs)),
            reprlib.repr(static_input_shapes),
            reprlib.repr(dynamic_input_shapes),
            *(_source_signature(path) for path in source_paths),
        ]
    )
    cache_id = hashlib.sha256(cache_metadata.encode("utf-8")).hexdigest()
    cache_dir = os.path.join("models", "openvino_cache")
    os.makedirs(cache_dir, exist_ok=True)
    xml_path = os.path.join(cache_dir, f"{cache_id}.xml")

    if os.path.isfile(xml_path):
        ov_model = core.read_model(xml_path)
    else:
        model = model.eval().cpu()
        with torch.no_grad():
            ov_model = ov.convert_model(model, example_input=example_inputs)
        if static_input_shapes:
            ov_model.reshape(
                {
                    ov_model.input(index): shape
                    for index, shape in static_input_shapes.items()
                }
            )
        if dynamic_input_shapes:
            ov_model.reshape(
                {
                    ov_model.input(index): ov.PartialShape(shape)
                    for index, shape in dynamic_input_shapes.items()
                }
            )
        ov.save_model(ov_model, xml_path)

    compiled_model = core.compile_model(ov_model, device_name)
    execution_devices = compiled_model.get_property(ov.properties.execution_devices)
    execution_types = sorted({inference_device_type(name) for name in execution_devices})
    if not execution_devices or execution_types != ['GPU']:
        raise RuntimeError(
            "OpenVINO GPU inference is required, but the compiled model reports "
            f"execution devices {execution_devices}. CPU fallback is disabled."
        )
    logger.info(
        "OpenVINO inference target: %s (requested: %s, selected: %s)",
        "/".join(execution_types),
        requested_device,
        device_name,
    )
    return OpenVINOModel(compiled_model, device_name)
