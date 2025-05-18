import importlib

import torch
from omegaconf import DictConfig, ListConfig

from .base import BasePipeline


def load_pipeline(
    pipeline_config: DictConfig | ListConfig,
    device: str | torch.device | None = None,
    output_dir: str = "output",
    **kwargs,
) -> BasePipeline:
    module_name, class_name = pipeline_config.name.rsplit(".", maxsplit=1)
    pipeline_cls = getattr(importlib.import_module(f".{module_name}", package=__package__), class_name)
    return pipeline_cls(config=pipeline_config, device=device, output_dir=output_dir, **kwargs)
