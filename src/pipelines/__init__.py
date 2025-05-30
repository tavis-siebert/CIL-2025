import importlib

from omegaconf import DictConfig, ListConfig

from .base import BasePipeline


def load_pipeline(pipeline_config: DictConfig | ListConfig, **kwargs) -> BasePipeline:
    """Load a pipeline class based on the configuration."""
    module_name, class_name = pipeline_config.name.rsplit(".", maxsplit=1)
    pipeline_cls = getattr(importlib.import_module(f".{module_name}", package=__package__), class_name)
    return pipeline_cls(config=pipeline_config, **kwargs)
