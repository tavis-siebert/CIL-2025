import importlib


def load_pipeline(pipeline_config):
    module_name, class_name = pipeline_config.name.rsplit(".", maxsplit=1)
    pipeline_cls = getattr(importlib.import_module(f"pipelines.{module_name}"), class_name)
    return pipeline_cls(pipeline_config)
