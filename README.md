# CIL-2025 Sentiment Analysis

## Setup (dev)
If you plan to develop in this repository, please install the [pre-commit](https://pre-commit.com/) hooks:
```
pip install pre-commit
pre-commit install
```

## Add a pipeline
To add new pipelines, create the following two files
* `config/<config_file>.yaml`: The configuration file with the module name of the pipeline and all hyperparameters.
* `src/pipelines/<module_file>.py`: The module file with the pipeline definition.

## Run a pipeline
To run a pipeline, use the command
```bash
python scripts/run_pipeline.py --config config/<config_file>.yaml
```
