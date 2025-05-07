# CIL-2025 Sentiment Analysis

## Setup (dev)
If you plan to develop in this repository, please install the [pre-commit](https://pre-commit.com/) hooks:
```
pip install pre-commit
pre-commit install
```

## Save embeddings
To extract and save embeddings using [`SentenceTransformer`](https://huggingface.co/models?library=sentence-transformers) models, run
```bash
python scripts/save_embeddings.py --out /work/courses/pmlr/17/embeddings --model <model_name>
```
To load the embeddings, use
```python
TBD
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
