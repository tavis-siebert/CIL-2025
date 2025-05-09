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

## Use the cache
To save intermediate outputs of expensive function calls to the cache, you can use the `CACHE` object provided by the `cache.py` module. To specifically save and load embeddings from the cache, you can use the `save_embeddings` and `load_embeddings` wrappers around the `CACHE` object.

### Cache embeddings
To extract and save embeddings from [`SentenceTransformer`](https://huggingface.co/models?library=sentence-transformers) models to the cache, run
```bash
python scripts/save_embeddings.py --cache <cache_dir> --pipeline sentencetransformer --model <model_name>
```

To extract and save embeddings and predictions from [HuggingFace](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending&search=sentiment) models to the cache, run
```bash
python scripts/save_embeddings.py --cache <cache_dir> --pipeline huggingface --model <model_name>
```

To save custom embeddings to the cache, use
```python
from cache import CACHE, save_embeddings

CACHE.init(cache_dir=<cache_dir>)

save_embeddings(embeddings, <pipeline_name>, <model_name>)
```

To load the saved embeddings from the cache, use
```python
from cache import CACHE, load_embeddings

CACHE.init(cache_dir=<cache_dir>)

embeddings = load_embeddings(<pipeline_name>, <model_name>)
```

### Cache custom outputs
To cache the output of a function call, use
```python
from cache import CACHE

CACHE.init(cache_dir=<cache_dir>)

y = f(x) # no cache
y = CACHE(lambda: f(x), "y.npz") # cached to <cache_dir>/y.npz
```

Depending on the provided file ending, `CACHE` expects the following return type from the function call:
- `.npy`: expects `np.ndarray`
- `.npz`: expects `dict[str, np.ndarray]`
- `.pt`: expects `torch.Tensor`
- `.csv`: expects `pd.DataFrame`
- `.pkl`: expects any object
