# CIL 2025 - Sentiment Analysis

This repository contains our code for the Computational Intelligence Lab (CIL) 2025 course project at the department of ETH Zurich on sentiment analysis. The goal of this project is to implement and evaluate various machine learning pipelines for sentiment classification on a given dataset.

## Setup
We use `Python 3.12.3` with the following dependencies:
```bash
pip install -r requirements.txt
```

## Run a pipeline
To run a pipeline, use the command
```bash
python scripts/run_pipeline.py --config config/<config_file>.yaml
```

We have implemented the following pipelines:
- `classical_ml_bow_*.yaml`: bag-of-words embeddings + classical machine learning models (e.g., logistic regression, random forest, SVM, XGBoost)
- `mlp_head.yaml`: pretrained embeddings + MLP head
- `boosted_mlp_head.yaml`: pretrained embeddings + boosted MLP head
- `pretrained_classifier.yaml`: pretrained language models (inference-only)
- `finetuned_classifier.yaml`: finetuned language models
- `mixture_of_experts.yaml`: multiple pretrained embeddings + mixture-of-experts module

## Extract embeddings
For some pipelines, we use pretrained embeddings extracted from pretrained models. To extract and save these embeddings to the cache, use the `save_embeddings.py` script.
- To extract and save embeddings from [`SentenceTransformer`](https://huggingface.co/models?library=sentence-transformers) models to the cache, run
    ```bash
    python scripts/save_embeddings.py --cache <cache_dir> --pipeline sentencetransformer --model <model_name>
    ```
- To extract and save embeddings and predictions from [HuggingFace](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending&search=sentiment) models to the cache, run
    ```bash
    python scripts/save_embeddings.py --cache <cache_dir> --pipeline huggingface --model <model_name>
    ```

## Final submission
To reproduce our final submission with a train score of `0.96351` and validation score of `0.90646`, first finetune `FacebookAI/roberta-large` with the following command:
```bash
python scripts/run_pipeline.py --config config/finetuned_classifier.yaml
```

Then update `config/finetuned_classifier.yaml` with the following parameters:
- `pipeline.model.pretrained_model_name_or_path`: change it to your last checkpoint path
- `pipeline.preprocessing.difficulty_filter`: uncomment
- `pipeline.trainer.learning_rate`: change it to `1e-6`

Finally, post-tune your finetuned model with the following command:
```bash
python scripts/run_pipeline.py --config config/finetuned_classifier.yaml
```

## Contributing
If you plan to contribute to this repository, run
```
pip install -r requirements_dev.txt
pre-commit install
nbstripout --install
```
to install the [pre-commit](https://pre-commit.com/) and [nbstripout](https://github.com/kynan/nbstripout) hooks.

To contribute to this repository, please work on a branch named `<name>/<description>` and create pull requests.

### Add a pipeline
To add new pipelines, create the following two files
* `config/<config_file>.yaml`: The configuration file with the module name of the pipeline and all hyperparameters.
* `src/pipelines/<module_file>.py`: The module file with the pipeline definition.

### Use the cache
To save intermediate outputs of expensive function calls to the cache, you can use the `CACHE` object provided by the `cache.py` module. To specifically save and load embeddings from the cache, you can use the `save_embeddings` and `load_embeddings` wrappers around the `CACHE` object.

#### Cache embeddings
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

#### Cache custom outputs
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


## Authors

The following individuals contributed equally to this project:

* **Redhardt, Florian** - [GitHub Profile](https://github.com/Florian-toll)
* **Siebert, Tavis** - [GitHub Profile](https://github.com/tavis-siebert)
* **Stante, Samuel** - [GitHub Profile](https://github.com/Timisorean)
* **Yang, Daniel** - [GitHub Profile](https://github.com/danielyxyang)
