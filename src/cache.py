import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, **kwargs):
        self.init(**kwargs)

    def init(self, cache_dir=None):
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            logger.info(f"Cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None

    def __call__(self, f, path, refresh=False, save_kwargs={}, load_kwargs={}, verbose=True):
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set. Please call init() first.")
        path = Path(self.cache_dir) / path

        if not path.is_file() or refresh:
            # evaluate function
            output = f()
            # save to cache
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npy":
                np.save(path, output, **save_kwargs)
            elif path.suffix == ".npz":
                np.savez_compressed(path, **output, **save_kwargs)
            elif path.suffix == ".pt":
                torch.save(output, path, **save_kwargs)
            elif path.suffix == ".csv":
                if not isinstance(output, pd.DataFrame):
                    raise ValueError(f"Output is not a DataFrame: {type(output)}")
                output.to_csv(path, index=False, **save_kwargs)
            elif path.suffix == ".pkl":
                with open(path, "wb") as f:
                    pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL, **save_kwargs)
            else:
                raise ValueError(f"Unknown file format: {path.suffix}")
            if verbose:
                logger.info(f"Saved to cache: {path}")
        else:
            # load from cache
            if path.suffix in [".npy", ".npz"]:
                output = np.load(path, **load_kwargs)
            elif path.suffix == ".pt":
                output = torch.load(path, **load_kwargs)
            elif path.suffix == ".csv":
                output = pd.read_csv(path, **load_kwargs)
            elif path.suffix == ".pkl":
                with open(path, "rb") as f:
                    output = pickle.load(f, **load_kwargs)
            else:
                raise ValueError(f"Unknown file format: {path.suffix}")
            if verbose:
                logger.info(f"Loaded from cache: {path}")
        return output

    def get_path(self, path):
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set. Please call init() first.")
        return self.cache_dir / path

CACHE = Cache()


def get_embeddings_folder(pipeline_name, model_name):
    return Path("embeddings") / pipeline_name / model_name.replace("/", "__")


def save_embeddings(embeddings, pipeline_name, model_name, file_name="embeddings.npz", **kwargs):
    """Save embeddings from the cache."""
    path = get_embeddings_folder(pipeline_name, model_name) / file_name
    return CACHE(lambda: embeddings, path, refresh=True, **kwargs)


def load_embeddings(pipeline_name, model_name, file_name="embeddings.npz", **kwargs):
    """Load embeddings from the cache."""
    path = get_embeddings_folder(pipeline_name, model_name) / file_name
    def raise_not_found():
        raise FileNotFoundError(f"Embeddings not found in cache: {path}")
    return CACHE(raise_not_found, path, **kwargs)
