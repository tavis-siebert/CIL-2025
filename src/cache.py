"""Cache utilities for storing and retrieving data efficiently.

This module provides a Cache class that allows caching of function outputs to disk
and retrieving them later. It supports various file formats such as NumPy arrays,
PyTorch tensors, Pandas DataFrames, and pickled objects.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import torch

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Cache:
    """A class for caching function outputs to disk.

    This class allows you to cache the output of a function to a file and retrieve it later.
    It supports various file formats such as NumPy arrays, PyTorch tensors, Pandas DataFrames,
    and pickled objects. The cache directory can be specified, and the cache can be refreshed
    by re-evaluating the function if the cached file does not exist or if the `refresh` flag is set.
    The cached files are stored in a directory structure based on the provided path.
    """
    def __init__(self, **kwargs):
        self.init(**kwargs)

    def init(self, cache_dir: str | Path | None = None):
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            logger.info(f"Cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None

    def __call__(
        self,
        f: Callable[[], T],
        path: str | Path,
        refresh: bool = False,
        save_kwargs: dict[str, Any] = {},
        load_kwargs: dict[str, Any] = {},
        verbose: bool = True,
    ) -> T:
        """Cache the output of a function to a file.

        Args:
            f (Callable[[], T]): Function without parameters wrapping around the
                function call to evaluate and cache.
            path (str | Path): Path to the file to save the output to. The file
                extension determines how the output is saved. Supported formats
                are .npz, .npy, .pt, .csv, and .pkl.
            refresh (bool, optional): Flag whether the cache should be refreshed
                or not. Defaults to False.
            save_kwargs (dict[str, Any], optional): Keyword arguments passed to
                the saving function. Defaults to {}.
            load_kwargs (dict[str, Any], optional): Keyword arguments passed to
                the loading function. Defaults to {}.
            verbose (bool, optional): Flag whether to be verbose or not.
                Defaults to True.

        Returns:
            output (T): Output of the function call.
        """
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set. Please call init() first.")
        path = Path(self.cache_dir) / path

        if not path.is_file() or refresh:
            # evaluate function
            output = f()
            # save to cache
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npy":
                if not isinstance(output, np.ndarray):
                    raise ValueError(f"Output is not a np.ndarray: {type(output)}")
                np.save(path, output, **save_kwargs)
            elif path.suffix == ".npz":
                np.savez_compressed(path, **output, **save_kwargs)
            elif path.suffix == ".pt":
                torch.save(output, path, **save_kwargs)
            elif path.suffix == ".csv":
                if not isinstance(output, pd.DataFrame):
                    raise ValueError(f"Output is not a pd.DataFrame: {type(output)}")
                output.to_csv(path, **save_kwargs)
            elif path.suffix == ".pkl":
                with open(path, "wb") as file:
                    pickle.dump(
                        output,
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                        **save_kwargs,
                    )
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
                with open(path, "rb") as file:
                    output = pickle.load(file, **load_kwargs)
            else:
                raise ValueError(f"Unknown file format: {path.suffix}")
            if verbose:
                logger.info(f"Loaded from cache: {path}")
        return output  # type: ignore

    def get_path(self, path: str | Path) -> Path:
        """Get the full path to the cached file."""
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set. Please call init() first.")
        return self.cache_dir / path


CACHE = Cache()


def get_embeddings_folder(pipeline_name: str, model_name: str) -> Path:
    """Get the cache folder for the embeddings."""
    return Path("embeddings") / pipeline_name / model_name.replace("/", "__")


def save_embeddings(
    embeddings: Any,
    pipeline_name: str,
    model_name: str,
    file_name: str = "embeddings.npz",
    **kwargs,
):
    """Save embeddings from the cache."""
    path = get_embeddings_folder(pipeline_name, model_name) / file_name
    CACHE(lambda: embeddings, path, refresh=True, **kwargs)


def load_embeddings(
    pipeline_name: str,
    model_name: str,
    file_name: str = "embeddings.npz",
    **kwargs,
) -> Any:
    """Load embeddings from the cache."""
    path = get_embeddings_folder(pipeline_name, model_name) / file_name

    def raise_not_found():
        raise FileNotFoundError(f"Embeddings not found in cache: {path}")

    return CACHE(raise_not_found, path, **kwargs)
