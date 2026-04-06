from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Callable, Dict, Optional

import grain
import datasets


def _is_random_access_dataset(obj: Any) -> bool:
    """Return True if obj looks like a random-access dataset."""
    return hasattr(obj, "__len__") and hasattr(obj, "__getitem__")


def _is_hf_iterable_dataset(obj: Any) -> bool:
    """
    Best-effort check for Hugging Face IterableDataset
    without importing datasets eagerly.
    """
    cls = obj.__class__
    return (
        cls.__name__ == "IterableDataset"
        and cls.__module__.startswith("datasets")
    )


class GeneralRandomAccessSource(grain.sources.RandomAccessDataSource):
    """
    Grain random-access source that wraps many Python/HF-style datasets.

    Supported inputs:
    - list / tuple
    - huggingface datasets.Dataset
    - any object implementing __len__ and __getitem__
    - dict / mapping:
        - default: converted to list[dict(key=..., value=...)]
        - or custom adapter via adapter=

    Notes:
    - This is intended for Grain MapDataset.source(...)
    - If your input is streaming / iterable-only, use GeneralIterDataset instead.
    """

    def __init__(
        self,
        data: Any,
        *,
        adapter: Callable[[Any], Any] | None = None,
        dict_mode: str = "items",
    ):
        """
        Args:
            data:
                Input data source.
            adapter:
                Optional callable applied to each raw sample before returning it.
            dict_mode:
                Only used when `data` is a mapping.
                - "items": each sample becomes {"key": k, "value": v}
                - "values": dataset consists of mapping values only
                - "keys": dataset consists of mapping keys only
        """
        self._adapter = adapter

        if isinstance(data, Mapping):
            if dict_mode == "items":
                self._data = [{"key": k, "value": v} for k, v in data.items()]
            elif dict_mode == "values":
                self._data = list(data.values())
            elif dict_mode == "keys":
                self._data = list(data.keys())
            else:
                raise ValueError(
                    "dict_mode must be one of {'items', 'values', 'keys'}"
                )
        elif _is_random_access_dataset(data):
            self._data = data
        else:
            raise TypeError(
                "GeneralRandomAccessSource requires a random-access object "
                "(must implement __len__ and __getitem__)."
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        item = self._data[index]
        if self._adapter is not None:
            item = self._adapter(item)
        return item

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self)})"


class _GeneralIterator(grain.DatasetIterator):
    """
    Checkpointable iterator for iterable-only sources.

    Strategy:
    - recreate the iterable from a factory
    - skip until saved position
    """

    def __init__(self, iterable_factory: Callable[[], Iterable[Any]], start_index: int = 0):
        super().__init__()
        self._iterable_factory = iterable_factory
        self._index = 0
        self._closed = False

        self._iter = iter(self._iterable_factory())
        for _ in range(start_index):
            try:
                next(self._iter)
                self._index += 1
            except StopIteration:
                break

    def __next__(self) -> Any:
        if self._closed:
            raise ValueError("Iterator is closed.")
        item = next(self._iter)
        self._index += 1
        return item

    def get_state(self) -> dict[str, Any]:
        return {"index": self._index}

    def set_state(self, state: dict[str, Any]) -> None:
        self._index = 0
        start_index = int(state["index"])
        self._iter = iter(self._iterable_factory())
        for _ in range(start_index):
            try:
                next(self._iter)
                self._index += 1
            except StopIteration:
                break

    def close(self) -> None:
        self._closed = True
        super().close()


class GeneralIterDataset(grain.IterDataset):
    """
    Grain IterDataset wrapper for iterable-only data.

    Supported inputs:
    - huggingface datasets.IterableDataset
    - generators
    - iterators
    - arbitrary iterable objects

    Notes:
    - This is for data without efficient random access.
    - Checkpoint restore works by replaying from the beginning and skipping items,
      so it is OK for simple streams but not ideal for huge expensive streams.
    """

    def __init__(
        self,
        data: Iterable[Any] | Callable[[], Iterable[Any]],
        *,
        adapter: Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        self._adapter = adapter

        if callable(data):
            self._factory = data
        else:
            # Important:
            # If user passes an iterator object directly, it is one-shot.
            # So we materialize a reusable factory when possible.
            if isinstance(data, Iterator):
                cached = list(data)
                self._factory = lambda: iter(cached)
            else:
                self._factory = lambda: iter(data)

    def __iter__(self) -> grain.DatasetIterator:
        adapter = self._adapter
        factory = self._factory

        if adapter is None:
            return _GeneralIterator(factory)

        def adapted_factory():
            for x in factory():
                yield adapter(x)

        return _GeneralIterator(adapted_factory)


def grain_from_source(
    data: Any,
    *,
    adapter: Callable[[Any], Any] | None = None,
    preprocess_fn: Optional[Callable] = None,
    dict_mode: str = "items",
    read_opts: grain.ReadOptions | None = None,
    batch_size: int | None = None,
    seed: int = 42,
):
    """
    Convert many common Python/HF data containers into a Grain pipeline.

    Behavior:
        - Mapping / random-access dataset -> MapDataset
        - iterable-only dataset -> IterDataset
        - shuffle is applied only to map-style datasets
        - if map-style, convert to IterDataset via `.to_iter_dataset(read_opts)`
        - if batch_size is given, apply `.batch(batch_size)`

    Returns:
        A Grain dataset pipeline, usually IterDataset if batching / reading is applied.
    """
    is_map = False

    if isinstance(data, Dict):
        data = datasets.Dataset.from_dict(data)

    if isinstance(data, Mapping):
        source = GeneralRandomAccessSource(
            data,
            adapter=adapter,
            dict_mode=dict_mode,
        )
        pipeline = grain.MapDataset.source(source)
        is_map = True

    elif _is_random_access_dataset(data) and not _is_hf_iterable_dataset(data):
        source = GeneralRandomAccessSource(data, adapter=adapter)
        pipeline = grain.MapDataset.source(source)
        is_map = True

    elif isinstance(data, Iterable) or _is_hf_iterable_dataset(data):
        pipeline = GeneralIterDataset(data, adapter=adapter)

    else:
        raise TypeError(
            "Unsupported data type. Expected mapping, random-access dataset, or iterable."
        )

    if is_map:
        pipeline = pipeline.shuffle(seed=seed)
        pipeline = (
            pipeline.to_iter_dataset(read_opts)
            if read_opts is not None
            else pipeline.to_iter_dataset()
        )
    if batch_size is not None:
        pipeline = pipeline.batch(batch_size)

    return pipeline
