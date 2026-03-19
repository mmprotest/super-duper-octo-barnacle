from __future__ import annotations

from collections import OrderedDict
from typing import Hashable


class TranslationCache:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._data: OrderedDict[Hashable, str] = OrderedDict()

    def get(self, key: Hashable) -> str | None:
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def set(self, key: Hashable, value: str) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
