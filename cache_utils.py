import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


class FileCache:
    def __init__(self, cache_dir: str = "cache", expiry_days: int = 10):
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        os.makedirs(self.cache_dir, exist_ok=True)

    def make_cache_key(
        self,
        selected_tickers: List[str],
        start_date: str,
        end_date: str,
        target_return: float,
        target_volatility: float,
        min_weights: Dict[str, float],
        max_weights: Dict[str, float],
    ) -> str:
        key_dict = {
            "tickers": sorted(selected_tickers),
            "start_date": start_date,
            "end_date": end_date,
            "target_return": target_return,
            "target_volatility": target_volatility,
            "min_weights": {k: min_weights[k] for k in sorted(min_weights)},
            "max_weights": {k: max_weights[k] for k in sorted(max_weights)},
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def save(self, key: str, data_bytes: bytes) -> None:
        with open(self._get_cache_path(key), "wb") as f:
            f.write(data_bytes)

    def load(self, key: str) -> Optional[bytes]:
        try:
            with open(self._get_cache_path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def exists(self, key: str) -> bool:
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return False
        cache_time = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - cache_time < timedelta(days=self.expiry_days)
