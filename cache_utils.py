import hashlib
import json, pickle
import os
from typing import Any
from datetime import datetime

# --- Configuration ---
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Cache Utilities ---
def make_cache_key(selected_tickers, start_date, end_date, target_return, target_volatility, min_weights, max_weights):
    # Sort tickers and weights to ensure deterministic key
    key_dict = {
        "tickers": sorted(selected_tickers),
        "start_date": start_date,
        "end_date": end_date,
        "target_return": target_return,
        "target_volatility": target_volatility,
        "min_weights": {k: min_weights[k] for k in sorted(min_weights)},
        "max_weights": {k: max_weights[k] for k in sorted(max_weights)}
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def save_cache(key: str, data: Any) -> None:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cache(key: str) -> Any:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def cache_exists(key: str) -> bool:
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if not os.path.exists(cache_path):
        return False
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    return (datetime.now() - cache_time).days < 10
