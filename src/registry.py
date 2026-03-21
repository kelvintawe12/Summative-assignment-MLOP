import json
import os
import logging

REGISTRY_PATH = "models/model_registry.json"
logger = logging.getLogger(__name__)

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {"champion": None, "history": []}

def save_registry(registry):
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

def register_model(model_path, metrics, is_champion=False):
    registry = load_registry()
    entry = {
        "path": model_path,
        "metrics": metrics,
        "timestamp": os.path.getctime(model_path) if os.path.exists(model_path) else None
    }
    registry["history"].append(entry)
    if is_champion:
        registry["champion"] = entry
    save_registry(registry)
    return registry

def get_champion_metrics():
    registry = load_registry()
    if registry["champion"]:
        return registry["champion"]["metrics"]
    return None
