# src/config.py
"""Load configuration from YAML file."""
import yaml
import os

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)