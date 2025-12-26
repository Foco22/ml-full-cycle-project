"""
Configuration loader utility
"""

import yaml
import json
import os
import re
from typing import Dict, Any


def expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in config values.

    Replaces patterns like ${VAR_NAME} with their environment variable values.

    Args:
        config: Configuration value (dict, list, str, or other)

    Returns:
        Config with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable value
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, lambda m: os.getenv(m.group(1), m.group(0)), config)
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file and expand environment variables.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary with environment variables expanded
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")

    # Expand environment variables in the loaded config
    config = expand_env_vars(config)

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML or JSON file

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif output_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {output_path}")
