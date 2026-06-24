"""
Configuration loader for HOPE pipeline.
Loads variant parameters from pipeline_config.json, supporting --variant CLI args.
"""
import os
import json
import inspect


def load_config(config_path=None):
    """Load pipeline_config.json from multiple fallback locations.
    
    Search order:
        1. Explicit config_path argument
        2. PIPELINE_CONFIG_PATH environment variable
        3. ./pipeline_config.json (CWD)
        4. Same directory as the calling script
    
    Returns:
        dict: Parsed JSON config, or empty dict if not found.
    """
    candidates = []
    
    # 1. Explicit path
    if config_path:
        candidates.append(config_path)
    
    # 2. Environment variable
    env_path = os.environ.get("PIPELINE_CONFIG_PATH")
    if env_path:
        candidates.append(env_path)
    
    # 3. CWD
    candidates.append(os.path.join(os.getcwd(), "pipeline_config.json"))
    
    # 4. Same directory as the calling script
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    candidates.append(os.path.join(caller_dir, "pipeline_config.json"))
    
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    
    return {}


def get_variant_params(variant_key, config=None):
    """Return the parameter dict for a given variant key.
    
    Args:
        variant_key: The variant key string (e.g., "loss_full_3c", "ema_0.9").
        config: Optional pre-loaded config dict. If None, load_config() is called.
    
    Returns:
        dict: The variant's parameter dictionary.
    
    Raises:
        KeyError: If variant_key is not found in config["variants"].
    """
    if config is None:
        config = load_config()
    
    variants = config.get("variants", {})
    if variant_key not in variants:
        available = list(variants.keys())
        raise KeyError(
            f"Variant '{variant_key}' not found in config. "
            f"Available variants: {available}"
        )
    
    return variants[variant_key]


def resolve_variant_args(args, param_keys=None):
    """Resolve --variant CLI arg by loading params from config and overriding args.
    
    If args.variant is set, loads the variant config and overrides the specified
    parameter keys on the args namespace. If args.variant is None, args is
    returned unchanged (backward compatible).
    
    Args:
        args: argparse.Namespace with an optional 'variant' attribute.
        param_keys: List of parameter names to override from config.
                    If None, defaults to common keys:
                    ['loss_type', 'num_classes', 'triplet_margin', 'm']
    
    Returns:
        argparse.Namespace: The (possibly modified) args.
    """
    variant_key = getattr(args, 'variant', None)
    if variant_key is None:
        return args
    
    if param_keys is None:
        param_keys = ['loss_type', 'num_classes', 'triplet_margin', 'm']
    
    config_path = getattr(args, 'config', None)
    params = get_variant_params(variant_key, config=load_config(config_path))
    
    # Map config keys to argparse attribute names
    # Config may use either 'target_loss' or 'loss_type'
    key_aliases = {
        'loss_type': ['loss_type', 'target_loss'],
        'target_loss': ['target_loss', 'loss_type'],
        'num_classes': ['num_classes', 'class_num'],
    }
    
    for key in param_keys:
        # Try the key directly, then aliases
        value = params.get(key)
        if value is None and key in key_aliases:
            for alias in key_aliases[key]:
                value = params.get(alias)
                if value is not None:
                    break
        
        if value is not None:
            setattr(args, key, value)
    
    return args
