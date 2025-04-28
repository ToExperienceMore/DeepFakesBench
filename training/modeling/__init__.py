from .efficient_vit import build_efficient_vit

_model_entrypoints = {
    # ... existing models ...
    'efficient_vit': build_efficient_vit,
} 