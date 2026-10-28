from .custom_cnn import build_custom_cnn
from .mobilenet import build_mobilenetv2_classifier


def get_model(name: str, **kwargs):
    name_clean = name.lower().replace("-", "_")
    if name_clean in ("custom_cnn", "scratch_cnn", "cnn", "scratch"):
        return build_custom_cnn(**kwargs)
    elif name_clean in ("mobilenetv2", "mobilenet", "transfer"):
        return build_mobilenetv2_classifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{name}'. Supported: custom_cnn, mobilenetv2"
        )


__all__ = ["build_custom_cnn", "build_mobilenetv2_classifier", "get_model"]
