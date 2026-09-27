"""Where a run executed: recorded in every config.json (runs moved from Colab to Kaggle mid-study, D32)."""
import os
import torch


def platform():
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/working"):
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content/sample_data"):
        return "colab"
    return "local"


def info():
    return dict(platform=platform(), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None)
