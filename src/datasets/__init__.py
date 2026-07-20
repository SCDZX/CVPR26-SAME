# WILDS
# CIFAR
from .cifar10 import CIFAR101, CIFAR102

# Small
from .collections import (
    CIFAR10,
    CIFAR100,
    DTD,
    MNIST,
    SUN397,
    Aircraft,
    Caltech101,
    EuroSAT,
    Flowers,
    Food,
    OxfordPet,
    StanfordCars,
)
try:
    from .fmow import FMOW, FMOWID, FMOWOOD
except ModuleNotFoundError as exc:
    if exc.name != "wilds":
        raise

    class FMOW:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FMOW requires the optional package wilds. "
                "It is not needed for the MTIL split2 workflow."
            ) from exc

    class FMOWID(FMOW):
        pass

    class FMOWOOD(FMOW):
        pass

# ImageNet
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_small import ImageNetSM
from .imagenet_sub import ImageNetSUB
from .imagenet_subclass import ImageNetSC
from .imagenet_vid_robust import ImageNetVidRobust
try:
    from .imagenetv2 import ImageNetV2
except ModuleNotFoundError as exc:
    if exc.name != "imagenetv2_pytorch":
        raise

    class ImageNetV2(ImageNet):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ImageNetV2 requires the optional package imagenetv2_pytorch. "
                "It is not needed for the MTIL split2 workflow."
            ) from exc
try:
    from .iwildcam import (
        IWildCam,
        IWildCamID,
        IWildCamIDNonEmpty,
        IWildCamOOD,
        IWildCamOODNonEmpty,
    )
except ModuleNotFoundError as exc:
    if exc.name != "wilds":
        raise

    class IWildCam:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "IWildCam requires the optional package wilds. "
                "It is not needed for the MTIL split2 workflow."
            ) from exc

    class IWildCamID(IWildCam):
        pass

    class IWildCamIDNonEmpty(IWildCam):
        pass

    class IWildCamOOD(IWildCam):
        pass

    class IWildCamOODNonEmpty(IWildCam):
        pass

from .joint import Joint

# Random Noise
from .noise import Noise
from .objectnet import ObjectNet
from .ytbb_robust import YTBBRobust

# Experimental datasets
dataset_list = [
    Aircraft,
    Caltech101,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
]
mtil_split2_base_dataset_names = [
    "Aircraft",
    "Caltech101",
    "Food",
    "MNIST",
    "OxfordPet",
    "Flowers",
    "SUN397",
    "DTD",
    "EuroSAT",
    "CIFAR100",
    "StanfordCars",
]
mtil_split2_dataset_names = [
    f"{name}_{part}_2"
    for part in (1, 2)
    for name in mtil_split2_base_dataset_names
]

# Expose MTIL split2 dataset classes such as Aircraft_1_2 and Aircraft_2_2.
from . import collections as _collections

for _dataset in dataset_list:
    for _part in (1, 2):
        _name = f"{_dataset.__name__}_{_part}_2"
        globals()[_name] = getattr(_collections, _name)


def show_datasets():
    print("Total: ", len(dataset_list))
    print("Dataset: (train_len, test_len, num_classes)")
    for dataset in dataset_list:
        d = dataset(None)
        print(f"{d.name}: ", d.stats())
        for i in range(3):
            print(f"T[{i}]: ", d.template(d.classnames[i]))
        

from .cc import conceptual_captions
