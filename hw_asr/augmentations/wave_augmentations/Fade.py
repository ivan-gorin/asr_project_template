import torchaudio
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Fade(AugmentationBase):
    def __init__(self, p=0.5, fade_shape="linear"):
        self._aug = RandomApply(torchaudio.transforms.Fade(fade_shape=fade_shape), p=p)

    def __call__(self, data: Tensor):
        length = data.shape[-1]
        self._aug.augmentation.fade_in_len = random.randint(0, length // 2)
        self._aug.augmentation.fade_out_len = random.randint(0, length // 2)
        return self._aug(data)
