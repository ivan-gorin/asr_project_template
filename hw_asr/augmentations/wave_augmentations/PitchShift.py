import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sample_rate, p=0.2, mode="per_example", p_mode="per_example", *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate, p=p, mode=mode, p_mode=p_mode, *args,
                                                     **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
