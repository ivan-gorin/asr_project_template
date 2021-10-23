import torch_audiomentations
from torch import Tensor
from pathlib import Path
import torchaudio

from hw_asr.augmentations.rirs_noises_load import load_noise_data
from hw_asr.augmentations.base import AugmentationBase


class AddBackgroundNoise(AugmentationBase):
    def __init__(self, sample_rate, data_dir=None, p=0.2, mode="per_example", p_mode="per_example", *args, **kwargs):
        if data_dir is None:
            self._data_dir = load_noise_data() / "pointsource_noises"
        else:
            data_dir = Path(data_dir)
            self._data_dir = data_dir
        self._aug = torch_audiomentations.AddBackgroundNoise(background_paths=self._data_dir, sample_rate=sample_rate,
                                                             p=p, mode=mode, p_mode=p_mode, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
