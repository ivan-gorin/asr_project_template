from operator import xor

from torch.utils.data import DataLoader, ConcatDataset

import hw_asr.augmentations
import hw_asr.batch_sampler as batch_sampler_module
import hw_asr.datasets
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser, text_encoder: BaseTextEncoder):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = hw_asr.augmentations.from_configs(configs)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, hw_asr.datasets, text_encoder=text_encoder, config_parser=configs,
                wave_augs=wave_augs, spec_augs=spec_augs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last, pin_memory=True
        )
        dataloaders[split] = dataloader
    return dataloaders
