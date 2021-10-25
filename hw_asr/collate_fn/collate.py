import logging
from typing import List

from torch.nn.utils.rnn import pad_sequence
import torch
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {'text_encoded': [],
                    'text_encoded_length': [],
                    'spectrogram': [],
                    'spectrogram_length': [],
                    'text': [],
                    }

    for item in dataset_items:
        result_batch['text_encoded'].append(item['text_encoded'].squeeze(0))
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[1])

        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).transpose(0, 1))
        result_batch['spectrogram_length'].append(item['spectrogram'].shape[2])

        result_batch['text'].append(CharTextEncoder.normalize_text(item['text']))

    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], batch_first=True)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], batch_first=True)
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])

    return result_batch
