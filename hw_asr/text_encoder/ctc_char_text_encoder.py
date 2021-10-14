from typing import List, Tuple
from itertools import groupby
from ctcdecode import CTCBeamDecoder

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        inds = [x[0] for x in groupby(inds)]
        res = ""
        for ind in inds:
            if ind:
                res += self.ind2char[ind]
        return res

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        decoder = CTCBeamDecoder(list(self.ind2char.values()), beam_width=beam_size)
        beam_results, beam_scores, _, out_lens = decoder.decode(probs.unsqueeze(0))
        hypos = []
        for i in range(beam_size):
            hypos.append((self.ctc_decode(beam_results[0][i][:out_lens[0][i]].tolist()),
                          1/torch.exp(beam_scores[0][i]).item()))

        return hypos
        # return sorted(hypos, key=lambda x: x[1], reverse=True)
