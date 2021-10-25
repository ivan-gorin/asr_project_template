from typing import List
from itertools import groupby
from torch import Tensor
import youtokentome

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class BPEEncoder(CharTextEncoder):
    EMPTY_TOK = "<PAD>"

    def __init__(self, alphabet: List[str], model):
        super().__init__(alphabet)
        self.model = model
        self.ind2char = {model.subword_to_id(token): token for token in alphabet}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    @classmethod
    def get_simple_alphabet(cls, args):
        model = youtokentome.BPE(model=args['model_path'])
        return cls(alphabet=model.vocab(), model=model)

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.model.encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}'")

    def ctc_decode(self, inds: List[int]) -> str:
        inds = [int(x[0]) for x in groupby(inds)]
        res = self.model.decode(inds, ignore_ids=[0, 1, 2, 3])[0]
        return res
