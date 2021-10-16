from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


class BaselineGRU(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, n_layers=3, dropout=0, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.gru = nn.GRU(input_size=n_feats, hidden_size=fc_hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        self.net = Sequential(
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden // 2, out_features=fc_hidden // 4),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden // 4, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        out, _ = self.gru(spectrogram)
        return {"logits": self.net(out)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
