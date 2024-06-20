
import math
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_perc):
        super().__init__()
        self.dropout = nn.Dropout(dropout_perc)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.dropout(self.embed(x))


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, rank=0):
        super().__init__()
        assert d_model % 2 == 0, "d_model is not even, even number suggested"
        self.d_model = d_model
        self.pe = torch.zeros(max_seq_len, d_model).to(rank)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe.data[pos, i] = math.sin(pos / (10000.0 ** ((2.0 * i) / d_model)))
                self.pe.data[pos, i + 1] = math.cos(pos / (10000.0 ** ((2.0 * i) / d_model)))
        self.pe.data = self.pe.data.unsqueeze(0)

    # x shape [ batch_size, seq_len, d_model]
    def forward(self, x):
        seq_len = x.shape[1]
        # we apply this to each row of the batch, it's automatically
        # broadcasted all along batches thanks to the pe = pe.unsqueeze(0)
        return self.pe.data[0, :seq_len]
