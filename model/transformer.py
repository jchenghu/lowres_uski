
import torch

from model.modules.generic_modules import PositionalEncoder, EmbeddingLayer
from model.modules.transformer_modules import FeedForward, MultiHeadAttention

from utils.masking_utils import create_pad_mask, create_no_peak_and_pad_mask

from model.seq2seq_model import Seq2Seq_Model
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_perc, eps=1e-6):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps=eps)  # one after the multihead
        self.norm_2 = nn.LayerNorm(d_model, eps=eps)  # another one after the feed forward
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.multi_head_attention(q=x2, k=x2, v=x2, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_perc, eps=1e-6):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps)
        self.norm_2 = nn.LayerNorm(d_model, eps)
        self.norm_3 = nn.LayerNorm(d_model, eps)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.multi_head_attention_1 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.multi_head_attention_2 = MultiHeadAttention(d_model, num_heads, dropout_perc)

        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, cross_connection_x, input_attention_mask, cross_attention_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.multi_head_attention_1(q=x2, k=x2, v=x2, mask=input_attention_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.multi_head_attention_2(q=x2, k=cross_connection_x, v=cross_connection_x,
                                                           mask=cross_attention_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class TransformerMT_SeparateVocab(Seq2Seq_Model):
    def __init__(self, d_model, N, num_heads, input_word2idx, output_word2idx, max_seq_len, d_ff, dropout_perc, rank=0):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.input_word2idx = input_word2idx
        self.output_word2idx = output_word2idx
        self.max_seq_len = max_seq_len

        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_perc) for _ in range(N)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_perc) for _ in range(N)])

        self.linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.src_embedder = EmbeddingLayer(len(input_word2idx), d_model, dropout_perc)
        self.trg_embedder = EmbeddingLayer(len(output_word2idx), d_model, dropout_perc)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len, rank)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads, is_pretraining=False):
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), enc_input.size(1), enc_input.size(1)),
                                   pad_along_row_input=enc_input_num_pads,
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        x = self.src_embedder(enc_input)
        x = x + self.positional_encoder(x)

        for i in range(self.N):
            x = self.encoders[i](x=x, mask=pad_mask)
        return x

    def forward_dec(self, cross_input, enc_input_num_pads,
                    dec_input, dec_input_num_pads,
                    apply_log_softmax=False, is_pretraining=False):
        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
            mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
            num_pads=dec_input_num_pads,
            rank=self.rank)

        pad_mask = create_pad_mask(mask_size=(cross_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_along_row_input=dec_input_num_pads,
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        y = self.trg_embedder(dec_input)
        y = y + self.positional_encoder(y)

        for i in range(self.N):
            y = self.decoders[i](x=y,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask)

        y = self.linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y


class TransformerMT(Seq2Seq_Model):
    def __init__(self, d_model, N, num_heads, shared_word2idx, max_seq_len, d_ff, dropout_perc, rank=0):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.shared_word2idx = shared_word2idx
        self.max_seq_len = max_seq_len

        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_perc) for _ in range(N)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_perc) for _ in range(N)])

        self.linear = torch.nn.Linear(d_model, len(shared_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.shared_embedder = EmbeddingLayer(len(shared_word2idx), d_model, dropout_perc)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len, rank)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads, is_pretraining=False):
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), enc_input.size(1), enc_input.size(1)),
                                   pad_along_row_input=enc_input_num_pads,
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        x = self.shared_embedder(enc_input)
        x = x + self.positional_encoder(x)

        for i in range(self.N):
            x = self.encoders[i](x=x, mask=pad_mask)
        return x

    def forward_dec(self, cross_input, enc_input_num_pads,
                    dec_input, dec_input_num_pads,
                    apply_log_softmax=False, is_pretraining=False):
        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
            mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
            num_pads=dec_input_num_pads,
            rank=self.rank)

        pad_mask = create_pad_mask(mask_size=(cross_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_along_row_input=dec_input_num_pads,
                                   pad_along_column_input=enc_input_num_pads,
                                   rank=self.rank)

        y = self.shared_embedder(dec_input)
        y = y + self.positional_encoder(y)

        for i in range(self.N):
            y = self.decoders[i](x=y,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask)

        y = self.linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y