
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_perc):
        super().__init__()
        assert d_model % num_heads == 0, "num heads must be multiple of d_model"

        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, self.d_k * num_heads)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads)
        self.Wv = nn.Linear(d_model, self.d_k * num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)  # take the last dimension

        self.value_dropout = nn.Dropout(dropout_perc)
        self.attention_dropout = nn.Dropout(dropout_perc)

    # q,k,v shape: [batch_size, seq_len, vector_size]
    def forward(self, q, k, v, mask, return_attention_matrix=False):
        batch_size, q_seq_len, _ = q.shape
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        k_proj = self.Wk(k).view(batch_size, k_seq_len, self.num_heads, self.d_k)
        q_proj = self.Wq(q).view(batch_size, q_seq_len, self.num_heads, self.d_k)
        v_proj = self.Wv(v).view(batch_size, v_seq_len, self.num_heads, self.d_k)

        k_proj = k_proj.transpose(2, 1)
        q_proj = q_proj.transpose(2, 1)
        v_proj = v_proj.transpose(2, 1)

        v_proj = self.value_dropout(v_proj)

        # output: [ batch_size, num_head, q_seq_len, k_seq_len ]
        sim_scores = torch.matmul(q_proj, k_proj.transpose(3, 2))
        sim_scores = sim_scores / self.d_k ** 0.5

        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        sim_scores = sim_scores.masked_fill(mask == 0, value=-np.inf)  # -np.inf mi da' "NaN"!
        sim_scores = self.softmax(input=sim_scores)  #, dim=-1)
        # output: [ batch_size, num_heads, q_seq_len, k_seq_len ]

        sim_scores = sim_scores.masked_fill(mask == 0, value=0)
        sim_scores = self.attention_dropout(sim_scores)

        attention_applied = torch.matmul(sim_scores, v_proj)
        # output: [ batch_size, num_heads, q_seq_len, d_k ]

        # [ batch_size,  num_heads, seq_len, d_k ] -> [ batch_size, seq_len, d_model ]
        attention_applied_concatenated = attention_applied.permute(0, 2, 1, 3).contiguous()\
            .view(batch_size, q_seq_len, self.d_model)

        out = self.out_linear(attention_applied_concatenated)

        if return_attention_matrix:
            return out, sim_scores

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_perc)
        self.linear_2 = nn.Linear(d_ff, d_model)

    # x : [ batch_size, seq_len, vector_size = d_model]
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x