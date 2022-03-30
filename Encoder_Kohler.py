import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, d_model, h, d_k):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        # integer division
        self.d_v = d_model // h

        assert (
                self.d_v * h == d_model
        ), "Internal dimension of the model needs to be divisible by the number of attention heads"
        self.d_k = d_k
        self.values = nn.Linear(self.d_model, self.d_model, bias=False)
        self.keys = nn.Linear(self.d_model, self.d_k * self.h, bias=False)
        self.queries = nn.Linear(self.d_model, self.d_k * self.h, bias=False)
        # matrix rho. input and output of this layer have the same dimension, cf. above
        self.fc_out = nn.Linear(self.h * self.d_v, d_model, bias=False)

    def forward(self, Z, mask):
        # Z is a matrix of dimension n x l_max x d_model

        # Get number of training examples
        n = Z.shape[0]

        # Get the length of the input sequence
        l_max = Z.shape[1]

        values = self.values(Z)  # multiplication with W_V
        keys = self.keys(Z)  # multiplication with W_K
        queries = self.queries(Z)  # multiplication with W_Q

        # Split the embedding into h different pieces
        values = values.reshape(n, l_max, self.h, self.d_v)
        keys = keys.reshape(n, l_max, self.h, self.d_k)
        queries = queries.reshape(n, l_max, self.h, self.d_k)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example

        C = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (n, l_max, h, d_k),
        # keys shape: (n, l_max, h, d_k)
        # compatibility: (n, h, l_max, l_max)

        # ignore the padded values (we pad from l -> l_max) by setting them to -infty
        masked_C = C.masked_fill(mask == 0, float('-inf'))

        C_tilde = torch.zeros(C.shape).to(torch.device("cuda" \
                                                           if torch.cuda.is_available() else "cpu"))
        # Compute indices where row-wise maximum is attained
        dummy_index_1, dummy_index_2, dummy_index_3 = np.indices(masked_C.argmax(axis=3).shape)
        # Define C_tilde by setting (in each row) all but the largest entry to zero
        C_tilde[dummy_index_1, dummy_index_2, dummy_index_3, torch.argmax(masked_C, dim=3)] = \
            C[dummy_index_1, dummy_index_2, dummy_index_3, torch.argmax(masked_C, dim=3)]

        y_bar = torch.einsum("nhll,nlhd->nlhd", [C_tilde, values]).reshape(
            n, l_max, self.h * self.d_v
        )
        # attention shape: (n, h, l_max, l_max)
        # values shape: (n, l_max, h, d_v)
        # out after matrix multiply: (n, l, h, d_v), then
        # we reshape and flatten the last two dimensions.

        after_rho = self.fc_out(y_bar)
        # Linear layer doesn't modify the shape, final shape will be
        # (n, l_max, d_model)

        return after_rho


class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_k, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_model, h, d_k)

        # This is W_2 * ReLU(W_1* y + b_1) + b_2
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, mask):
        after_rho = self.attention(Z, mask)

        # Dropout then residual connection
        y = self.dropout(after_rho) + Z
        # Feedforward
        forward = self.feed_forward(y)
        # Dropout then residual connection
        Z_new = forward + self.dropout(y)
        return Z_new


class Encoder(nn.Module):
    def __init__(
            self,
            d_model,
            N,
            h,
            d_ff,
            d_k,
            dropout
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    h,
                    d_k,
                    d_ff,
                    dropout
                )
                for _ in range(N)
            ]
        )

    def forward(self, Z, mask):
        # Dropout before we start
        Z = self.dropout(Z)
        for layer in self.layers:
            Z = layer(Z, mask)
        return Z


class Transformer_Kohler(nn.Module):
    def __init__(
            self,
            d_model,
            N,
            d_k,
            d_ff,
            h,
            l_max,
            dropout
    ):
        super(Transformer_Kohler, self).__init__()

        self.d_model = d_model,
        self.N = N,
        self.d_k = d_k
        self.d_ff = d_ff,
        self.h = h,
        self.l_max = l_max,

        self.encoder = Encoder(
            d_model,
            N,
            h,
            d_ff,
            d_k,
            dropout
        )
        # Formula for number of parameters from thesis.
        # print((2 * d_k + d_model / h) * d_model * N * h +
        #      (d_model * d_model + 2 * d_ff * d_model + d_ff + d_model) * N + d_model * l_max + 1)
        self.pred = nn.Linear(d_model * l_max, 1)

    def forward(self, x, mask):
        encoder_output = self.encoder(x, mask)
        out = self.pred(encoder_output.reshape(encoder_output.shape[0],
                                               encoder_output.shape[1] * encoder_output.shape[2]))
        return out
