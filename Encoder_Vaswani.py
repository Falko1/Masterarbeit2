import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        # integer division
        self.d_int = d_model // h

        assert (
                self.d_int * h == d_model
        ), "Internal dimension of the model needs to be divisible by the number of attention heads"
        self.values = nn.Linear(self.d_model, self.d_model, bias=False)
        self.keys = nn.Linear(self.d_model, self.d_model, bias=False)
        self.queries = nn.Linear(self.d_model, self.d_model, bias=False)
        # matrix rho. input and output of this layer have the same dimension, cf. above
        self.fc_out = nn.Linear(self.h * self.d_int, d_model, bias=False)

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
        values = values.reshape(n, l_max, self.h, self.d_int)
        keys = keys.reshape(n, l_max, self.h, self.d_int)
        queries = queries.reshape(n, l_max, self.h, self.d_int)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example

        C = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (n, l_max, h, d_int),
        # keys shape: (n, l_max, h, d_int)
        # compatibility (C) shape: (n, h, l_max, l_max)

        # ignore the padded values (we pad from l -> l_max) by setting them to -infty
        masked_C = C.masked_fill(mask == 0, float('-inf'))

        # Softmax with scaling instead of max
        C_tilde = torch.softmax(masked_C / (self.d_int ** (1 / 2)), dim=3)

        y_bar = torch.einsum("nhll,nlhd->nlhd", [C_tilde, values]).reshape(
            n, l_max, self.h * self.d_int
        )
        # attention shape: (n, h, l_max, l_max)
        # values shape: (n, l_max, h, d_int)
        # out after matrix multiply: (n, l, h, d_int), then
        # we reshape and flatten the last two dimensions.

        after_rho = self.fc_out(y_bar)
        # Linear layer doesn't modify the shape, final shape will be
        # (n, l_max, d_model)

        return after_rho


class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # This is W_2 * ReLU(W_1* y + b_1) + b_2
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, mask):
        after_rho = self.attention(Z, mask)

        # Dropout, residual connection then layer norm
        y = self.norm1(self.dropout(after_rho) + Z)
        # Feedforward
        forward = self.feed_forward(y)
        # Dropout, residual connection then layer norm
        Z_new = self.norm2(self.dropout(forward) + y)
        return Z_new


class Encoder(nn.Module):
    def __init__(
            self,
            d_model,
            N,
            h,
            d_ff,
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
                    d_ff,
                    dropout
                )
                for _ in range(N)
            ]
        )

    def forward(self, X, mask):
        # Dropout before we start
        Z = self.dropout(X)
        for layer in self.layers:
            Z = layer(Z, mask)
        return Z


class Transformer(nn.Module):
    def __init__(
            self,
            d_model,
            N,
            d_ff,
            h,
            l_max,
            dropout,
    ):
        super(Transformer, self).__init__()

        self.d_model = d_model,
        self.N = N,
        self.d_ff = d_ff,
        self.h = h,
        self.l_max = l_max,

        self.encoder = Encoder(
            d_model,
            N,
            h,
            d_ff,
            dropout
        )

        self.pred = nn.Linear(d_model * l_max, 1)

    def forward(self, x, mask):
        encoder_output = self.encoder(x, mask)
        clf_output = encoder_output.reshape(encoder_output.shape[0],
                                            encoder_output.shape[1] * encoder_output.shape[2])
        out = self.pred(clf_output)
        return out
