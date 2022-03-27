''' Define the Layers '''
import torch.nn as nn
import torch
from models.archs.transformer.SubLayers import MultiHeadAttention4, PositionwiseFeedForward4


__author__ = "Yu-Hsiang Huang"

class EncoderLayer3(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer3, self).__init__()
        self.slf_attn = MultiHeadAttention4(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward4(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

