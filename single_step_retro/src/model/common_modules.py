"""
Initial embedding layers for tokens.
Supposed to be a part of any model.
"""

import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """
    Embedding of token indices into a vector space.
    """

    def __init__(self, vocab_size: int, emb_size: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        return self.embedding(tokens)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        """
        Absolute positional encoding.
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerDecoderWithExtraPositionalEncoding(nn.TransformerDecoder):
    """
    A modification of the basic TransformerDecoder.
    Here, positional encodings are added to the target embeddings in every decoder layer.
    """

    def __init__(self, *args, pos_encoder: PositionalEncoding, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoder = pos_encoder

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = self.pos_encoder(output)
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
