"""
Pure PyTorch modules without any PyTorch Lightning.
"""

import torch
from torch import LongTensor, BoolTensor, Tensor, FloatTensor
from torch import nn

from single_step_retro.src.model.common_modules import PositionalEncoding, TokenEmbedding


class VanillaTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            embedding_dim: int = 128,
            num_heads: int = 4,
            feedforward_dim: int = 256,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            src_pad_token_idx: int = 0,
            tgt_pad_token_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.src_pad_token_i = src_pad_token_idx
        self.tgt_pad_token_i = tgt_pad_token_idx

        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers
        self.emb_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Embedding constructor
        self.token_embedding = TokenEmbedding(
            self.vocab_size, self.emb_dim, padding_idx=self.src_pad_token_i
        )
        self.positional_encoding = PositionalEncoding(self.emb_dim)

        # Embedding updater
        layer_norm_eps = 1e-5
        batch_first = True
        norm_first = False
        self.transformer = nn.Transformer(
            d_model=self.emb_dim,
            nhead=self.num_heads,
            batch_first=batch_first,
            custom_encoder=nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                    norm_first=norm_first,
                ),
                self.num_enc_layers,
                nn.LayerNorm(self.emb_dim, eps=layer_norm_eps),
            ),
            custom_decoder=nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                    norm_first=norm_first,
                ),
                self.num_dec_layers,
                nn.LayerNorm(self.emb_dim, eps=layer_norm_eps),
            ),
        )

        # Decision function
        self.next_token_classifier = nn.Linear(
            self.emb_dim, self.vocab_size, bias=False
        )
        self.next_token_classifier.weight = self.token_embedding.embedding.weight

    def forward(self, src_token_ids: LongTensor, tgt_token_ids: LongTensor):
        """
        Calculates the target decoder output embeddings
        Args:
            src_token_ids (LongTensor of size (b_sz, src_seq_len)): the token indices of the source sequences
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences

        Returns:
            logits (FloatTensor of size (b_sz, tgt_seq_len, vocab_size)): the model output logits
        """

        tgt_emb = self.get_decoder_output_embs(src_token_ids, tgt_token_ids)

        logits = self.next_token_classifier(tgt_emb)
        return logits

    def encode_src(self, src_token_ids: LongTensor, src_pad_mask: BoolTensor):
        # Embed tokens
        src_emb = self.positional_encoding(self.token_embedding(src_token_ids))

        # Update embeddings
        src_emb = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return src_emb

    def decode_tgt(
            self,
            tgt: LongTensor,
            memory: Tensor,
            memory_pad_mask: BoolTensor
    ):

        tgt_emb = self.get_decoder_output_embs_using_src_memory(tgt, memory, memory_pad_mask)

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits

    def get_decoder_output_embs(self, src_token_ids: LongTensor, tgt_token_ids: LongTensor):
        """
        Calculates the target decoder output embeddings
        Args:
            src_token_ids (LongTensor of size (b_sz, src_seq_len)): the token indices of the source sequences
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences

        Returns:
            tgt_emb (FloatTensor of size (b_sz, tgt_seq_len, emb_dim)): the decoder output
        """
        src_pad_mask: torch.Tensor = torch.where(
            src_token_ids != self.src_pad_token_i, torch.tensor(0.0), torch.tensor(float("-inf"))
        )
        memory = self.encode_src(src_token_ids, src_pad_mask)
        tgt_emb = self.get_decoder_output_embs_using_src_memory(tgt_token_ids, memory, src_pad_mask)
        return tgt_emb

    def get_decoder_output_embs_using_src_memory(self,
                                                 tgt_token_ids: LongTensor,
                                                 memory: Tensor,
                                                 memory_pad_mask: BoolTensor
                                                 ) -> FloatTensor:
        """
        Calculates the target decoder output embeddings getting the source encoder output embeddings and
        the corresponding target indices
        Args:
            tgt_token_ids (LongTensor of size (b_sz, tgt_seq_len)): the token indices of the target sequences
            memory (FloatTensor of size (b_sz, src_seq_len, emb_dim)): the token indices of the target sequences
            memory_pad_mask (BoolTensor of size (b_sz, src_seq_len)): the token indices of the target sequences

        Returns:
            tgt_emb (FloatTensor of size (b_sz, tgt_seq_len, emb_dim)): the decoder output
        """

        # Embed tokens
        tgt_emb = self.positional_encoding(self.token_embedding(tgt_token_ids))

        # Update embeddings
        tgt_pad_mask = torch.where(tgt_token_ids != self.tgt_pad_token_i, torch.tensor(0.0), torch.tensor(float("-inf")))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_token_ids.shape[1]).type_as(tgt_emb)
        tgt_emb = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return tgt_emb
