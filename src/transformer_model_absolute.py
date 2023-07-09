import copy
import math
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, init_atoms, init_residues, d_encoder: int = 512, d_decoder: int = 512, d_model: int = 512, 
                 nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_transform_layer = LinearTransformLayer(d_encoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
                                                batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.decoder_transform_layer = LinearTransformLayer(d_decoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
                                                batch_first, norm_first, **factory_kwargs)
        affinity_output_layer = AffinityOutputLayer(d_model, dropout, activation, **factory_kwargs)
        pairwise_output_layer = PairwiseOutputLayer(d_model, dropout, activation, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, affinity_output_layer, 
                                          pairwise_output_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first
        self.compound_embedding = nn.Embedding.from_pretrained(init_atoms)
        self.protein_embedding = nn.Embedding.from_pretrained(init_residues)

    def forward(self, src: Tensor, tgt: Tensor, pids = None, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
              
        embed_src = self.protein_embedding_block(src)
        pos_src = self.pos_encoder(embed_src)
        memory = self.encoder(pos_src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        
        embed_tgt = self.compound_embedding_block(tgt)
        output = self.decoder(embed_tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def protein_embedding_block(self, residues):
        x1 = self.protein_embedding(residues)
        x2 = self.encoder_transform_layer(x1)
        return x2
    
    def compound_embedding_block(self, atoms):
        y1 = self.compound_embedding(atoms)
        y2 = self.decoder_transform_layer(y1)
        return y2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2050):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.transpose(0,1)
        return x

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for mod in self.layers:
                output = mod(output, src_mask=mask,
                             src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, affinity_output_layer, pairwise_output_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.affinity_output_layer = affinity_output_layer
        self.pairwise_output_layer = pairwise_output_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        affinity_output = self.affinity_output_layer(output)
        pairwise_output = self.pairwise_output_layer(output, memory, tgt_key_padding_mask, memory_key_padding_mask)

        return affinity_output, pairwise_output

class PairwiseOutputLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PairwiseOutputLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear_compound = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear_protein = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: Tensor, y: Tensor, c_bool_mask: Tensor, p_bool_mask: Tensor) -> Tensor:
        # need dropout here?
        # x = self.dropout(self.activation(self.linear_compound(x)))
        # y = self.dropout(self.activation(self.linear_protein(y)))
        x = self.activation(self.linear_compound(x))
        y = self.activation(self.linear_protein(y))
        pairwise_pred = torch.sigmoid(torch.matmul(x, y.transpose(1,2)))
        pairwise_mask = torch.matmul(torch.unsqueeze(1 - c_bool_mask.float(), 2), torch.unsqueeze(1 - p_bool_mask.float(), 1))
        pairwise_pred = pairwise_pred * pairwise_mask

        return pairwise_pred

class AffinityOutputLayer(nn.Module):
    def __init__(self, d_input: int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AffinityOutputLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_input, d_input, **factory_kwargs)
        self.linear2 = nn.Linear(d_input, d_input // 4, **factory_kwargs)
        self.linear3 = nn.Linear(d_input // 4, 1, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(x, 1)
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.activation(self.linear2(x)))
        x = self.linear3(x)
        return x

class LinearTransformLayer(nn.Module):
    def __init__(self, d_input: int, d_model : int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearTransformLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_input, d_model, **factory_kwargs)
        self.linear2 = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout1(self.activation(self.linear1(x)))
        return x
    
    # def forward(self, x: Tensor) -> Tensor:
    #     x = self.dropout1(self.activation(self.linear1(x)))
    #     x = self.norm1(x + self._ff_block(x))
    #     return x
    
    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.activation(self.linear2(x)))

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = True, norm_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask,
                                   tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory,
                                    memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory,
                           memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'elu':
        return F.elu
    elif activation == 'leaky_relu':
        return F.leaky_relu
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'tanh':
        return F.tanh

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))
