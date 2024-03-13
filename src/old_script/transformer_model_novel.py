import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, init_atoms, init_residues, d_encoder: int = 512, d_decoder: int = 512, d_model: int = 512, 
                 nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.protein_transform_layer = LinearTransformLayer(d_encoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)
        self.compound_transform_layer = LinearTransformLayer(d_decoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)

        encoder_layer = AttentionEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
                                              batch_first, **factory_kwargs)
        affinity_output_layer = AffinityOutputLayer(d_model, dropout, activation, **factory_kwargs)
        pairwise_output_layer = PairwiseOutputLayer(d_model, dropout, activation, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.predictor = AttentionPredictor(encoder_layer, affinity_output_layer, pairwise_output_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()
        self.protein_embedding = nn.Embedding.from_pretrained(init_residues)
        self.compound_embedding = nn.Embedding.from_pretrained(init_atoms)    

    def forward(self, src: Tensor, tgt: Tensor, pids = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        prot_embed = self.protein_embedding_block(src)
        comp_embed = self.compound_embedding_block(tgt)
        affinity_output, pairwise_output = self.predictor(prot_embed, comp_embed, prot_key_padding_mask = src_key_padding_mask, 
                                                          comp_key_padding_mask = tgt_key_padding_mask)
        
        return affinity_output, pairwise_output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def protein_embedding_block(self, prot):
        x1 = self.protein_embedding(prot)
        x2 = self.protein_transform_layer(x1)
        return x2
    
    def compound_embedding_block(self, comp):
        y1 = self.compound_embedding(comp)
        y2 = self.compound_transform_layer(y1)
        return y2
    
class AttentionPredictor(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, affinity_output_layer, pairwise_output_layer, num_layers, norm=None):
        super(AttentionPredictor, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.affinity_output_layer = affinity_output_layer
        self.pairwise_output_layer = pairwise_output_layer
        self.norm_prot = norm
        self.norm_comp = norm

    def forward(self, prot: Tensor, comp: Tensor, prot_key_padding_mask: Optional[Tensor] = None, 
                comp_key_padding_mask: Optional[Tensor] = None):
        for mod in self.layers:
            prot, comp = mod(prot, comp, prot_key_padding_mask = prot_key_padding_mask, comp_key_padding_mask = comp_key_padding_mask)
            
        if self.norm_prot is not None:
            prot = self.norm_prot(prot)
            comp = self.norm_comp(comp)

        affinity_output = self.affinity_output_layer(prot, comp, prot_key_padding_mask, comp_key_padding_mask)
        pairwise_output = self.pairwise_output_layer(prot, comp, prot_key_padding_mask, comp_key_padding_mask)

        return affinity_output, pairwise_output
    
class AttentionEncoderLayer(nn.Module):
    __constants__ = ['batch_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AttentionEncoderLayer, self).__init__()

        self.prot_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        self.comp_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    **factory_kwargs)
        
        self.prot_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                     **factory_kwargs)
        self.comp_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                     **factory_kwargs)
        
        self.prot_ff_0 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        self.comp_ff_0 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        self.prot_ff_1 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        self.comp_ff_1 = FeedForwardLayer(d_model, dim_feedforward, dropout, activation)
        
        self.prot_norm0 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.comp_norm0 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.prot_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.comp_norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.prot_norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.comp_norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.prot_norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.comp_norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, prot: Tensor, comp: Tensor, prot_mask: Optional[Tensor] = None, comp_mask: Optional[Tensor] = None,
                prot_key_padding_mask: Optional[Tensor] = None, comp_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = prot
        y = comp
        x = self.prot_norm0(x + self.dropout1(
            self.prot_self_attn(x, x, x, attn_mask=prot_mask, key_padding_mask=prot_key_padding_mask, need_weights=False)[0]))
        y = self.comp_norm0(y + self.dropout2(
            self.comp_self_attn(y, y, y, attn_mask=comp_mask, key_padding_mask=comp_key_padding_mask, need_weights=False)[0]))
        x = self.prot_norm1(x + self.prot_ff_0(x))
        y = self.comp_norm1(y + self.comp_ff_0(y))
        x1 = self.prot_norm2(x + self.dropout3(
            self.prot_cross_attn(x, y, y, attn_mask=comp_mask, key_padding_mask=comp_key_padding_mask, need_weights=False)[0]))
        y1 = self.comp_norm2(y + self.dropout4(
            self.comp_cross_attn(y, x, x, attn_mask=prot_mask, key_padding_mask=prot_key_padding_mask, need_weights=False)[0]))
        x1 = self.prot_norm3(x1 + self.prot_ff_1(x1))
        y1 = self.comp_norm3(y1 + self.comp_ff_1(y1))

        return x1, y1
    
class AffinityOutputLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AffinityOutputLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear_protein = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear_compound = nn.Linear(d_model, d_model, **factory_kwargs)
        self.final_layer = nn.Linear(d_model * d_model, 1, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def mask_softmax(self, a, bool_mask, dim=-1):
        mask = 1 - bool_mask.float()
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax

    def forward(self, prot: Tensor, comp: Tensor, p_bool_mask: Tensor, c_bool_mask: Tensor) -> Tensor:
        prot_f = self.dropout1(self.activation(self.linear_protein(prot)))
        comp_f = self.dropout2(self.activation(self.linear_compound(comp)))
        # prot_f = self.activation(self.linear_protein(prot))
        # comp_f = self.activation(self.linear_compound(comp))
        prot_norm = LA.norm(prot_f, dim = 2)
        comp_norm = LA.norm(comp_f, dim = 2)
        prot_norm = self.mask_softmax(prot_norm, p_bool_mask)
        comp_norm = self.mask_softmax(comp_norm, c_bool_mask)
        prot_sum = torch.sum(prot_f * prot_norm.unsqueeze(-1), dim = 1)
        comp_sum = torch.sum(comp_f * comp_norm.unsqueeze(-1), dim = 1)
        flatten = self.activation(torch.flatten(torch.matmul(torch.unsqueeze(comp_sum, 2), torch.unsqueeze(prot_sum, 1)), start_dim=1))
        # flatten = self.dropout3(self.activation(torch.flatten(torch.matmul(torch.unsqueeze(comp_sum, 2), torch.unsqueeze(prot_sum, 1)), start_dim=1)))
        affinity_pred = self.final_layer(flatten)
        return affinity_pred
    
class PairwiseOutputLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PairwiseOutputLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear_protein = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear_compound = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, prot: Tensor, comp: Tensor, p_bool_mask: Tensor, c_bool_mask: Tensor) -> Tensor:
        # need dropout here?
        prot_f = self.dropout1(self.activation(self.linear_protein(prot)))
        comp_f = self.dropout2(self.activation(self.linear_compound(comp)))
        # prot_f = self.activation(self.linear_protein(prot))
        # comp_f = self.activation(self.linear_compound(comp))
        pairwise_pred = torch.sigmoid(torch.matmul(comp_f, prot_f.transpose(1,2)))
        pairwise_mask = torch.matmul(torch.unsqueeze(1 - c_bool_mask.float(), 2), torch.unsqueeze(1 - p_bool_mask.float(), 1))
        pairwise_pred = pairwise_pred * pairwise_mask

        return pairwise_pred
    
class LinearTransformLayer(nn.Module):
    def __init__(self, d_input: int, d_model : int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearTransformLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_input, d_model, **factory_kwargs)
        # self.linear2 = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout1(self.activation(self.linear1(x)))
        # x = self.norm1(self.dropout1(self.activation(self.linear1(x))))
        return x
    
    # def forward(self, x: Tensor) -> Tensor:
    #     x = self.dropout1(self.activation(self.linear1(x)))
    #     x = self.norm1(x + self._ff_block(x))
    #     return x

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FeedForwardLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

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
