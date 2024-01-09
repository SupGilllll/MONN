import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

bond_fdim = 6

class Transformer(nn.Module):
    def __init__(self, init_atoms, init_bonds, init_residues, d_encoder: int = 512, d_decoder: int = 512, d_model: int = 512, 
                 nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.encoder_transform_layer = LinearTransformLayer(d_encoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
                                                batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.decoder_transform_layer = LinearTransformLayer(d_decoder, d_model, dropout, activation, layer_norm_eps, **factory_kwargs)
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, 
        #                                         batch_first, norm_first, **factory_kwargs)
        affinity_output_layer = AffinityOutputLayerPlus(d_model, dropout, activation, **factory_kwargs)
        # affinity_output_layer = AffinityOutputLayer(d_model, dropout, activation, **factory_kwargs)
        pairwise_output_layer = PairwiseOutputLayer(d_model, dropout, activation, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.decoder = TransformerDecoder(d_model, init_bonds, decoder_layer, affinity_output_layer, 
        #                                   pairwise_output_layer, num_decoder_layers, decoder_norm)
        self.decoder = TransformerDecoder(d_model, init_bonds, affinity_output_layer,
                                          pairwise_output_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.compound_embedding = nn.Embedding.from_pretrained(init_atoms)
        self.protein_embedding = nn.Embedding.from_pretrained(init_residues)

    def forward(self, src: Tensor, tgt: Tensor, edge: Tensor, atom_adj: Tensor, bond_adj: Tensor, nbs_mask: Tensor, pids = None, 
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
              
        embed_src = self.protein_embedding_block(src)
        memory = self.encoder(embed_src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        
        embed_tgt = self.compound_embedding_block(tgt)
        output = self.decoder(tgt = embed_tgt, edge = edge, atom_adj = atom_adj, bond_adj = bond_adj, nbs_mask = nbs_mask,
                              memory = memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
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

    def __init__(self, d_model, init_bonds, affinity_output_layer, pairwise_output_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.label_U2 = nn.ModuleList([nn.Linear(d_model + bond_fdim, d_model) for i in range(num_layers)]) #assume no edge feature transformation
        self.label_U1 = nn.ModuleList([nn.Linear(2 * d_model, d_model) for i in range(num_layers)])
        self.bond_embedding = nn.Embedding.from_pretrained(init_bonds)

        # self.layers = _get_clones(decoder_layer, num_layers)
        self.affinity_output_layer = affinity_output_layer
        self.pairwise_output_layer = pairwise_output_layer
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, edge: Tensor, atom_adj: Tensor, bond_adj: Tensor, nbs_mask: Tensor,
                memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        batch_size = tgt.size(0)
        edge_initial = self.bond_embedding(edge)
        vertex_mask = 1 - tgt_key_padding_mask.float()
        vertex_feature = tgt

        for GWM_iter in range(self.num_layers):
            vertex_feature = self.graph_unit(batch_size, vertex_mask, vertex_feature, edge_initial, atom_adj, bond_adj, nbs_mask, GWM_iter)  

        # if self.norm is not None:
        #     output = self.norm(output)
        # vertex_feature = self.norm(vertex_feature)

        pairwise_output = self.pairwise_output_layer(vertex_feature, memory, tgt_key_padding_mask, memory_key_padding_mask)

        return pairwise_output
    
    def graph_unit(self, batch_size, vertex_mask, vertex_features, edge_initial, atom_adj, bond_adj, nbs_mask, GNN_iter):
        n_vertex = vertex_mask.size(1)
        n_nbs = nbs_mask.size(2)
        
        vertex_mask = vertex_mask.view(batch_size,n_vertex,1)
        nbs_mask = nbs_mask.view(batch_size,n_vertex,n_nbs,1)
        
        vertex_nei = torch.index_select(vertex_features.view(-1, self.d_model), 0, atom_adj).view(batch_size, n_vertex, n_nbs, self.d_model)
        edge_nei = torch.index_select(edge_initial.view(-1, bond_fdim), 0, bond_adj).view(batch_size, n_vertex, n_nbs, bond_fdim)
        
        # Weisfeiler Lehman relabelling
        l_nei = torch.cat((vertex_nei, edge_nei), -1)
        nei_label = F.leaky_relu(self.label_U2[GNN_iter](l_nei), 0.1)
        nei_label = torch.sum(nei_label*nbs_mask, dim=-2)
        new_label = torch.cat((vertex_features, nei_label), 2)
        new_label = self.label_U1[GNN_iter](new_label)
        vertex_features = F.leaky_relu(new_label, 0.1)
        
        return vertex_features

class PairwiseOutputLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PairwiseOutputLayer, self).__init__()

        # Implementation of Feedforward model
        self.linear_compound = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear_protein = nn.Linear(d_model, d_model, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: Tensor, y: Tensor, c_bool_mask: Tensor, p_bool_mask: Tensor) -> Tensor:
        # need dropout here?
        # x = self.dropout1(self.activation(self.linear_compound(x)))
        # y = self.dropout2(self.activation(self.linear_protein(y)))
        x = self.activation(self.linear_compound(x))
        y = self.activation(self.linear_protein(y))
        pairwise_pred = torch.sigmoid(torch.matmul(x, y.transpose(1,2)))
        # pairwise_mask = torch.matmul(torch.unsqueeze(1 - c_bool_mask.float(), 2), torch.unsqueeze(1 - p_bool_mask.float(), 1))
        # pairwise_pred = pairwise_pred * pairwise_mask

        return pairwise_pred
    
class AffinityOutputLayerPlus(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AffinityOutputLayerPlus, self).__init__()

        # Implementation of Feedforward model
        self.linear_protein = nn.Linear(d_model, d_model, **factory_kwargs)
        self.linear_compound = nn.Linear(d_model, d_model, **factory_kwargs)
        self.final_layer = nn.Linear(d_model * d_model, 1, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
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
        # prot_f = self.dropout1(self.activation(self.linear_protein(prot)))
        # comp_f = self.dropout2(self.activation(self.linear_compound(comp)))
        prot_f = self.activation(self.linear_protein(prot))
        comp_f = self.activation(self.linear_compound(comp))
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
