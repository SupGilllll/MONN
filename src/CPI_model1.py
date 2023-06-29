import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
#from sru import SRU, SRUCell
from pdbbind_utils import *

#some predefined parameters
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

#define the model
class Net(nn.Module):
	def __init__(self, init_atom_features, init_word_features, params):
		super(Net, self).__init__()
		
		self.init_atom_features = init_atom_features
		self.init_word_features = init_word_features
		"""hyper part"""
		inner_CNN_depth, kernel_size, hidden_size1, hidden_size2 = params
		self.inner_CNN_depth = inner_CNN_depth
		self.kernel_size = kernel_size
		self.hidden_size1 = hidden_size1
		self.hidden_size2 = hidden_size2
						
		"""CNN-RNN Module"""
		#CNN parameters
		self.embed_seq = nn.Embedding(len(self.init_word_features), 20, padding_idx=0)
		self.embed_seq.weight = nn.Parameter(self.init_word_features)
		self.embed_seq.weight.requires_grad = False
		
		self.conv_first = nn.Conv1d(20, self.hidden_size1, kernel_size=self.kernel_size, padding='same')
		self.conv_last = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size, padding='same')
		
		self.plain_CNN = nn.ModuleList([])
		for i in range(self.inner_CNN_depth):
			self.plain_CNN.append(nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=self.kernel_size, padding='same'))

		self.conv_first_c = nn.Conv1d(atom_fdim, self.hidden_size1, kernel_size=3, padding='same')
		self.conv_last_c = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=3, padding='same')
		
		self.plain_CNN_c = nn.ModuleList([])
		for i in range(self.inner_CNN_depth):
			self.plain_CNN_c.append(nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size=3, padding='same'))
			
		"""Affinity Prediction Module"""
		self.c_final = nn.Linear(self.hidden_size1, self.hidden_size2)
		self.p_final = nn.Linear(self.hidden_size1, self.hidden_size2)

		#Output layer
		self.W_out = nn.Linear(self.hidden_size2*self.hidden_size2, 1)
		
		"""Pairwise Interaction Prediction Module"""
		self.pairwise_compound = nn.Linear(self.hidden_size1, self.hidden_size1)
		self.pairwise_protein = nn.Linear(self.hidden_size1, self.hidden_size1)
		
			
	def CNN_module_protein(self, batch_size, seq_mask, sequence):
		
		ebd = self.embed_seq(sequence)
		ebd = ebd.transpose(1,2)
		x = F.leaky_relu(self.conv_first(ebd), 0.1)
		
		for i in range(self.inner_CNN_depth):
			x = self.plain_CNN[i](x)
			x = F.leaky_relu(x, 0.1)
		
		x = F.leaky_relu(self.conv_last(x), 0.1)
		H = x.transpose(1,2)
		#H, hidden = self.rnn(H)
		
		return H
	
	def CNN_module_compound(self, batch_size, vertex_mask, vertex):
		
		vertex_f = torch.index_select(self.init_atom_features, 0, vertex.view(-1))
		vertex_f = vertex_f.view(batch_size, -1, atom_fdim)
		vertex_f = vertex_f.transpose(1,2)

		x = F.leaky_relu(self.conv_first_c(vertex_f), 0.1)
		
		for i in range(self.inner_CNN_depth):
			x = self.plain_CNN_c[i](x)
			x = F.leaky_relu(x, 0.1)
		
		x = F.leaky_relu(self.conv_last_c(x), 0.1)
		H = x.transpose(1,2)
		#H, hidden = self.rnn(H)
		
		return H
	
	
	def Pairwise_pred_module(self, batch_size, comp_feature, prot_feature, vertex_mask, seq_mask):
		
		pairwise_c_feature = F.leaky_relu(self.pairwise_compound(comp_feature), 0.1)
		pairwise_p_feature = F.leaky_relu(self.pairwise_protein(prot_feature), 0.1)
		pairwise_pred = torch.sigmoid(torch.matmul(pairwise_c_feature, pairwise_p_feature.transpose(1,2)))
		pairwise_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))
		pairwise_pred = pairwise_pred*pairwise_mask
		
		return pairwise_pred
	
	
	def Affinity_pred_module(self, batch_size, comp_feature, prot_feature):
		
		comp_feature = F.leaky_relu(self.c_final(comp_feature), 0.1)
		prot_feature = F.leaky_relu(self.p_final(prot_feature), 0.1)

		comp_feature = torch.mean(comp_feature, 1)
		prot_feature = torch.mean(prot_feature, 1)
		kroneck = F.leaky_relu(torch.matmul(comp_feature.view(batch_size,-1,1), prot_feature.view(batch_size,1,-1)).view(batch_size,-1), 0.1)
		
		affinity_pred = self.W_out(kroneck)
		return affinity_pred
	
	def forward(self, vertex_mask, vertex, seq_mask, sequence):
		batch_size = vertex.size(0)
		
		atom_feature = self.CNN_module_compound(batch_size, vertex_mask, vertex)
		prot_feature = self.CNN_module_protein(batch_size, seq_mask, sequence)
		
		pairwise_pred = self.Pairwise_pred_module(batch_size, atom_feature, prot_feature, vertex_mask, seq_mask)
		affinity_pred = self.Affinity_pred_module(batch_size, atom_feature, prot_feature)
		
		return affinity_pred, pairwise_pred

