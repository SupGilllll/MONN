from transformer_model import *
from torchinfo import summary
from pdbbind_utils import loading_emb
import numpy as np

# cuda1 = torch.device('cuda:1')
# transformer = Transformer(d_encoder = 256, d_decoder= 64, d_model = 512, device = cuda1)
# src = torch.rand((32, 10, 256)).to(cuda1)
# tgt = torch.rand((32, 20, 64)).to(cuda1)
# out = transformer(src, tgt)
# # summary(transformer, input_size=((10, 32, 512), (20, 32, 512)))
# # print(transformer)
# print(src.shape)
# print(tgt.shape)
# print(out.shape)

# init_atoms, _, init_residues = loading_emb('KIKD', 'blosum62')
# atoms = torch.LongTensor([[2,4,6], [1,3,5]]).cuda()
# compound1 = torch.index_select(init_atoms, 0, atoms.view(-1))
# compound1 = compound1.view(2, -1, 82)

# seq_embedding = nn.Embedding.from_pretrained(init_atoms).cuda()
# compound2 = seq_embedding(atoms)

# print(compound1)
# print(compound2)

# target output size of 5
# pool of size=3, stride=2
# input = torch.FloatTensor([[[1,2,3],[2,3,4]], [[4,5,6],[5,6,7]]])
# print(input)
# output = torch.mean(input, 1)
# print(output)

# x = torch.FloatTensor([[[2,4],[4,6]]])
# x_padding1 = torch.BoolTensor([[True, True]])
# x_padding2 = torch.BoolTensor([[False, False]])
# x_padding3 = torch.BoolTensor([[False, True]])
# x_padding4 = torch.BoolTensor([[True, False]])
# model = nn.MultiheadAttention(2, num_heads=2, batch_first = True)
# output1 = model(x, x, x)[0]
# output2 = model(x, x, x, need_weights=True)
# output3 = model(x, x, x, need_weights=False)
# output2 = model(x, x, x, key_padding_mask=x_padding1)[0]
# output3 = model(x, x, x, key_padding_mask=x_padding2)[0]
# output4 = model(x, x, x, key_padding_mask=x_padding3)[0]
# output5 = model(x, x, x, key_padding_mask=x_padding4)[0]
# print(output1)
# print(output2)
# print(output3)
# print(output4)
# print(output5)

# data = np.zeros((1,6))
# data[0,0:4] = 1
# t = torch.BoolTensor(1-data)
# print(data)
# print(t)

c_bool_mask = torch.BoolTensor([[False,False,True,True], [False,False,False,False]])
p_bool_mask = torch.BoolTensor([[False,False,False,True,True,True], [False,False,False,False,False,False]])
c_bool_mask = c_bool_mask.float()
p_bool_mask = p_bool_mask.float()
n = torch.matmul(torch.unsqueeze(1 - c_bool_mask, 2), torch.unsqueeze(1 - p_bool_mask, 1))
c_mask = torch.FloatTensor([[1,1,0,0], [1,1,1,1]])
p_mask = torch.FloatTensor([[1,1,1,0,0,0], [1,1,1,1,1,1]])
pairwise_mask = torch.matmul(c_mask.view(2,-1,1), p_mask.view(2,1,-1))
x = torch.unsqueeze(c_mask, 2)
y = torch.unsqueeze(p_mask, 1)
z = torch.matmul(x, y)
print(c_mask)
print(p_mask)
print(pairwise_mask)
print("        ")
print(x)
print(y)
print(z)
print(n)
print(pairwise_mask.equal(z))
print(pairwise_mask.equal(n))

