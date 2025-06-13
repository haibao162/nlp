import torch
torch.manual_seed(2)

embedding = torch.nn.Embedding(4,4, padding_idx=0)
print(embedding.state_dict(), 'embedding')

input = torch.LongTensor([[1,1,2,3]])
print(embedding(input))