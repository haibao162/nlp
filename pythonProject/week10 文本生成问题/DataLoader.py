from torch.utils.data import Dataset, DataLoader

a = [1,2,3,4]
b = DataLoader(a, shuffle=True, batch_size=2)
for data in b:
    print(data)
# print(b[0])