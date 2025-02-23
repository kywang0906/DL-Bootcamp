import torch

print('------ Task 3-1 ------')
matrix = [[2, 3, 1], [5, -2, 1]]
tensor1 = torch.tensor(matrix)
print(tensor1.dtype, tensor1.shape)

print('------ Task 3-2 ------')
tensor2 = torch.rand(3, 4, 2)
print(tensor2.shape)
print(tensor2)

print('------ Task 3-3 ------')
tensor3 = torch.ones(2, 1, 5)
print(tensor3.shape)
print(tensor3)

print('------ Task 3-4 ------')
matrix1 = [[1, 2, 4], [2, 1, 3]] # 2 x 3
matrix2 = [[5], [2], [1]] # 3 x 1
tensor4 = torch.matmul(torch.tensor(matrix1), torch.tensor(matrix2))
print(tensor4)

print('------ Task 3-5 ------')
matrix1 = [[1, 2], [2, 3], [-1, 3]] # 3 x 2
matrix2 = [[5, 4], [2, 1], [1, -5]] # 3 x 2
tensor5 = torch.tensor(matrix1) * torch.tensor(matrix2)
print(tensor5)