#计算多模态数据的因果权重矩阵
from castle.algorithms import PC
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
import torch
##ISRUC
# def causal_M(data):  # data shape: (batch, 10, 10)
#     data = data.numpy()
#     batch_size = data.shape[0]  # 获取 batch 大小
#     cau = PC()  # 初始化 PC 算法对象
#     M = torch.zeros([batch_size, 4, 4])  # 初始化结果张量

#     for i in range(batch_size):
#          # 对每个 10x10 矩阵使用 PC 算法
#         cau.learn(data[i])  
#         causal_matrix = cau.causal_matrix  # 获取因果算法计算的10*10权重矩阵
#          #计算不同模态之间的权重矩阵
#         T = torch.zeros([4,4])
#         T[0][1] = (causal_matrix[:6,6].sum()+causal_matrix[:6,7].sum()).item()
#         T[0][2] = (causal_matrix[:6,8].sum()).item()
#         T[0][3] = (causal_matrix[:6,9].sum()).item()
#         T[1][0] = (causal_matrix[6,:6].sum()+causal_matrix[7,:6].sum()).item()
#         T[1][2] = (causal_matrix[6:8,8].sum()).item()
#         T[1][3] = (causal_matrix[6:8,9].sum()).item()
#         T[2][0] = (causal_matrix[8,:6].sum()).item()
#         T[2][1] = (causal_matrix[8,6:8].sum()).item()
#         T[2][3] = (causal_matrix[8,9]).item()
#         T[3][0] = (causal_matrix[9,:6].sum()).item()
#         T[3][1] = (causal_matrix[9,6:8].sum()).item()
#         T[3][2] = (causal_matrix[9,8]).item()

#          # 将结果赋值给 M
#         M[i] = T

#     return M  # 返回结果张量，形状为 (batch, 4, 4)

#multivariate
# def causal_M(data):  # data shape: (batch, k, k)
#     data = data.numpy()
#     batch_size = data.shape[0]  # 获取 batch 大小
#     K = data.shape[2]
#     cau = PC()  # 初始化 PC 算法对象
#     M = torch.zeros([batch_size, K, K])
#     for i in range(batch_size):
#         # 对每个 10x10 矩阵使用 PC 算法
#         cau.learn(data[i])  
#         causal_matrix = cau.causal_matrix  # 获取因果算法计算的k*k权重矩阵
#         #计算不同模态之间的权重矩阵
#         M[i] = torch.from_numpy(causal_matrix)


#     return M  # 返回结果张量，形状为 (batch, K, K)

# #Racket, Basicmotions
def causal_M(data):  # data shape: (batch, 6, 6)
   data = data.numpy()
   batch_size = data.shape[0]  # 获取 batch 大小
   cau = PC()  # 初始化 PC 算法对象
   M = torch.zeros([batch_size, 2, 2])  # 初始化结果张量

   for i in range(batch_size):
       # 对每个 10x10 矩阵使用 PC 算法
       cau.learn(data[i])  
       causal_matrix = cau.causal_matrix  # 获取因果算法计算的10*10权重矩阵
       #计算不同模态之间的权重矩阵
       T = torch.zeros([2,2])
       T[0][1] = (causal_matrix[0,3:6].sum()+causal_matrix[1,3:6].sum()+causal_matrix[2,3:6].sum()).item()
       T[1][0] = (causal_matrix[3:6,0].sum()+causal_matrix[3:6,1].sum()+causal_matrix[3:6,2].sum()).item()

       # 将结果赋值给 M
       M[i] = T

   return M  # 返回结果张量，形状为 (batch, 2, 2)