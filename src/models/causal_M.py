# Compute causal weight matrix for multimodal data
from castle.algorithms import PC
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
import torch

## ISRUC dataset configuration
# def causal_M(data):  # data shape: (batch, 10, 10)
#     data = data.numpy()
#     batch_size = data.shape[0]  # Get batch size
#     cau = PC()  # Initialize PC algorithm object
#     M = torch.zeros([batch_size, 4, 4])  # Initialize result tensor

#     for i in range(batch_size):
#         # Apply PC algorithm to each 10x10 matrix
#         cau.learn(data[i])
#         causal_matrix = cau.causal_matrix  # Get 10*10 causal weight matrix from PC algorithm
#         # Compute weight matrix between different modalities
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

#         # Assign result to M
#         M[i] = T

#     return M  # Return result tensor with shape (batch, 4, 4)

## Multivariate configuration
# def causal_M(data):  # data shape: (batch, k, k)
#     data = data.numpy()
#     batch_size = data.shape[0]  # Get batch size
#     K = data.shape[2]
#     cau = PC()  # Initialize PC algorithm object
#     M = torch.zeros([batch_size, K, K])
#     for i in range(batch_size):
#         # Apply PC algorithm to each matrix
#         cau.learn(data[i])
#         causal_matrix = cau.causal_matrix  # Get k*k causal weight matrix from PC algorithm
#         # Compute weight matrix between different modalities
#         M[i] = torch.from_numpy(causal_matrix)

#     return M  # Return result tensor with shape (batch, K, K)

## Racket, BasicMotions configuration
def causal_M(data):  # data shape: (batch, 6, 6)
   data = data.numpy()
   batch_size = data.shape[0]  # Get batch size
   cau = PC()  # Initialize PC algorithm object
   M = torch.zeros([batch_size, 2, 2])  # Initialize result tensor

   for i in range(batch_size):
       # Apply PC algorithm to each matrix
       cau.learn(data[i])
       causal_matrix = cau.causal_matrix  # Get causal weight matrix from PC algorithm
       # Compute weight matrix between different modalities
       T = torch.zeros([2,2])
       T[0][1] = (causal_matrix[0,3:6].sum()+causal_matrix[1,3:6].sum()+causal_matrix[2,3:6].sum()).item()
       T[1][0] = (causal_matrix[3:6,0].sum()+causal_matrix[3:6,1].sum()+causal_matrix[3:6,2].sum()).item()

       # Assign result to M
       M[i] = T

   return M  # Return result tensor with shape (batch, 2, 2)