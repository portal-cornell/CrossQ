import numpy as np
# import pdb

from tslearn.metrics import SoftDTW

# have a cost matrix (4, 3) where the cost to [:, 0] is the lowest
cost_matrix = np.array([[0.1, 0.5, 0.9],
                        [0.1, 0.5, 0.9],
                        [0.1, 0.9, 0.9],
                        [0.1, 0.9, 0.9],
                        [0.1, 0.5, 0.9],
                        [0.1, 0.5, 0.9]])

print(f"cost_matrix: {cost_matrix.shape}\n{cost_matrix}")
# breakpoint()

sdtw = SoftDTW(cost_matrix, gamma=10)
dist_sq = sdtw.compute()  # We don't actually use this

print(f"dist_sq: {dist_sq.shape}\n{dist_sq}")
# breakpoint()

a = sdtw.grad()

print(f"Assignment: {a.shape}\n{a}")

print(np.sum(a, axis=1))
print(np.sum(a, axis=0))

# m = cost_matrix.shape[0]
# n = cost_matrix.shape[1]

# D = np.vstack((cost_matrix, np.zeros(n)))
# D = np.hstack((D, np.zeros((m + 1, 1))))

# print(f"D: {D.shape}\n{D}")
# m = cost_matrix.shape[0]
# n = cost_matrix.shape[1]
# for i in range(1, m + 1):
#     for j in range(1, n + 1):
#         print(f"{i}, {j}: {cost_matrix[i-1, j-1]}")
                