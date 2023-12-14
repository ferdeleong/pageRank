import numpy as np
from scipy.sparse import csc_matrix

def page_rank(pi0, H, n, alpha, epsilon):
    row_sum_vector = np.sum(H, axis=1).A1  # Get row sums for H
    non_zero_rows = np.where(row_sum_vector != 0)[0]
    zero_rows = np.where(row_sum_vector == 0)[0]

    a = np.zeros(n)
    a[zero_rows] = 1

    k = 0
    residual = 1
    pi = pi0.copy()

    while residual >= epsilon:
        prev_pi = pi.copy()
        k += 1
        pi = alpha * pi.dot(H) + (alpha * (pi.dot(a) + 1 - alpha)) * (np.ones(n) / n)
        residual = np.linalg.norm(pi - prev_pi, ord=1)

    return pi, k

# Example usage

#Dictionay
links = {
    1: [2, 3, 7, 8],
    2: [1, 4, 5, 9],
    3: [2, 4, 8],
    4: [1, 3, 5],
    5: [2, 4, 7, 8],
    6: [2, 3, 5, 10],
    7: [1, 4, 6, 9],
    8: [2,3,7,8],
    9: [1, 3, 5, 7],
    10: [2, 4, 6, 9],
}

# Find the maximum page number from the links
n = max(links.keys())

# Create a matrix H filled with zeros of size (n x n)
H = np.zeros((n, n))

# Fill the matrix H based on links information
for i, outbound_links in links.items():
    for j in outbound_links:
        H[i - 1][j - 1] = round(1 / len(outbound_links), 3)
H = H / H.sum(axis=1)[:, np.newaxis]  # Normalize rows to ensure it is row-stochastic

pi0 = np.ones(n) / n  # Starting vector (usually set to the uniform vector)
alpha = 0.9  # Scaling parameter
epsilon = 1e-8  # Convergence tolerance

# Convert H to a sparse matrix for efficiency
H_sparse = csc_matrix(H)

# Run the PageRank algorithm
pagerank_vector, iterations = page_rank(pi0, H_sparse, n, alpha, epsilon)
print("PageRank vector:", pagerank_vector)
print("Number of iterations:", iterations)
