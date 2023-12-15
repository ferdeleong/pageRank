import numpy as np
from scipy.sparse import csc_matrix

def print_matrix(matrix):
  #Funtion to print matrixes, we'll use it throughout
  for row in matrix:
    print(row)

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
    10: [],
}

# Find the maximum page number from the links
n = max(links.keys())

# Create a matrix H filled with zeros of size (n x n)
H = np.zeros((n, n))

# Fill the matrix H based on links information
for i, outbound_links in links.items():
    for j in outbound_links:
        H[i - 1][j - 1] = round(1 / len(outbound_links), 3)
        
print_matrix(H)