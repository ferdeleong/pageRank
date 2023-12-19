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

# Create the column vector 'e'
e = np.ones((n, 1))

# Create the column vector 'a' indicating pages with no outbound links
a = np.zeros((n, 1))

# Find pages with no outbound links and set 'a' accordingly
for i in range(n):
    if np.sum(H[i]) == 0:
        a[i] = 1

# Calculate matrix S
S = H + (1 / n) * np.dot(a, e.T)

print("Matrix S:")
print_matrix(S)

alpha_15 = 0.15
alpha_50 = 0.5
alpha_85 = 0.85

G_15 = alpha_15*(S) + ((1 - alpha_15)*(1/n)*e*e.T) # When α is 0.15
G_50 = alpha_50*(S) + ((1 - alpha_50)*(1/n)*e*e.T) # When α is 0.5
G_85 = alpha_85*(S) + ((1 - alpha_85)*(1/n)*e*e.T) # When α is 0.85

print("When α is 0.15")
print_matrix(G_15)

print("When α is 0.5")
print_matrix(G_50)

print("When α is 0.85")
print_matrix(G_85)

print("When α is 0.15")
for row in G_15:
    print(sum(row))
    
print("When α is 0.5")
for row in G_50:
    print(sum(row))

print("When α is 0.85")
for row in G_85:
    print(sum(row))