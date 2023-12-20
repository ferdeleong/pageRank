import numpy as np

def print_matrix(matrix):
    for row in matrix:
        print(row)

# Dictionary representing links
links = {
    1: [2, 3, 7, 8],
    2: [1, 4, 5, 9],
    3: [2, 4, 8],
    4: [1, 3, 5],
    5: [2, 4, 7, 8],
    6: [2, 3, 5, 10],
    7: [1, 4, 6, 9],
    8: [2, 3, 7, 8],
    9: [1, 3, 5, 7],
    10: [],
}

def create_adjacency_matrix(links_dict):
    n = max(links_dict.keys())
    H = np.zeros((n, n))

    for i, outbound_links in links_dict.items():
        for j in outbound_links:
            H[i - 1][j - 1] = round(1 / len(outbound_links), 3)
    
    return H

def find_pages_with_no_outbound_links(matrix):
    n = len(matrix)
    a = np.zeros((n, 1))

    for i in range(n):
        if np.sum(matrix[i]) == 0:
            a[i] = 1
    
    return a

def calculate_matrix_S(H, a, n):
    e = np.ones((n, 1))
    S = H + (1 / n) * np.dot(a, e.T)
    return S

def calculate_G_matrix(S, n, alpha):
    e = np.ones((n, 1))
    return alpha * S + ((1 - alpha) * (1 / n) * e * e.T)

def pagerank_iteration(G, alpha, pi):
    return alpha * G.T @ pi

def pagerank_solver(G, alpha, tol=1e-6, max_iter=1000):
    pi = np.ones(len(G)) / len(G)

    for i in range(max_iter):
        next_pi = pagerank_iteration(G, alpha, pi)

        if np.linalg.norm(next_pi - pi, 1) < tol:
            break

        pi = next_pi

    return pi / sum(pi), i

def show_ranking(pagerank_vector):
    pagerank_vector = sorted(enumerate(pagerank_vector), key=lambda x: x[1], reverse=True)

    for index, value in pagerank_vector:
        print(f"Page: {index+1}, Value: {value}")

# Main part
adjacency_matrix = create_adjacency_matrix(links)
pages_no_outbound_links = find_pages_with_no_outbound_links(adjacency_matrix)
matrix_S = calculate_matrix_S(adjacency_matrix, pages_no_outbound_links, len(links))

alpha_values = [0.15, 0.5, 0.85]
G_matrices = [calculate_G_matrix(matrix_S, len(links), alpha) for alpha in alpha_values]

pagerank_vectors = []
convergence_speeds = []

for i, alpha in enumerate(alpha_values):
    pagerank_vector, convergence_speed = pagerank_solver(G_matrices[i], alpha)
    pagerank_vectors.append(pagerank_vector)
    convergence_speeds.append(convergence_speed)

for i, alpha in enumerate(alpha_values):
    print(f"PageRank Vector when α is {alpha}:")
    show_ranking(pagerank_vectors[i])
    print("Convergence Speed:", convergence_speeds[i])
    print()