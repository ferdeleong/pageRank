import numpy as np
import matplotlib.pyplot as plt

def print_matrix(matrix):
    for row in matrix:
        print(row)

def create_links():
    return {
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

def build_matrix(links, n):
    H = np.zeros((n, n))
    for i, outbound_links in links.items():
        for j in outbound_links:
            H[i - 1][j - 1] = round(1 / len(outbound_links), 3)
    return H

def create_vectors(n):
    e = np.ones((n, 1))
    a = np.zeros((n, 1))
    return e, a

def calculate_matrix_s(H, n, a, e):
    S = H + (1 / n) * np.dot(a, e.T)
    return S

def calculate_matrix_g(S, alpha, n, e):
    return alpha * S + ((1 - alpha) / n) * np.dot(e, e.T)

def pagerank_iteration(G, alpha, pi):
    return alpha * G.T @ pi

def pagerank_solver(G, alpha, tol=1e-6, max_iter=1000):
    pi = np.ones(len(G)) / len(G)

    for i in range(max_iter):
        next_pi = pagerank_iteration(G, alpha, pi)

        if np.linalg.norm(next_pi - pi, 1) < tol:
            break

        pi = next_pi
        convergence_speed = i

    return pi / sum(pi), convergence_speed

def show_ranking(pagerank_vector):
    pagerank_vector = sorted(enumerate(pagerank_vector), key=lambda x: x[1], reverse=True)

    for index, value in pagerank_vector:
        print(f"Page: {index+1}, Value: {value}")

def plot_convergence_speeds(alpha_values, convergence_speeds):
    plt.figure(figsize=(10, 6))
    plt.scatter(alpha_values, convergence_speeds, color=['blue', 'red', 'green'])
    plt.plot(alpha_values, convergence_speeds, linestyle='--', color='black')
    plt.xlabel('Alpha Values')
    plt.ylabel('Convergence Speed')
    plt.title('Convergence Speed vs. Alpha Values')
    plt.show()

def main():
    links = create_links()
    n = max(links.keys())
    H = build_matrix(links, n)
    print_matrix(H)

    e, a = create_vectors(n)
    S = calculate_matrix_s(H, n, a, e)
    print("Matrix S:")
    print_matrix(S)

    alpha_values = [0.15, 0.5, 0.85]
    G_values = [calculate_matrix_g(S, alpha, n, e) for alpha in alpha_values]

    for alpha, G in zip(alpha_values, G_values):
        print(f"When α is {alpha}")
        print_matrix(G)

    convergence_speeds = []
    pagerank_vectors = []

    for alpha, G in zip(alpha_values, G_values):
        pagerank_vector, convergence_speed = pagerank_solver(G, alpha)
        convergence_speeds.append(convergence_speed)
        pagerank_vectors.append(pagerank_vector)

        print(f"\nPageRank Vector when α is {alpha}:")
        show_ranking(pagerank_vector)
        print("Convergence Speed: ", convergence_speed)

    for G, pagerank_vector in zip(G_values, pagerank_vectors):
        print(np.dot(G.T, pagerank_vector))
        print(pagerank_vector)

    plot_convergence_speeds(alpha_values, convergence_speeds)

if __name__ == "__main__":
    main()