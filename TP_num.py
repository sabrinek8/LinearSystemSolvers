import numpy as np
import time
import matplotlib.pyplot as plt

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))) / A[i][i]

    return x

def gauss_seidel(A, b, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.zeros(n)
    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i][:i], x[:i]) - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def generate_diagonally_dominant_system(n):
    A = np.random.rand(n, n)
    # Calcul des sommes des valeurs absolues des éléments hors diagonale pour chaque ligne
    row_sums = np.sum(np.abs(A), axis=1)
    # Réglage des éléments diagonaux pour assurer la domination diagonale
    np.fill_diagonal(A, row_sums + np.random.rand(n))
    b = np.random.rand(n)
    return A, b

sizes = [100, 400, 500, 700, 1000, 1500, 2000]

gauss_times = []
gs_times = []

for n in sizes:
    print(f"Size of matrix: {n}x{n}")
    A, b = generate_diagonally_dominant_system(n)
    
    start_time = time.time()
    x_gauss = gauss_elimination(A.copy(), b.copy())
    gauss_time = time.time() - start_time
    gauss_times.append(gauss_time)
    print(f"Gauss elimination time: {gauss_time:.6f} seconds")

    start_time = time.time()
    x_gs = gauss_seidel(A.copy(), b.copy())
    gs_time = time.time() - start_time
    gs_times.append(gs_time)
    print(f"Gauss-Seidel time: {gs_time:.6f} seconds")

    print()

plt.plot(sizes, gauss_times, label='Gauss Elimination')
plt.plot(sizes, gs_times, label='Gauss-Seidel')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True)
plt.show()
