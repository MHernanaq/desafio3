import numpy as np
def gauss_seidel(A, B, x0, max_iter=100, tol=1e-6):
    n = len(B)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum_ax = sum(A[i][j] * x_new[j] if j != i else 0 for j in range(n))
            x_new[i] = (B[i] - sum_ax) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x
def ingresar_sistema():
    n = int(input("Ingrese el número de ecuaciones: "))
    A = []
    print("Ingrese la matriz de coeficientes (una fila a la vez, separados por espacio):")
    for i in range(n):
        fila = list(map(float, input(f"Fila {i+1}: ").split()))
        A.append(fila)
    B = list(map(float, input("Ingrese los términos independientes, separados por espacio: ").split()))
    x0 = list(map(float, input("Ingrese las estimaciones iniciales para las incógnitas, separadas por espacio: ").split()))
    return np.array(A), np.array(B), np.array(x0)
A, B, x0 = ingresar_sistema()
solucion = gauss_seidel(A, B, x0)
print("La solución aproximada es:", solucion)