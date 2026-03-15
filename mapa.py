import numpy as np
import matplotlib.pyplot as plt

nodos = ["A", "B", "C", "D"]

matriz = np.array([
    [0, 1, 2, 0],
    [1, 0, 1, 3],
    [2, 1, 0, 1],
    [0, 3, 1, 0]
])

sigma = 0.2
ruido = np.random.normal(0, sigma, matriz.shape)
matriz_ruido = matriz + ruido

plt.imshow(matriz_ruido)
plt.colorbar(label="Intensidad de relación")

plt.xticks(range(len(nodos)), nodos)
plt.yticks(range(len(nodos)), nodos)

for i in range(len(nodos)):
    for j in range(len(nodos)):
        plt.text(j, i, f"{matriz_ruido[i, j]:.2f}",
                 ha="center", va="center")

plt.title("Mapa de calor de la matriz adyacente con ruido gaussiano")
plt.xlabel("Nodos")
plt.ylabel("Nodos")
plt.show()

