import numpy as np
import matplotlib.pyplot as plt

# calcula el residuo de ortogonalidad a·(a x b) para k en [1,10]
def experimento(dtype):
    ks = np.arange(1, 11)
    residuos = []
    for k in ks:
        a = np.array([1, 0, 0], dtype=dtype)
        b = np.array([1, 10.0**(-k), 0], dtype=dtype)
        c = np.cross(a, b)                 # producto vectorial en NumPy
        residuo = abs(np.dot(a, c))        # verificacion de ortogonalidad teorica
        residuos.append(residuo)
    return ks, np.array(residuos, dtype=dtype)

# calculo en las dos precisiones
ks, res32 = experimento(np.float32)
_,  res64 = experimento(np.float64)

print("k\tresiduo float32\t\tresiduo float64")
for k, r32, r64 in zip(ks, res32, res64):
    print(f"{k}\t{r32}\t\t{r64}")

# para graficar en escala log evitamos log(0) usando el "piso" de precision
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps
res32_plot = np.maximum(res32, eps32)
res64_plot = np.maximum(res64, eps64)

plt.figure(figsize=(8, 5))
plt.semilogy(ks, res32_plot, 'o-', label='float32')
plt.semilogy(ks, res64_plot, 's-', label='float64')
plt.xlabel('k')
plt.ylabel(r'$|\vec{a}\cdot\vec{c}|$ (residuo, con piso de eps si es 0)')
plt.title('Residuo de ortogonalidad numérica vs. k')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ortogonalidad_residuo.png', dpi=150)
plt.show()
