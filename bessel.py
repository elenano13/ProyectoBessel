#Proyecto Bessel
#Métodos matemáticos para la física 2
#Elena Rodríguez 21774, Javier Rucal 21779, Diego de Florán 21565

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, n, theta = sp.symbols('x n theta')
#Para definir la forma integral de la función de Bessel de primer tipo 
#Definimos la función 
funcion1 = sp.cos(x * sp.sin(theta) - n * theta)
funcion2= sp.cosh(x*sp.cos(theta))
# Definimos la integral 
J = (1/sp.pi) * sp.integrate(funcion1, (theta, 0, sp.pi))
I= (1/sp.pi)*sp.integrate(funcion2, (theta, 0, sp.pi))

f1 = sp.lambdify((x, n), J)
f2= sp.lambdify(x,I)

# Para graficar
xs = np.linspace(0, 15, 100)
xs2 = np.linspace(0, 4, 100)
ns = [0, 1, 2, 3, 4, 5]

#Primera integral
plt.figure(figsize=(8, 6))
for i in ns:
    Jn = [f1(xi, i) for xi in xs]
    
    plt.plot(xs, Jn, label=f"n={i}")

plt.title(r"Función de Bessel de primer tipo")
plt.xlabel("$x$")
plt.ylabel("Jn(x)")
plt.grid(True)
plt.legend()
plt.show()

#Segunda integral
I0=[f2(xi) for xi in xs2]
plt.figure(2)
plt.plot(xs2, I0)
plt.title(r"Función de Bessel modificada de primer tipo de orden 0")
plt.xlabel("$x$")
plt.ylabel("I0(x)")
plt.grid(True)
plt.legend()
plt.show()