import math

import numpy as np
import sympy as sp
from scipy.linalg import eigh

from integration import potential_matrix


def characteristic(rho, d1, a):
    return (
        (a**4 + 2*a*a + 1) * math.sin(rho)
        + (2*a**4 - 2) * math.sin(rho * (2*d1 - 1))
        - (a*a - 1)**2 * math.sin(rho * (4*d1 - 1))
    ) / (4*a*a*rho)


def bisection(left, right, d1, a, eps):
    f_left = characteristic(left, d1, a)
    while right - left > eps:
        middle = (left + right) / 2
        f_middle = characteristic(middle, d1, a)
        if f_left * f_middle <= 0:
            right = middle
        else:
            left = middle
            f_left = f_middle
    return (left + right) / 2


def model_eigenvalues(n, d1, a, eps, step):
    roots = []
    left = 1e-8
    while len(roots) < n:
        right = left + step
        if characteristic(left, d1, a) * characteristic(right, d1, a) < 0:
            roots.append(bisection(left, right, d1, a, eps))
        left = right
    return np.array(roots)


def coefficients(rho, d1, a):
    # На каждом интервале y = u*cos(rho*t) + v*sin(rho*t).
    d2 = 1 - d1
    s = np.sin(rho * d1)
    c = np.cos(rho * d1)
    u2, v2 = a*s, c/a
    t = rho * (d2 - d1)
    u3 = s*np.cos(t) + c*np.sin(t)/(a*a)
    v3 = c*np.cos(t) - a*a*s*np.sin(t)
    return [(np.zeros(len(rho)), np.ones(len(rho))),
            (u2, v2), (u3, v3)]


# Интеграл квадрата тригонометрической функции считаем символьно один раз.
t, u_sym, v_sym = sp.symbols('t u v', real=True)
r_sym, length_sym = sp.symbols('rho L', positive=True, real=True)
y_sym = u_sym*sp.cos(r_sym*t) + v_sym*sp.sin(r_sym*t)
norm_integral = sp.trigsimp(sp.integrate(y_sym**2, (t, 0, length_sym)))
# Превращаем аналитическое выражение в функцию NumPy для массивов rho.
norm_numpy = sp.lambdify((u_sym, v_sym, r_sym, length_sym), norm_integral, 'numpy')


def basis_norms(rho, d1, coeffs):
    lengths = [d1, 1 - 2*d1, d1]
    norm2 = np.zeros(len(rho))
    for (u, v), length in zip(coeffs, lengths):
        norm2 += norm_numpy(u, v, rho, length)
    return np.sqrt(norm2)


def solve(n, d1, a, points, values, m, eps, step):
    rho = model_eigenvalues(n, d1, a, eps, step)
    coeffs = coefficients(rho, d1, a)
    norms = basis_norms(rho, d1, coeffs)

    def basis(x, part):
        starts = [0, d1, 1-d1]
        u, v = coeffs[part]
        t = rho[:, None] * (x[None, :] - starts[part])
        return (u[:, None]*np.cos(t) + v[:, None]*np.sin(t)) / norms[:, None]

    Q = potential_matrix(n, basis, points, values, d1, m)
    A = np.diag(rho**2) + Q
    eigenvalues, eigenvectors = eigh(A)
    return rho**2, eigenvalues, eigenvectors, basis
