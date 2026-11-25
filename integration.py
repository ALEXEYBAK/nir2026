import numpy as np
from interpolation import interpolate


def simpson(values, h):
    # Составная формула Симпсона на 2m равных частях
    return h / 3 * (
        values[0] + values[-1]
        + 4 * np.sum(values[1:-1:2])
        + 2 * np.sum(values[2:-1:2])
    )


def potential_matrix(n, basis, points, values, d1, m):
    # Q_ij = integral_0^1 q(x)*phi_i(x)*phi_j(x) dx
    d2 = 1 - d1
    borders = [0, d1, d2, 1]
    Q = np.zeros((n, n))

    for part in range(3):
        left, right = borders[part], borders[part + 1]
        # Разбиваем интеграл дополнительно в узлах CSV, если они есть внутри.
        cuts = [left] + [p for p in points if left < p < right] + [right]
        for k in range(len(cuts) - 1):
            x = np.linspace(cuts[k], cuts[k + 1], 2 * m + 1)
            h = (cuts[k + 1] - cuts[k]) / (2 * m)
            q = interpolate(x, points, values)
            phi = basis(x, part)

            for i in range(n):
                for j in range(i, n):
                    f = q * phi[i] * phi[j]
                    Q[i, j] += simpson(f, h)

    for i in range(n):
        for j in range(i):
            Q[i, j] = Q[j, i]
    return Q
