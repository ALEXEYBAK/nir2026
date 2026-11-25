import argparse
import csv
import json

import numpy as np

from eigenvalues import characteristic, coefficients, basis_norms
from integration import potential_matrix
from interpolation import read_potential


def main():
    parser = argparse.ArgumentParser(description='Проверка результатов метода Ритца')
    parser.add_argument('--results', default='results.csv')
    parser.add_argument('--csv', default='potential.csv')
    parser.add_argument('--m', type=int, default=400)
    args = parser.parse_args()

    with open(args.results, newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    n = len(rows)
    d1 = float(rows[0]['d1'])
    d2 = 1 - d1
    a = float(rows[0]['a'])
    lambda0 = np.array([float(row['lambda_model']) for row in rows])
    lambdas = np.array([float(row['lambda_ritz']) for row in rows])
    vectors = np.column_stack([np.array(json.loads(row['vector'])) for row in rows])

    rho = np.sqrt(lambda0)
    coeffs = coefficients(rho, d1, a)
    norms = basis_norms(rho, d1, coeffs)
    starts = [0, d1, d2]
    borders = [0, d1, d2, 1]

    def basis(x, part, derivative=False):
        x = np.atleast_1d(x)
        u, v = coeffs[part]
        t = rho[:, None] * (x[None, :] - starts[part])
        if derivative:
            result = rho[:, None] * (-u[:, None] * np.sin(t) + v[:, None] * np.cos(t))
        else:
            result = u[:, None] * np.cos(t) + v[:, None] * np.sin(t)
        return result / norms[:, None]

    # Матрица скалярных произведений собственных функций.
    gram = np.zeros((n, n))
    for part in range(3):
        left, right = borders[part], borders[part + 1]
        x = np.linspace(left, right, 2 * args.m + 1)
        y = vectors.T @ basis(x, part)
        weights = np.ones(len(x))
        weights[1:-1:2] = 4
        weights[2:-1:2] = 2
        h = (right - left) / (2 * args.m)
        gram += h / 3 * (y * weights) @ y.T

    diagonal_error = np.max(np.abs(np.diag(gram) - 1))
    off_diagonal = gram - np.diag(np.diag(gram))
    orthogonality_error = np.max(np.abs(off_diagonal))

    with open('orthogonality.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['k'] + list(range(1, n + 1)))
        for k in range(n):
            writer.writerow([k + 1] + gram[k].tolist())

    # Краевые условия и условия сопряжения.
    def values(x, part, derivative=False):
        return (vectors.T @ basis(x, part, derivative))[:, 0]

    boundary_error = max(np.max(np.abs(values(0, 0))),
                         np.max(np.abs(values(1, 2))))
    interface_error = max(
        np.max(np.abs(values(d1, 1) - a * values(d1, 0))),
        np.max(np.abs(values(d1, 1, True) - values(d1, 0, True) / a)),
        np.max(np.abs(values(d2, 2) - values(d2, 1) / a)),
        np.max(np.abs(values(d2, 2, True) - a * values(d2, 1, True)))
    )

    # Невязки характеристического уравнения и матричной задачи.
    root_error = max(abs(characteristic(r, d1, a)) for r in rho)
    points, q = read_potential(args.csv)
    q_matrix = potential_matrix(n, basis, points, q, d1, args.m)
    matrix = np.diag(lambda0) + q_matrix
    residuals = np.linalg.norm(matrix @ vectors - vectors * lambdas, axis=0)

    print(f'Невязка корней характеристического уравнения с q=0: {root_error:.3e}')
    print(f'Максимальная матричная невязка:                  {np.max(residuals):.3e}')
    print(f'max ошибка нормировки:                 {diagonal_error:.3e}')
    print(f'max ошибка ортогональности:           {orthogonality_error:.3e}')
    print(f'Невязка краевых условий:                        {boundary_error:.3e}')
    print(f'Невязка условий сопряжения:                     {interface_error:.3e}')
    print('\nМатрица скалярных произведений для первых 5 функций):')
    print(np.array2string(gram[:5, :5], precision=5, suppress_small=True))
    print('\nПолная матрица сохранена в orthogonality.csv')


if __name__ == '__main__':
    main()
