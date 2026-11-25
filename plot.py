import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from eigenvalues import coefficients, basis_norms


selected = [1, 2, 3]
result_file = 'results.csv'


def main():
    with open(result_file, newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    d1 = float(rows[0]['d1'])
    a = float(rows[0]['a'])
    d2 = 1 - d1
    rho = np.sqrt([float(row['lambda_model']) for row in rows])
    coeffs = coefficients(rho, d1, a)
    norms = basis_norms(rho, d1, coeffs)
    starts = [0, d1, d2]
    borders = [0, d1, d2, 1]

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 6))

    for index, number in enumerate(selected):
        row = rows[number - 1]
        vector = np.array(json.loads(row['vector']))
        color = colors[index % len(colors)]

        for part in range(3):
            x = np.linspace(borders[part], borders[part + 1], 200)
            u, v = coeffs[part]
            t = rho[:, None] * (x[None, :] - starts[part])
            phi = (u[:, None] * np.cos(t) + v[:, None] * np.sin(t)) / norms[:, None]
            y = vector @ phi
            plt.plot(x, y, color=color,
                     label=f'Функция {number}, λ = {float(row["lambda_ritz"]):.6f}'
                     if part == 0 else None)

    plt.axvline(d1, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(d2, linestyle='--', color='gray', linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Приближённые собственные функции')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('graphs.png', dpi=160)
    plt.show()


if __name__ == '__main__':
    main()
