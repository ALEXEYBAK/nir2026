import argparse
import csv
import json

from interpolation import read_potential
from eigenvalues import solve


def main():
    parser = argparse.ArgumentParser(description='Метод Ритца для задачи Штурма-Лиувилля')
    parser.add_argument('--d1', type=float, required=True)
    parser.add_argument('--a', type=float, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--m', type=int, default=100)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--step', type=float, default=0.03)
    parser.add_argument('--out', default='results.csv')
    args = parser.parse_args()

    points, values = read_potential(args.csv)
    model, lambdas, vectors, basis = solve(
        args.n, args.d1, args.a, points, values, args.m, args.eps, args.step
    )

    with open(args.out, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'd1', 'a', 'lambda_model', 'lambda_ritz', 'vector'])
        for k in range(args.n):
            writer.writerow([
                k + 1, args.d1, args.a, model[k], lambdas[k],
                json.dumps(vectors[:, k].tolist())
            ])

    print(' k      lambda_0          lambda_Ritz')
    for k in range(args.n):
        print(f'{k+1:2d}   {model[k]:14.7f}    {lambdas[k]:14.7f}')
    print('Результаты сохранены в', args.out)


if __name__ == '__main__':
    main()
