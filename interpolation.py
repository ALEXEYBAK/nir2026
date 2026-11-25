import csv
import numpy as np

def read_potential(filename):
    # В CSV два столбца: x,q
    x, q = [], []
    with open(filename, newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x.append(float(row[0]))
            q.append(float(row[1]))
    return np.array(x), np.array(q)

def interpolate(x, points, values):
    # Формула Лагранжа; при числе узлов > 100 - линейная интерполяция
    if len(points) > 100:
        return np.interp(x, points, values)

    result = np.zeros_like(x, dtype=float)
    for k in range(len(points)):
        term = np.ones_like(x, dtype=float) * values[k]
        for j in range(len(points)):
            if j != k:
                term *= (x - points[j]) / (points[k] - points[j])
        result += term
    return result
