import numpy as np
from assignment_2 import (
    neville_interpolation,
    forward_difference_table,
    newton_forward_interpolation,
    hermite_interpolation,
    cubic_spline
)

# Test Neville's Method
x_vals_neville = [3.6, 3.8, 3.9]
y_vals_neville = [1.675, 1.436, 1.318]
x_target_neville = 3.7
neville_result = neville_interpolation(x_vals_neville, y_vals_neville, x_target_neville)
print(f"Neville's interpolation result: {neville_result}")

# Test Newton's Forward Method
x_vals_newton = [7.2, 7.4, 7.5, 7.6]
y_vals_newton = [23.5492, 25.3913, 26.8224, 27.4589]
table_newton = forward_difference_table(x_vals_newton, y_vals_newton)

print("Forward difference table:")
print(table_newton)

x_target_newton = 7.3
newton_result = newton_forward_interpolation(x_vals_newton, table_newton, x_target_newton, 3)
print(f"Newton's interpolation result: {newton_result}")

# Test Hermite Interpolation
x_vals_hermite = [3.6, 3.8, 3.9]
y_vals_hermite = [1.675, 1.436, 1.318]
derivatives = [-1.195, -1.188, -1.182]
z, hermite_table = hermite_interpolation(x_vals_hermite, y_vals_hermite, derivatives)

print("\nHermite interpolation table:")
print("Z values:", z)
print("Hermite table:")
for i in range(len(z)):
    row = [z[i]] + [hermite_table[i, j] for j in range(len(z))]
    print(row)

# Test Cubic Spline Interpolation
x_vals_spline = [2, 5, 8, 10]
y_vals_spline = [3, 5, 7, 9]
A, b, x_solution = cubic_spline(x_vals_spline, y_vals_spline)

print("\nCubic Spline:")
print("Matrix A:")
for row in A:
    print([int(value) for value in row])

print("Vector b:", [int(value) for value in b])
print("Solution (x values):", [f"{value:.8f}" for value in x_solution])
