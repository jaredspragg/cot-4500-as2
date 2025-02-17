import numpy as np

# Neville's Method (2nd Degree Interpolation)
def neville_interpolation(x_values, y_values, x):
    n = len(x_values)
    P = np.zeros((n, n))

    for i in range(n):
        P[i, 0] = y_values[i]

    for j in range(1, n):
        for i in range(n - j):
            P[i, j] = ((x - x_values[i + j]) * P[i, j - 1] + (x_values[i] - x) * P[i + 1, j - 1]) / (x_values[i] - x_values[i + j])

    return P[0, n - 1]

# Newton's Forward Method 
def forward_difference_table(x_values, y_values):
    n = len(x_values)
    table = np.zeros((n, n))

    for i in range(n):
        table[i, 0] = y_values[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_values[i + j] - x_values[i])

    return table

def newton_forward_interpolation(x_values, table, x, degree):
    result = table[0, 0]
    for i in range(1, degree + 1):
        product = 1
        for j in range(i):
            product *= (x - x_values[j])
        result += product * table[0, i]

    return result

# Hermite Interpolation Method
def hermite_interpolation(x_values, y_values, derivatives):
    n = len(x_values)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = z[2 * i + 1] = x_values[i]
        Q[2 * i, 0] = Q[2 * i + 1, 0] = y_values[i]
        Q[2 * i + 1, 1] = derivatives[i]
        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])

    return z, Q

# Cubic Spline Interpolation
def cubic_spline(x_values, y_values):
    n = len(x_values)

    A = np.zeros((n, n))
    b = np.zeros(n)

    h = np.diff(x_values)
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y_values[i + 1] - y_values[i]) - (3 / h[i - 1]) * (y_values[i] - y_values[i - 1])

    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    b[1:n - 1] = alpha[1:n - 1]

    # Solve for x
    x_solution = np.linalg.solve(A, b)

    return A, b, x_solution

def main():
    # Neville's Method
    x_vals_neville = [3.6, 3.8, 3.9]
    y_vals_neville = [1.675, 1.436, 1.318]
    x_target = 3.7
    neville_result = neville_interpolation(x_vals_neville, y_vals_neville, x_target)
    print(f"{neville_result}\n")

    # Newton's Forward Method
    x_vals_newton = [7.2, 7.4, 7.5, 7.6]
    y_vals_newton = [23.5492, 25.3913, 26.8224, 27.4589]
    table = forward_difference_table(x_vals_newton, y_vals_newton)

    for value in table[0, 1:]:
        print(value) 

    x_target_newton = 7.3
    newton_result_3 = newton_forward_interpolation(x_vals_newton, table, x_target_newton, 3)
    
    print("\n")
    print(f"{newton_result_3}\n")

    # Hermite Interpolation
    x_vals_hermite = [3.6, 3.8, 3.9]
    y_vals_hermite = [1.675, 1.436, 1.318]
    derivatives = [-1.195, -1.188, -1.182]
    z, hermite_table = hermite_interpolation(x_vals_hermite, y_vals_hermite, derivatives)

 
    hermite_output = []
    for i in range(len(z)):
        row = [z[i]] + [hermite_table[i, j] for j in range(len(z))]
        hermite_output.append(row)

    for row in hermite_output:
        print(f"[{', '.join(f'{value:.6f}' for value in row)}]")

    # Cubic Spline Interpolation
    x_vals_spline = [2, 5, 8, 10]
    y_vals_spline = [3, 5, 7, 9]
    A, b, x_solution = cubic_spline(x_vals_spline, y_vals_spline)

    print("\n")
    cubic_spline_output = []
    for row in A:
        cubic_spline_output.append(f"[{', '.join(f'{int(value)}' for value in row)}]")

    for row in cubic_spline_output:
        print(row)


    print(f"[{', '.join(f'{int(value)}' for value in b)}]")

    print(f"[{', '.join(f'{value:.8f}' for value in x_solution)}]")

if __name__ == "__main__":
    main()
