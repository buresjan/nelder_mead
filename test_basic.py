import numpy as np
from nelder_mead import nelder_mead
from scipy.optimize import minimize


def compare_methods(f, x_start, nelder_mead_args, scipy_args, description):
    # Run custom Nelder-Mead
    solution, fval = nelder_mead(f, x_start, **nelder_mead_args)
    # Run SciPy Nelder-Mead
    res_scipy = minimize(f, x_start, method="Nelder-Mead", options=scipy_args)

    print(f"Comparison for ({description}):")
    print(f"Custom Nelder-Mead ({description}): Solution = {solution}, fval = {fval}")
    print(
        f"SciPy Nelder-Mead ({description}): Solution = {res_scipy.x}, fval = {res_scipy.fun}\n"
    )


if __name__ == "__main__":
    nelder_mead_args = {
        "step": 0.1,
        "no_improve_thr": 1e-6,
        "no_improv_break": 30,
        "max_iter": 200,
    }
    scipy_args = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 200}

    # 1. Quadratic
    def test_func1(x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

    compare_methods(
        test_func1, np.array([0.0, 0.0]), nelder_mead_args, scipy_args, "Quadratic"
    )

    # 2. Rosenbrock
    def test_func2(x):
        return sum(
            100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            for i in range(len(x) - 1)
        )

    compare_methods(
        test_func2, np.array([0.0, 0.0]), nelder_mead_args, scipy_args, "Rosenbrock"
    )

    # 3. Rastrigin  # custom method actually outperforms the one from SciPy with default params here
    def test_func3(x):
        A = 10
        return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

    compare_methods(
        test_func3, np.array([0.5, -0.5]), nelder_mead_args, scipy_args, "Rastrigin"
    )

    # 4. Plateau
    def test_func4(x):
        return np.floor(x[0]) + np.floor(x[1])

    compare_methods(
        test_func4, np.array([0.5, 1.5]), nelder_mead_args, scipy_args, "Plateau"
    )

    # 5. Custom
    def test_func5(x):
        return np.sin(x[0]) + np.cos(x[1]) + 1 / (1 + np.exp(-x[0] + x[1]))

    compare_methods(
        test_func5, np.array([0.0, 0.0]), nelder_mead_args, scipy_args, "Custom"
    )
