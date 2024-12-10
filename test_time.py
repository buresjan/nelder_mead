import numpy as np
from time import time, sleep
import random
from scipy.optimize import minimize
from nelder_mead import nelder_mead


def objective_with_delay(x):
    """
    Objective function with artificial delay to simulate a long evaluation time.
    """
    t = random.uniform(0.5, 1.0)  # Simulate a random delay between 0.05 and 0.1 seconds
    sleep(t)

    return (x[0] - 3) ** 2 + (x[1] + 1) ** 2


def compare_runtimes(nelder_mead_args, scipy_args, x_start):
    """
    Compare runtimes of the custom Nelder-Mead implementation and SciPy's implementation.
    """
    # Custom Nelder-Mead
    start_time = time()
    solution_custom, fval_custom = nelder_mead(
        objective_with_delay, x_start, **nelder_mead_args
    )
    custom_time = time() - start_time

    # SciPy Nelder-Mead
    start_time = time()
    res_scipy = minimize(
        objective_with_delay, x_start, method="Nelder-Mead", options=scipy_args
    )
    scipy_time = time() - start_time

    # Print results
    print(
        f"Custom Nelder-Mead: Solution = {solution_custom}, fval = {fval_custom}, Time = {custom_time:.2f}s"
    )
    print(
        f"SciPy Nelder-Mead: Solution = {res_scipy.x}, fval = {res_scipy.fun}, Time = {scipy_time:.2f}s"
    )


if __name__ == "__main__":
    # Running the test takes longer time, ~1min!
    nelder_mead_args = {
        "step": 0.1,
        "no_improve_thr": 1e-6,
        "no_improv_break": 20,
        "max_iter": 20,
    }
    scipy_args = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20}

    x_start = np.array([0.0, 0.0])

    compare_runtimes(nelder_mead_args, scipy_args, x_start)
