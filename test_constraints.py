import numpy as np
from nelder_mead import nelder_mead


def constrained_objective(x):
    """
    Objective function with constraints applied via an extreme barrier function.
    The known minimum is within the feasible region.
    """
    # Define constraints
    lower_bound = np.array([-5, -5])
    upper_bound = np.array([5, 5])

    # Extreme barrier: penalize infeasible points
    if np.any(x < lower_bound) or np.any(x > upper_bound):
        return 1e12  # Large penalty for infeasible points

    # Objective function: known minimum at (1, -1)
    return (x[0] - 1) ** 2 + (x[1] + 1) ** 2


if __name__ == "__main__":
    # Initial guess
    x_start = np.array([0.0, 0.0])

    # Parameters for Nelder-Mead
    nelder_mead_args = {
        "step": 0.5,
        "no_improve_thr": 1e-6,
        "no_improv_break": 50,
        "max_iter": 200,
        "verbose": True,
    }

    # Run the custom Nelder-Mead optimizer
    solution, fval = nelder_mead(constrained_objective, x_start, **nelder_mead_args)

    # Print results
    print(f"Solution found: {solution}")
    print(f"Function value at solution: {fval}")
    print(f"Function's optimum should be at (1, -1).")
