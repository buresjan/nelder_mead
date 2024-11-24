import numpy as np


def initialize_simplex(f, x_start, step):
    """
    Generate the initial simplex around the starting point.

    Parameters:
    - f (function): The objective function to minimize.
    - x_start (array-like): Starting point for the algorithm.
    - step (float): Step size for generating the initial simplex.

    Returns:
    - tuple: The simplex as a numpy array and the corresponding function scores.
    """
    dim = len(x_start)
    simplex = [x_start]
    scores = [f(x_start)]

    for i in range(dim):
        x = np.copy(x_start)
        x[i] += step
        simplex.append(x)
        scores.append(f(x))

    return np.array(simplex), np.array(scores)


def order_simplex(simplex, scores):
    """
    Order the simplex points and scores in ascending order of the function values.

    Parameters:
    - simplex (ndarray): Array of simplex points.
    - scores (ndarray): Array of corresponding function values.

    Returns:
    - tuple: Ordered simplex and scores.
    """
    order = np.argsort(scores)
    return simplex[order], scores[order]


def compute_centroid(simplex):
    """
    Compute the centroid of the simplex excluding the worst point.

    Parameters:
    - simplex (ndarray): Array of simplex points.

    Returns:
    - ndarray: The centroid of the best points (excluding the worst).
    """
    return np.mean(simplex[:-1], axis=0)


def generate_candidate_points(centroid, worst, delta_e, delta_oc, delta_ic):
    """
    Generate candidate points: reflection, expansion, outside contraction, and inside contraction.

    Parameters:
    - centroid (ndarray): Centroid of the simplex.
    - worst (ndarray): Worst point in the simplex.
    - delta_e (float): Expansion coefficient.
    - delta_oc (float): Outside contraction coefficient.
    - delta_ic (float): Inside contraction coefficient.

    Returns:
    - dict: Candidate points as a dictionary with keys 'reflection', 'expansion',
      'outside_contraction', and 'inside_contraction'.
    """
    x_r = centroid + (centroid - worst)  # Reflection
    x_e = centroid + delta_e * (x_r - centroid)  # Expansion
    x_oc = centroid + delta_oc * (centroid - worst)  # Outside contraction
    x_ic = centroid + delta_ic * (worst - centroid)  # Inside contraction

    return {
        "reflection": x_r,
        "expansion": x_e,
        "outside_contraction": x_oc,
        "inside_contraction": x_ic,
    }


def shrink_simplex(simplex, scores, gamma, f):
    """
    Shrink the simplex towards the best point.

    Parameters:
    - simplex (ndarray): Array of simplex points.
    - scores (ndarray): Array of corresponding function values.
    - gamma (float): Shrink coefficient.
    - f (function): The objective function.

    Returns:
    - tuple: The updated simplex and scores after shrinkage.
    """
    best_point = simplex[0]
    for i in range(1, len(simplex)):
        simplex[i] = best_point + gamma * (simplex[i] - best_point)
        scores[i] = f(simplex[i])
    return simplex, scores


def nelder_mead(
    f,
    x_start,
    step=0.01,
    no_improve_thr=1e-8,
    no_improv_break=10,
    max_iter=100,
    delta_e=2.0,
    delta_oc=0.5,
    delta_ic=0.5,
    gamma=0.5,
):
    """
    Perform the Nelder-Mead optimization algorithm.

    Parameters:
    - f (function): The objective function to minimize.
    - x_start (array-like): Starting point for the optimization.
    - step (float): Step size for the initial simplex generation.
    - no_improve_thr (float): Threshold for improvement to reset the no improvement counter.
    - no_improv_break (int): Maximum iterations without improvement before stopping.
    - max_iter (int): Maximum number of iterations.
    - delta_e (float): Expansion coefficient.
    - delta_oc (float): Outside contraction coefficient.
    - delta_ic (float): Inside contraction coefficient.
    - gamma (float): Shrink coefficient.

    Returns:
    - tuple: Best point and the corresponding function value.
    """
    simplex, scores = initialize_simplex(f, x_start, step)
    prev_best = scores[0]
    no_improv = 0
    iter_count = 0

    while True:
        simplex, scores = order_simplex(simplex, scores)
        best = scores[0]

        # Check stopping criteria
        if max_iter and iter_count >= max_iter:
            return simplex[0], best
        if no_improv >= no_improv_break:
            return simplex[0], best

        iter_count += 1
        if best < prev_best - no_improve_thr:
            prev_best = best
            no_improv = 0
        else:
            no_improv += 1

        centroid = compute_centroid(simplex)
        candidates = generate_candidate_points(
            centroid, simplex[-1], delta_e, delta_oc, delta_ic
        )

        # Evaluate function values for candidate points
        f_r = f(candidates["reflection"])
        f_e = f(candidates["expansion"])
        f_oc = f(candidates["outside_contraction"])
        f_ic = f(candidates["inside_contraction"])

        # Apply the Nelder-Mead algorithm
        if scores[0] <= f_r < scores[-2]:  # Reflection
            simplex[-1] = candidates["reflection"]
            scores[-1] = f_r
            continue

        if f_r < scores[0]:  # Expansion
            if f_e < f_r:
                simplex[-1] = candidates["expansion"]
                scores[-1] = f_e
            else:
                simplex[-1] = candidates["reflection"]
                scores[-1] = f_r
            continue

        if f_r >= scores[-1]:  # Contraction
            if f_oc < scores[-1]:  # Outside contraction
                simplex[-1] = candidates["outside_contraction"]
                scores[-1] = f_oc
                continue
            if f_ic < scores[-1]:  # Inside contraction
                simplex[-1] = candidates["inside_contraction"]
                scores[-1] = f_ic
                continue

        simplex, scores = shrink_simplex(simplex, scores, gamma, f)


if __name__ == "__main__":
    pass
