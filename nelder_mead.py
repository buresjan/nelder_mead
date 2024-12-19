import numpy as np
import concurrent.futures
import os


def initialize_simplex(f, x_start, step, verbose):
    """
    Generate the initial simplex around the starting point.
    """
    dim = len(x_start)
    simplex = [x_start]

    # Generate simplex points
    for i in range(dim):
        x = np.copy(x_start)
        x[i] += step[i]
        simplex.append(x)

    simplex = np.array(simplex)

    # Evaluate scores in parallel
    scores = np.empty(len(simplex))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_index = {
            executor.submit(f, simplex[i]): i for i in range(len(simplex))
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            scores[index] = future.result()

    if verbose:
        print(f"Initialized simplex shape: {simplex.shape}")
        print(f"Initialized scores shape: {scores.shape}")

    return simplex, scores


def order_simplex(simplex, scores):
    """
    Order the simplex points and scores in ascending order of the function values.
    """
    scores = np.array(scores).flatten()  # Ensure 1D scores array
    order = np.argsort(scores)

    return simplex[order], scores[order]


def compute_centroid(simplex):
    """
    Compute the centroid of the simplex excluding the worst point.
    """
    return np.mean(simplex[:-1], axis=0)


def generate_candidate_points(centroid, worst, delta_e, delta_oc, delta_ic):
    """
    Generate candidate points for reflection, expansion, outside contraction, and inside contraction.
    """
    # Reflection (factor implicitly 1)
    x_r = centroid + (centroid - worst)
    # Expansion
    x_e = centroid + delta_e * (x_r - centroid)
    # Outside Contraction
    x_oc = centroid + delta_oc * (centroid - worst)
    # Inside Contraction
    x_ic = centroid + delta_ic * (worst - centroid)

    return {
        "reflection": x_r,
        "expansion": x_e,
        "outside_contraction": x_oc,
        "inside_contraction": x_ic,
    }


def shrink_point(i, simplex, best_point, gamma, f):
    """
    Shrink a single simplex point towards the best point and evaluate its score.
    """
    new_point = best_point + gamma * (simplex[i] - best_point)
    new_score = f(new_point)
    return i, new_point, new_score


def shrink_simplex(simplex, scores, gamma, f):
    """
    Shrink the simplex towards the best point in parallel.
    """
    best_point = simplex[0]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(shrink_point, i, simplex, best_point, gamma, f): i
            for i in range(1, len(simplex))
        }

        for future in concurrent.futures.as_completed(futures):
            i, new_point, new_score = future.result()
            simplex[i] = new_point
            scores[i] = new_score

    return simplex, scores


def nelder_mead(
    f,
    x_start,
    step=0.01,
    no_improve_thr=1e-6,
    no_improv_break=20,
    max_iter=1000,
    # Standard Nelder-Mead parameters
    delta_e=2.0,  # expansion coefficient
    delta_oc=0.5,  # outside contraction coefficient
    delta_ic=0.5,  # inside contraction coefficient
    gamma=0.5,  # shrink coefficient
    verbose=False,
    log=False,
):
    if not isinstance(step, list):
        step = [step]

    log_file = "log.txt"
    if log:
        if os.path.exists(log_file):
            os.remove(log_file)
        with open(log_file, "w") as f_log:
            f_log.write("Iteration\tBest Estimate\tCurrent Location\n")

    simplex, scores = initialize_simplex(f, x_start, step, verbose)
    prev_best = scores[0]
    no_improv = 0
    iter_count = 0
    best = scores[0]

    while True:
        # Check stopping criteria
        if iter_count:
            if max_iter and iter_count >= max_iter:
                return simplex[0], best
            if no_improv >= no_improv_break:
                return simplex[0], best

        # Order simplex by score
        simplex, scores = order_simplex(simplex, scores)
        best = scores[0]

        if log:
            with open(log_file, "a") as f_log:
                f_log.write(f"{iter_count}\t{best:.6f}\t{simplex[0]}\n")

        # Verbose output
        if verbose:
            print(
                f"Iteration {iter_count}: Best estimate = {simplex[0]}, "
                f"Function value = {best}, "
                f"Simplex: {simplex}"
            )

        iter_count += 1
        if best < prev_best - no_improve_thr:
            prev_best = best
            no_improv = 0
        else:
            no_improv += 1

        # Compute centroid (excluding the worst point)
        centroid = compute_centroid(simplex)

        # Generate candidate points
        candidates = generate_candidate_points(
            centroid, simplex[-1], delta_e, delta_oc, delta_ic
        )

        # Evaluate candidates in parallel
        candidate_keys = [
            "reflection",
            "expansion",
            "outside_contraction",
            "inside_contraction",
        ]
        candidate_values = [candidates[key] for key in candidate_keys]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_key = {
                executor.submit(f, candidate): key
                for key, candidate in zip(candidate_keys, candidate_values)
            }
            results = {}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()

        f_r = results["reflection"]
        f_e = results["expansion"]
        f_oc = results["outside_contraction"]
        f_ic = results["inside_contraction"]

        # Nelder-Mead logic:
        # 1. If reflection is better than the best point, try expansion
        if f_r < scores[0]:
            if f_e < f_r:
                # Expansion is better than reflection
                simplex[-1] = candidates["expansion"]
                scores[-1] = f_e
                continue
            else:
                # Keep reflection
                simplex[-1] = candidates["reflection"]
                scores[-1] = f_r
                continue

        # 2. If reflection is not better than best, but better than second-worst, accept reflection
        elif f_r < scores[-2]:
            simplex[-1] = candidates["reflection"]
            scores[-1] = f_r
            continue

        # 3. If reflection is worse or equal to second-worst but better than worst, do outside contraction
        elif f_r < scores[-1]:
            if f_oc <= f_r:
                # Outside contraction improved over worst
                simplex[-1] = candidates["outside_contraction"]
                scores[-1] = f_oc
                continue
            else:
                # Outside contraction didn't improve, shrink
                simplex, scores = shrink_simplex(simplex, scores, gamma, f)
                continue

        # 4. Reflection is not better than worst, try inside contraction
        else:
            if f_ic < scores[-1]:
                # Inside contraction improved over worst
                simplex[-1] = candidates["inside_contraction"]
                scores[-1] = f_ic
                continue
            else:
                # Inside contraction didn't improve, shrink
                simplex, scores = shrink_simplex(simplex, scores, gamma, f)
                continue


if __name__ == "__main__":
    pass
