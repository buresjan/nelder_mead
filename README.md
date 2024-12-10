# Custom Nelder-Mead Optimization Algorithm

This repository contains a custom implementation of the Nelder-Mead optimization algorithm designed for unconstrained and constrained optimization problems. The implementation leverages parallelization to improve performance, especially for expensive objective functions. The repository also includes tests to demonstrate the algorithm's capabilities and compare it to the SciPy implementation.

## Features

- **Custom Nelder-Mead Implementation**: A fully modular and parallelized implementation of the Nelder-Mead algorithm.
- **Support for Constraints**: Handles constrained optimization using an extreme barrier function approach.
- **Performance-Oriented**: Optimized for cases where the objective function is computationally expensive.
- **Extensive Testing**: Includes various test cases to validate and demonstrate the algorithm's capabilities.

## Files

### `nelder_mead.py`
The main implementation of the Nelder-Mead optimization algorithm. Key features include:
- Parallel evaluation of candidate points.
- Support for reflection, expansion, contraction, and shrinking operations.
- Convergence criteria based on user-defined thresholds.

### `test_basic.py`
A basic test comparing the custom Nelder-Mead implementation with SciPy's implementation using a simple quadratic function:
- Ensures correctness on a simple convex problem.
- Validates that the solution and function values are comparable.

### `test_time.py`
A test designed to compare the runtime of the custom Nelder-Mead implementation and SciPy's implementation:
- Simulates a computationally expensive objective function with artificial delays.
- Demonstrates the performance advantage of the parallelized implementation.

### `test_constraints.py`
A test showcasing the custom implementation's ability to handle constrained optimization problems using an extreme barrier function:
- Solves a constrained problem with a known optimum.
- Verifies that the algorithm converges to the correct solution within the feasible domain.

## Example Usage

### Running the Custom Nelder-Mead Implementation

```python
import numpy as np
from nelder_mead import nelder_mead

# Define an objective function
def objective(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# Initial guess
x_start = np.array([0.0, 0.0])

# Run the optimizer
solution, fval = nelder_mead(objective, x_start, step=0.1, max_iter=100, verbose=True)

print(f"Optimal solution: {solution}")
print(f"Function value: {fval}")
```

## Running Tests

To run the tests, simply execute the test scripts individually:

- **Basic Test**:
  ```bash
  python test_basic.py
  ```
- **Runtime Comparison**:
  ```bash
  python test_time.py
  ```
- **Constrained Optimization**:
  ```bash
  python test_constraints.py
  ```

## Dependencies

- Python 3.x
- NumPy
- SciPy