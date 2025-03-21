# Algebraic Multigrid (AMG)


| **Build Status**|
|:----------------------------------------------------------------:|
| [![Build Status](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/workflows/CI/badge.svg)](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/actions?query=workflow%3ACI)  |

This package lets you solve sparse linear systems using Algebraic Multigrid (AMG). This works especially well for symmetric positive definite matrices.

## Usage

### Using the CommonSolve interface

This is highest level API. It internally creates the multilevel object 
and calls the multigrid cycling `_solve`. 

```julia
A = poisson(100); 
b = rand(100);
solve(A, b, RugeStubenAMG(), maxiter = 1, abstol = 1e-6)
```

### Multigrid cycling

```julia
using AlgebraicMultigrid

A = poisson(1000) # Creates a sample symmetric positive definite sparse matrix
ml = ruge_stuben(A) # Construct a Ruge-Stuben solver
# Multilevel Solver
# -----------------
# Operator Complexity: 1.9859906604402935
# Grid Complexity: 1.99
# No. of Levels: 8
# Coarse Solver: AMG.Pinv()
# Level     Unknowns     NonZeros
# -----     --------     --------
#     1         1000         2998 [50.35%]
#     2          500         1498 [25.16%]
#     3          250          748 [12.56%]
#     4          125          373 [ 6.26%]
#     5           62          184 [ 3.09%]
#     6           31           91 [ 1.53%]
#     7           15           43 [ 0.72%]
#     8            7           19 [ 0.32%]


AlgebraicMultigrid._solve(ml, A * ones(1000)) # should return ones(1000)
```

### As a Preconditioner
You can use AMG as a preconditioner for Krylov methods such as Conjugate Gradients.
```julia
import IterativeSolvers: cg
p = aspreconditioner(ml)
c = cg(A, A*ones(1000), Pl = p)
```


### As a preconditioner with LinearSolve.jl

`RugeStubenPreconBuilder` and `SmoothedAggregationPreconBuilder` work with the 
[`precs` API](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/#Specifying-Preconditioners)
of LinearSolve.jl

```julia
A = poisson( (100,100) )
u0= rand(size(A,1))
b=A*u0

prob = LinearProblem(A, b)
strategy = KrylovJL_CG(precs = RugeStubenPreconBuilder())
sol = solve(prob, strategy, atol=1.0e-14)

strategy = KrylovJL_CG(precs = SmoothedAggregationPreconBuilder())
sol = solve(prob, strategy, atol=1.0e-14)
```

## Features and Roadmap

This package currently supports:

AMG Styles:
* Ruge-Stuben Solver
* Smoothed Aggregation (SA)

Strength of Connection:
* Classical Strength of Connection
* Symmetric Strength of Connection

Smoothers:
* Gauss Seidel (Symmetric, Forward, Backward)
* Damped Jacobi

Cycling:
* V, W and F cycles

In the future, this package will support:
1. Other splitting methods (like CLJP)
2. SOR smoother
3. AMLI cycles

#### Acknowledgements
This package has been heavily inspired by the [`PyAMG`](http://github.com/pyamg/pyamg) project.
