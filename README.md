# Algebraic Multigrid (AMG)

[![Build Status](https://travis-ci.org/JuliaLinearAlgebra/AlgebraicMultigrid.jl.svg?branch=master)](https://travis-ci.org/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaLinearAlgebra/AlgebraicMultigrid.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaLinearAlgebra/AlgebraicMultigrid.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaLinearAlgebra/AlgebraicMultigrid.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaLinearAlgebra/AlgebraicMultigrid.jl?branch=master)
[![Build status](https://ci.appveyor.com/api/projects/status/0wnj4lhk1rvl5pjp?svg=true)](https://ci.appveyor.com/project/ranjanan/algebraicmultigrid-jl)

This package lets you solve sparse linear systems using Algebraic Multigrid (AMG). This works especially well for symmetric positive definite matrices.

## Usage

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


solve(ml, A * ones(1000)) # should return ones(1000)
```

### As a Preconditioner
You can use AMG as a preconditioner for Krylov methods such as Conjugate Gradients.
```julia
import IterativeSolvers: cg
p = aspreconditioner(ml)
c = cg(A, A*ones(1000), Pl = p)
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
