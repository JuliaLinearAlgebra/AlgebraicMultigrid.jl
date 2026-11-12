# Solver tests over a set of MatrixDepot.jl test matrices, symmetric and non-symmetric
# (https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/issues/109).
#
# The matrices come from MatrixDepot's built-in generators, called directly. The
# `matrixdepot("name", ...)` lookup and the SuiteSparse collection need MatrixDepot's
# online index, which cannot be downloaded from CI runners.
using AlgebraicMultigrid, SparseArrays, LinearAlgebra, Random
using IterativeSolvers, MatrixDepot
using Test
import AlgebraicMultigrid as AMG

# (name, matrix, solvers expected to fail on it)
const MATRIXDEPOT_CASES = [
    # Symmetric
    ("poisson(32)",            MatrixDepot.poisson(Float64, 32),                  ()),
    ("wathen(15, 15)",         MatrixDepot.wathen(Float64, 15, 15),               ()),
    ("blur(16)",               MatrixDepot.blur(Float64, 16),                     ()),
    ("tridiag(500, -1, 3, -1)", sparse(MatrixDepot.tridiag(Float64, 500, -1, 3, -1)), ()),
    # Non-symmetric
    ("tridiag(500, -2, 4, -1)", sparse(MatrixDepot.tridiag(Float64, 500, -2, 4, -1)), ()),
    # Smoothed aggregation diverges on strongly non-symmetric matrices: its hierarchy
    # (R = Pᵀ, Jacobi-smoothed prolongation) is built for the symmetric case.
    ("tridiag(500, -4, 5, -1)", sparse(MatrixDepot.tridiag(Float64, 500, -4, 5, -1)), (:smoothed_aggregation,)),
    # Singular (constant null space); the right-hand side is kept in the range of A.
    ("neumann(32)",            MatrixDepot.neumann(Float64, 32),                  ()),
]

const MATRIXDEPOT_SOLVERS = (
    ruge_stuben = ruge_stuben,
    smoothed_aggregation = smoothed_aggregation,
)

@testset "MatrixDepot matrices" begin
    @testset "$name" for (name, A, expected_failures) in MATRIXDEPOT_CASES
        n = size(A, 1)
        x_exact = rand(Xoshiro(0), n)
        x_exact .-= sum(x_exact) / n   # orthogonal to the constant vector, for neumann
        b = A * x_exact
        tol = 1e-8

        @testset "$solver_name" for (solver_name, solver) in pairs(MATRIXDEPOT_SOLVERS)
            broken = solver_name in expected_failures
            ml = solver(A)

            # Standalone multigrid iteration
            x, residuals = AMG._solve(ml, b; maxiter = 100, reltol = tol, log = true)
            @test norm(A * x - b) <= 10tol * norm(b) broken = broken

            # As a preconditioner: CG for symmetric matrices, GMRES otherwise
            P = aspreconditioner(ml)
            if issymmetric(A)
                x, history = cg(A, b; Pl = P, reltol = tol, maxiter = 100, log = true)
            else
                x, history = gmres(A, b; Pl = P, reltol = tol, maxiter = 100, log = true)
            end
            @test history.isconverged && norm(A * x - b) <= 10tol * norm(b) broken = broken
        end
    end
end
