# Regression tests for https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/issues/95:
# a plain (unwrapped) non-symmetric matrix must not be handled as if it were symmetric.
using AlgebraicMultigrid, SparseArrays, LinearAlgebra
using Test
import AlgebraicMultigrid as AMG

# 2D convection-diffusion -Δu + c⋅∇u on an n×n grid, first-order upwind convection.
function convection_diffusion_2d(n; c = (60.0, 30.0))
    h = 1 / (n + 1)
    idx(i, j) = (j - 1) * n + i
    I, J, V = Int[], Int[], Float64[]
    for j in 1:n, i in 1:n
        k = idx(i, j)
        push!(I, k); push!(J, k); push!(V, 4 / h^2 + (abs(c[1]) + abs(c[2])) / h)
        for (di, dj, cc) in ((-1, 0, c[1]), (1, 0, -c[1]), (0, -1, c[2]), (0, 1, -c[2]))
            ii, jj = i + di, j + dj
            (1 <= ii <= n && 1 <= jj <= n) || continue
            push!(I, k); push!(J, idx(ii, jj)); push!(V, -1 / h^2 - max(cc, 0) / h)
        end
    end
    return sparse(I, J, V, n^2, n^2)
end

@testset "Non-symmetric matrices (issue #95)" begin
    A = convection_diffusion_2d(32)
    @test !issymmetric(A)
    b = A * ones(size(A, 1))

    @testset "Default symmetry follows the matrix" begin
        @test AMG.default_symmetry(A) isa AMG.NoSymmetry
        @test AMG.default_symmetry(poisson(10)) isa AMG.HermitianSymmetry
    end

    @testset "Smoother call without a symmetry hint smooths A, not Aᵀ" begin
        x = zeros(size(A, 1))
        GaussSeidel(ForwardSweep())(A, x, b)
        # One forward Gauss-Seidel sweep from zero solves the lower triangle of A.
        @test x ≈ LowerTriangular(Matrix(A)) \ b
    end

    @testset "$name converges with default settings" for (name, solver) in (
        ("ruge_stuben", ruge_stuben),
        ("smoothed_aggregation", smoothed_aggregation),
    )
        ml = solver(A)
        x, residuals = AMG._solve(ml, b; maxiter = 30, reltol = 1e-8, log = true)
        @test residuals[end] < 1e-8 * norm(b)
        @test norm(A * x - b) < 1e-6 * norm(b)
    end
end
