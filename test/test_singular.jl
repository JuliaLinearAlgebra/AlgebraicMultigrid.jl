# Coarse solvers on numerically singular matrices. A singular problem (e.g. a pure
# Neumann problem) has a coarsest matrix that is singular only up to rounding; the
# coarse solve must treat its tiny singular values as zero, not divide by them.
using AlgebraicMultigrid, SparseArrays, LinearAlgebra, Random
using MatrixDepot
using Test
import AlgebraicMultigrid as AMG

@testset "Singular coarse problems" begin
    @testset "Coarse solver on a numerically singular matrix: $name" for (name, solver) in (
        ("Pinv", AMG.Pinv),
        ("QRSolver", AMG.QRSolver),
    )
        rng = Xoshiro(1)
        U = Matrix(qr(randn(rng, 6, 6)).Q)
        V = Matrix(qr(randn(rng, 6, 6)).Q)
        # σmin/σmax = 2.5e-14: singular for practical purposes, but above the default
        # `pinv` cutoff (n⋅eps⋅σmax ≈ 5e-15), which would invert it.
        σ = [4.0, 3.0, 2.0, 1.0, 0.5, 1e-13]
        A = sparse(U * Diagonal(σ) * V')
        c = [1.0, -2.0, 0.5, 1.5, -1.0]
        # Right-hand side in the range of A, plus a tiny component along the null
        # direction, as a coarse residual would carry.
        b = U[:, 1:5] * c + 1e-13 * U[:, 6]
        x_min_norm = V[:, 1:5] * (c ./ σ[1:5])

        x = zeros(6)
        solver(A)(x, b)
        @test norm(A * x - b) <= 1e-8 * norm(b)
        # Inverting σmin would add a component of size 1 along V[:, 6].
        @test norm(x) <= 1.5 * norm(x_min_norm)
        if solver === AMG.Pinv
            @test x ≈ x_min_norm rtol = 1e-6
        end
    end

    @testset "neumann(32) with $sname and $cname" for (sname, s) in (
            ("ruge_stuben", ruge_stuben), ("smoothed_aggregation", smoothed_aggregation)),
        (cname, cs) in (("Pinv", AMG.Pinv), ("QRSolver", AMG.QRSolver))

        A = MatrixDepot.neumann(Float64, 32)   # singular, constant null space
        n = size(A, 1)
        x_exact = rand(Xoshiro(0), n)
        x_exact .-= sum(x_exact) / n
        b = A * x_exact
        ml = s(A; coarse_solver = cs)
        x = AMG._solve(ml, b; maxiter = 50, reltol = 1e-8)
        @test norm(A * x - b) <= 1e-7 * norm(b)
    end
end
