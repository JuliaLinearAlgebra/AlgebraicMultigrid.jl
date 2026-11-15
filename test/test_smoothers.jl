# In this file we test the build-in smoothers
using AlgebraicMultigrid, SparseArrays, LinearAlgebra
using Test
import AlgebraicMultigrid as AMG

@testset "Smoothers" begin

N = 50

# Make a diagonal dominant problem
A = sprand(N,N,0.05) + 5I
x0 = rand(N)
b = ones(N)

@testset "Mildly non-symmetric matrix with $smoother" for smoother in [
    AMG.Jacobi(1 / 6, iter=500),
    GaussSeidel(ForwardSweep(), 100),
    GaussSeidel(BackwardSweep(), 100),
    GaussSeidel(SymmetricSweep(), 100),
    SOR(0.5, ForwardSweep(), 100),
    SOR(0.5, BackwardSweep(), 100),
    SOR(0.5, SymmetricSweep(), 100),
]
    x = copy(x0)
    smoother(A, x, b, AlgebraicMultigrid.NoSymmetry())
    @test A*x ≈ b
end

@testset "Symmetric matrix with $smoother"  for smoother in [
    AMG.Jacobi(4 / 5, iter=2),
    GaussSeidel(SymmetricSweep(), iter=2),
    SOR(0.5; iter=2),
]
    A_sym = poisson(N)

    x_fast = copy(x0)
    s_fast = AlgebraicMultigrid.setup_smoother(smoother, A_sym, AlgebraicMultigrid.HermitianSymmetry())
    AlgebraicMultigrid.smooth!(x_fast, s_fast, b)

    x_general = copy(x0)
    s_general = AlgebraicMultigrid.setup_smoother(smoother, A_sym, AlgebraicMultigrid.NoSymmetry())
    AlgebraicMultigrid.smooth!(x_general, s_general, b)

    @test x_fast ≈ x_general
end


@testset "Jacobi with an iterate wider than the matrix ($(nameof(typeof(sym))))" for sym in (
    AlgebraicMultigrid.NoSymmetry(),
    AlgebraicMultigrid.HermitianSymmetry(),
)
    A_dd = poisson(N) + 2I   # symmetric and diagonally dominant, so Jacobi converges quickly
    jacobi = AMG.Jacobi(2 / 3, iter = 200)

    # Float32 matrix with a Float64 iterate: computed in Float64, not truncated to Float32.
    x = zeros(N)
    jacobi(Float32.(A_dd), x, b, sym)
    @test A_dd * x ≈ b

    # ... without allocating a new buffer.
    s = AlgebraicMultigrid.setup_smoother(jacobi, Float32.(A_dd), sym)
    smooth_allocs(x) = (AlgebraicMultigrid.smooth!(x, s, b); @allocated AlgebraicMultigrid.smooth!(x, s, b))
    @test smooth_allocs(zeros(N)) == 0

    # Real matrix with a complex iterate.
    bc = complex.(b, 2 .* b)
    x = zeros(ComplexF64, N)
    jacobi(A_dd, x, bc, sym)
    @test A_dd * x ≈ bc
end


@testset "Jacobi with a user-supplied buffer in a hierarchy" begin
    # The buffer is sized for the finest level; coarser levels must get their own.
    A_h = poisson(200)
    jacobi = AMG.Jacobi(2 / 3, zeros(200))
    ml = ruge_stuben(A_h; presmoother = jacobi, postsmoother = jacobi, symmetry = AlgebraicMultigrid.NoSymmetry())
    x = AMG._solve(ml, ones(200); reltol = 1e-10)
    @test A_h * x ≈ ones(200)
end

end
