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

end
