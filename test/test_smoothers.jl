# In this file we test the build-in smoothers
using AlgebraicMultigrid, SparseArrays, LinearAlgebra
using Test

@testset "Smoothers" begin

N = 100

# Make a diagonal dominant problem
A = sprand(100,100,0.05) + 5I
x0 = rand(100)
b = ones(100)

@testset "$smoother" for smoother in [
    GaussSeidel(ForwardSweep(), 100),
    GaussSeidel(BackwardSweep(), 100),
    GaussSeidel(SymmetricSweep(), 100),
    SOR(0.5, ForwardSweep(), 100),
    SOR(0.5, BackwardSweep(), 100),
    SOR(0.5, SymmetricSweep(), 100),
]
    x = copy(x0)
    smoother(A, x, b)
    @test A*x ≈ b
end

end

@testset "FastSymmetricGaussSeidelSmoother" begin

using Random: seed!
seed!(42)

N = 50
A_sym = poisson(N)

@testset "fast smoother matches general smoother after one sweep" begin
    x0 = rand(N)
    b = rand(N)

    x_fast = copy(x0)
    s_fast = AlgebraicMultigrid.setup_smoother(GaussSeidel(SymmetricSweep(), 1), Symmetric(A_sym))
    ldiv!(x_fast, s_fast, b)

    x_general = copy(x0)
    s_general = AlgebraicMultigrid.setup_smoother(GaussSeidel(SymmetricSweep(), 1), A_sym)
    ldiv!(x_general, s_general, b)

    @test x_fast ≈ x_general
end

@testset "fast smoother converges on SPD matrix" begin
    b = A_sym * ones(N)
    x = zeros(N)
    s = AlgebraicMultigrid.setup_smoother(GaussSeidel(SymmetricSweep(), 200), Symmetric(A_sym))
    ldiv!(x, s, b)
    @test norm(A_sym * x - b) / norm(b) < 1e-2
end

end

@testset "FastSymmetricSORSmoother" begin

using Random: seed!
seed!(42)

N = 50
A_sym = spdiagm(0 => 2*ones(N), -1 => -ones(N-1), 1 => -ones(N-1))

@testset "Symmetric wrapper dispatches to fast path" begin
    s = AlgebraicMultigrid.setup_smoother(SOR(1.0, SymmetricSweep()), Symmetric(A_sym))
    @test s isa AlgebraicMultigrid.FastSymmetricSORSmoother
end

@testset "fast SSOR matches general SSOR after one sweep" begin
    x0 = rand(N)
    b = rand(N)

    x_fast = copy(x0)
    s_fast = AlgebraicMultigrid.setup_smoother(SOR(1.0, SymmetricSweep(), 1), Symmetric(A_sym))
    ldiv!(x_fast, s_fast, b)

    x_general = copy(x0)
    s_general = AlgebraicMultigrid.setup_smoother(SOR(1.0, SymmetricSweep(), 1), A_sym)
    ldiv!(x_general, s_general, b)

    @test x_fast ≈ x_general
end

@testset "fast SSOR converges on SPD matrix" begin
    b = A_sym * ones(N)
    x = zeros(N)
    s = AlgebraicMultigrid.setup_smoother(SOR(1.0, SymmetricSweep(), 200), Symmetric(A_sym))
    ldiv!(x, s, b)
    @test norm(A_sym * x - b) / norm(b) < 1e-2
end

end
