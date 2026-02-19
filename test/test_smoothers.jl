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
