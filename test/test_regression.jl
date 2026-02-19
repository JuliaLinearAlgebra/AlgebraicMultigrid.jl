# In this file we test the build-in smoothers
using AlgebraicMultigrid, SparseArrays, LinearAlgebra
using Test

@testset "Regression for issue #24" begin
    A = include("onetoall.jl")
    ml = smoothed_aggregation(A)
    @test size(ml.levels[2].A) == (11,11)
    @test size(ml.final_A) == (2,2)
end

@testset "Regression for issue #26" begin
    A = poisson(10)
    s = GaussSeidel(SymmetricSweep(), 4)
    x = ones(size(A,1))
    b = zeros(size(A,1))
    s(A, x, b)
    @test sum(abs2, x - [0.176765; 0.353529; 0.497517; 0.598914;
                            0.653311; 0.659104; 0.615597; 0.52275;
                            0.382787; 0.203251]) < 1e-6
end

@testset "Regression for issue #46" begin
    for f in (
        (smoothed_aggregation, SmoothedAggregationAMG), 
        (ruge_stuben, RugeStubenAMG),
    )
        a = load("bug.jld2")["G"]
        ml = f[1](a)
        p = aspreconditioner(ml)
        b = zeros(size(a,1))
        b[1] = 1
        b[2] = -1
        @test sum(abs2, a * solve(a, b, f[2]()) - b) < 1e-10
        @test sum(abs2, a * cg(a, b, Pl = p, maxiter = 1000) - b) < 1e-10
    end
end

@testset "Regression for issue #31" begin
    for sz in [10, 5, 2]
        a = poisson(sz)
        ml = ruge_stuben(a)
        @test isempty(ml.levels)
        @test size(ml.final_A) == (sz,sz)
        @test AlgebraicMultigrid.operator_complexity(ml) == 1
        @test AlgebraicMultigrid.grid_complexity(ml) == 1

        a = poisson(sz)
        ml = smoothed_aggregation(a)
        @test isempty(ml.levels)
        @test size(ml.final_A) == (sz,sz)
        @test AlgebraicMultigrid.operator_complexity(ml) == 1
        @test AlgebraicMultigrid.grid_complexity(ml) == 1
    end
end

@testset "Regression for issue #56" begin
    # Issue #56
    X = poisson(27_000)+24.0*I
    ml = ruge_stuben(X)
    b = rand(27_000)
    @test AlgebraicMultigrid._solve(ml, b, reltol = 1e-10) ≈ X \ b rtol = 1e-10
end

@testset "Non-symmetric system (Issue #95)" begin
    N = 10000

    # Make a diagonal dominant problem
    A = sprand(N,N,0.001) + 5I
    b = ones(N)

    xrs = solve(A, b, RugeStubenAMG())
    @test A*xrs ≈ b rtol = 1.0e-8

    xsa = solve(A, b, SmoothedAggregationAMG())
    @test A*xsa ≈ b rtol = 1.0e-8
end
