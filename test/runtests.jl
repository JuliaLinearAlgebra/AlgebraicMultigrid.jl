using Compat, Compat.Test, Compat.LinearAlgebra
using Compat.SparseArrays, Compat.DelimitedFiles, Compat.Random
using IterativeSolvers, FileIO, AlgebraicMultigrid
import AlgebraicMultigrid: V, coarse_solver, Pinv, Classical

include("sa_tests.jl")

@testset "AlgebraicMultigrid Tests" begin

graph = include("test.jl")
ref_S = include("ref_S_test.jl")
ref_split = readdlm("ref_split_test.txt")

@testset "Strength of connection" begin

# classical strength of connection
A = poisson(5)
A = float.(A)
S, T = Classical(0.2)(A)
@test Matrix(S) == [ 1.0  0.5  0.0  0.0  0.0
                   0.5  1.0  0.5  0.0  0.0
                   0.0  0.5  1.0  0.5  0.0
                   0.0  0.0  0.5  1.0  0.5
                   0.0  0.0  0.0  0.5  1.0 ]
S, T = Classical(0.25)(graph)
diff = S - ref_S
@test maximum(diff) < 1e-10

end

@testset "Splitting" begin

# Ruge-Stuben splitting
S = poisson(7)
@test RS()(S) == [0, 1, 0, 1, 0, 1, 0]
srand(0)
S = sprand(10,10,0.1); S = S + S'
@test RS()(S) ==  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

a = include("thing.jl")
S, T = Classical(0.25)(a)
@test RS()(S) == [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
1, 0]

@test RS()(ref_S) == Int.(vec(ref_split))

end

@testset "Interpolation" begin

# Direct Interpolation
using AlgebraicMultigrid
A = poisson(5)
A = Float64.(A)
splitting = [1,0,1,0,1]
P, R = AlgebraicMultigrid.direct_interpolation(A, copy(A), splitting)
@test P ==  [ 1.0  0.0  0.0
              0.5  0.5  0.0
              0.0  1.0  0.0
              0.0  0.5  0.5
              0.0  0.0  1.0 ]
A = include("thing.jl")
ml = ruge_stuben(A)
@test size(ml.levels[2].A, 1) == 19
end

@testset "Coarse Solver" begin
A = float.(poisson(10))
b = A * ones(10)
@test sum(abs2, coarse_solver(Pinv(), A, b) - ones(10)) < 1e-6
end

@testset "Multilevel" begin
A = poisson(1000)
A = float.(A) #FIXME
ml = AlgebraicMultigrid.ruge_stuben(A)
@test length(ml) == 8
s = [1000, 500, 250, 125, 62, 31, 15]
n = [2998, 1498, 748, 373, 184, 91, 43]
for i = 1:7
    @test size(ml.levels[i].A, 1) == s[i]
    @test nnz(ml.levels[i].A) == n[i]
end
@test size(ml.final_A, 1) == 7
@test nnz(ml.final_A) == 19

A = include("randlap.jl")
ml = ruge_stuben(A)
@test length(ml) == 3
s = [100, 17]
n = [2066, 289]
for i = 1:2
    @test size(ml.levels[i].A, 1) == s[i]
    @test nnz(ml.levels[i].A) == n[i]
end
@test size(ml.final_A, 1) == 2
@test nnz(ml.final_A) == 4
@static if VERSION < v"0.7-"
    @test round(AlgebraicMultigrid.operator_complexity(ml), 3) ≈ 1.142
    @test round(AlgebraicMultigrid.grid_complexity(ml), 3) ≈ 1.190
else
    @test round(AlgebraicMultigrid.operator_complexity(ml), digits=3) ≈ 1.142
    @test round(AlgebraicMultigrid.grid_complexity(ml), digits=3) ≈ 1.190
end

include("gmg.jl")

A = float.(poisson(10^6))
ml = multigrid(A)
@test length(ml) == 10

end

@testset "Solver" begin
fsmoother = GaussSeidel(ForwardSweep())

A = poisson(1000)
A = float.(A)
ml = ruge_stuben(A)
x = solve(ml, A * ones(1000))
@test sum(abs2, x - ones(1000)) < 1e-8

ml = ruge_stuben(A, presmoother = fsmoother,
                     postsmoother = fsmoother)
x = solve(ml, A * ones(1000))
@test sum(abs2, x - ones(1000)) < 1e-8


A = include("randlap.jl")

ml = ruge_stuben(A, presmoother = fsmoother,
                    postsmoother = fsmoother)
x = solve(ml, A * ones(100))
@test sum(abs2, x - zeros(100)) < 1e-8

ml = ruge_stuben(A)
x = solve(ml, A * ones(100))
@test sum(abs2, x - zeros(100)) < 1e-6

end

@testset "Preconditioning" begin
A = include("thing.jl")
n = size(A, 1)
smoother = GaussSeidel(ForwardSweep())
ml = ruge_stuben(A, presmoother = smoother,
                    postsmoother = smoother)
p = aspreconditioner(ml)
b = zeros(n)
b[1] = 1
b[2] = -1
x = solve(p.ml, A * ones(n), maxiter = 1, tol = 1e-12)
diff = x - [  1.88664780e-16,   2.34982727e-16,   2.33917697e-16,
         8.77869044e-17,   7.16783490e-17,   1.43415460e-16,
         3.69199021e-17,   9.70950385e-17,   4.77034895e-17,
         3.77491328e-17,   5.07592420e-18,   6.32131628e-19,
        -1.60361276e-18,  -1.36749626e-16,  -4.16651794e-17,
        -3.48207590e-17,  -4.19334783e-17,  -4.60500098e-17,
        -6.91113945e-17,  -1.35997904e-16,  -9.85940056e-17,
         8.99433377e-17,   1.63924842e-16,  -2.43048120e-16,
        -3.21888624e-16,  -1.56389252e-16,  -3.66495834e-17,
         1.70606350e-16,   1.66788345e-16,  -3.26736922e-16,
        -3.39591125e-16,  -3.67075849e-16,  -3.89891523e-16,
        -4.34537627e-16,  -4.42863579e-16,  -6.62433333e-16,
        -5.56056397e-16,  -5.70242981e-16,  -6.48075679e-16,
        -6.58200572e-16,  -7.24886474e-16,  -7.55973538e-16,
        -6.76965535e-16,  -7.00643227e-16,  -6.23581397e-16,
        -7.03016682e-16]
@test sum(abs2, diff) < 1e-8
x = solve(p.ml, b, maxiter = 1, tol = 1e-12)
diff = x - [ 0.76347046, -0.5498286 , -0.2705487 , -0.15047352, -0.10248021,
        0.60292674, -0.11497073, -0.08460548, -0.06931461,  0.38230708,
       -0.055664  , -0.04854558, -0.04577031,  0.09964325,  0.01825624,
       -0.01990265, -0.02866185, -0.03049521,  0.03310897, -0.01709034,
       -0.02038031, -0.01325201, -0.01051535,  0.02992818,  0.01493605,
       -0.00633922, -0.01285614, -0.01155069, -0.01095907,  0.04415807,
        0.02213755,  0.018686  ,  0.02625713,  0.02007781,  0.01898018,
        0.02107552,  0.01909623,  0.01874986,  0.01852736,  0.01844719,
        0.01841821,  0.01841695,  0.01953195,  0.01885713,  0.01864432,
        0.0185079 ]
@test sum(abs2, diff) < 1e-8
x = cg(A, b, Pl = p)
diff = x - [ 0.82365077, -0.537589  , -0.30632349, -0.19370186, -0.14773294,
        0.68489145, -0.15550115, -0.1278148 , -0.11197922,  0.45362483,
       -0.08577219, -0.08598307, -0.08477946,  0.12985118,  0.02805496,
       -0.03907565, -0.05950957, -0.06544269,  0.05446686, -0.047537  ,
       -0.05203899, -0.04685981, -0.04491762,  0.05639249,  0.02792704,
       -0.02282528, -0.04062864, -0.04321821, -0.0441893 ,  0.07593055,
        0.05212038,  0.04464215,  0.05835841,  0.05079815,  0.04830733,
        0.05272397,  0.05028666,  0.0494817 ,  0.04960952,  0.0496615 ,
        0.04968258,  0.04968737,  0.05105749,  0.05009268,  0.04972329,
        0.04970173]
@test sum(abs2, diff) < 1e-8

# Symmetric GaussSeidel Smoothing

ml = ruge_stuben(A)
p = aspreconditioner(ml)

x = cg(A, b, Pl = p, maxiter = 100_000, tol = 1e-6)
diff = x - [0.823762, -0.537478, -0.306212, -0.19359, -0.147621, 0.685002,
            -0.155389, -0.127703, -0.111867, 0.453735, -0.0856607, -0.0858715,
            -0.0846678, 0.129962, 0.0281662, -0.0389642, -0.0593981, -0.0653311,
            0.0545782, -0.0474255, -0.0519275, -0.0467483, -0.0448061, 0.056504,
            0.0280386, -0.0227138, -0.0405172, -0.0431067, -0.0440778, 0.076042,
            0.052232, 0.0447537, 0.05847, 0.0509098, 0.0484189, 0.0528356,
            0.0503983, 0.0495933, 0.0497211, 0.0497731, 0.0497942, 0.049799,
            0.0511691, 0.0502043, 0.0498349, 0.0498134]
@test sum(abs2, diff) < 1e-8

x = solve(ml, b, maxiter = 1, tol = 1e-12)
diff = x - [0.775725, -0.571202, -0.290989, -0.157001, -0.106981, 0.622652,
            -0.122318, -0.0891874, -0.0709834, 0.392621, -0.055544, -0.0507485,
            -0.0466376, 0.107175, 0.0267468, -0.0200843, -0.0282827, -0.0299929,
            0.0420468, -0.0175585, -0.0181318, -0.0121591, -0.00902523, 0.0394795,
            0.019981, -0.00270916, -0.0106855, -0.0093661, -0.00837619, 0.052532,
            0.0301423, 0.0248904, 0.0333098, 0.0262179, 0.0246211, 0.026778,
            0.0245746, 0.0238448, 0.0233892, 0.0231593, 0.0230526, 0.0229771,
            0.0247913, 0.0238555, 0.0233681, 0.023096]
@test sum(abs2, diff) < 1e-8


end

@testset "Precision" begin

a = poisson(100)
b = rand(size(a,1))

# Iterate through all types
for (T,V) in ((Float64, Float64), (Float32,Float32),
        (Float64,Float32), (Float32,Float64))
        a = T.(a)
        ml = smoothed_aggregation(a)
        b = V.(b)
        c = cg(a, b, maxiter = 10)
        @test eltype(solve(ml, b)) == eltype(c)
end    

end

# Smoothed Aggregation
@testset "Smoothed Aggregation" begin

@testset "Symmetric Strength of Connection" begin

    test_symmetric_soc()
end

@testset "Standard Aggregation" begin

    test_standard_aggregation()
end

@testset "Fit Candidates" begin
    test_fit_candidates()
end

@testset "Approximate Spectral Radius" begin
    test_approximate_spectral_radius()
end
end

@testset "Gauss Seidel" begin
    test_gauss_seidel()
end

@testset "Jacobi Prolongation" begin
    test_jacobi_prolongator()
end

@testset "Int32 support" begin
    a = sparse(Int32.(1:10), Int32.(1:10), rand(10))
    @inferred smoothed_aggregation(a)
end

# Issue #24
nodes_not_agg()

# Issue #26
test_symmetric_sweep()

# Issue #31
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
