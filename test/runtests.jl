using AMG
using Base.Test
using JLD

@testset "AMG Tests" begin

graph = load("test.jld")["G"]
ref_S = load("ref_S_test.jld")["G"]
ref_split = readdlm("ref_split_test.txt")

@testset "Strength of connection" begin

# classical strength of connection
A = poisson(5)
S, T = strength_of_connection(Classical(0.2), A)
@test full(S) == [ 1.0  0.5  0.0  0.0  0.0
                   0.5  1.0  0.5  0.0  0.0
                   0.0  0.5  1.0  0.5  0.0
                   0.0  0.0  0.5  1.0  0.5
                   0.0  0.0  0.0  0.5  1.0 ]
S, T = strength_of_connection(Classical(0.25), graph)
diff = S - ref_S
@test maximum(diff) < 1e-10

end

@testset "Splitting" begin

# Ruge-Stuben splitting
S = poisson(7)
@test split_nodes(RS(), S) == [0, 1, 0, 1, 0, 1, 0]
srand(0)
S = sprand(10,10,0.1); S = S + S'
@test split_nodes(RS(), S) ==  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

a = load("thing.jld")["G"]
S, T = AMG.strength_of_connection(AMG.Classical(0.25), a)
@test split_nodes(RS(), S) == [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
1, 0]

@test split_nodes(RS(), ref_S, ref_S') == Int.(vec(ref_split))

end

@testset "Interpolation" begin

# Direct Interpolation
using AMG
A = poisson(5)
splitting = [1,0,1,0,1]
P, R = AMG.direct_interpolation(A, copy(A), splitting)
@test P ==  [ 1.0  0.0  0.0
              0.5  0.5  0.0
              0.0  1.0  0.0
              0.0  0.5  0.5
              0.0  0.0  1.0 ]
A = load("thing.jld")["G"]
ml = ruge_stuben(A)
@test size(ml.levels[2].A, 1) == 19
end

@testset "Coarse Solver" begin
A = poisson(10)
b = A * ones(10)
@test sum(abs2, AMG.coarse_solver(AMG.Pinv(), A, b) - ones(10)) < 1e-6
end

@testset "Multilevel" begin
A = poisson(1000)
A = float.(A) #FIXME
ml = AMG.ruge_stuben(A)
@test length(ml) == 8
s = [1000, 500, 250, 125, 62, 31, 15]
n = [2998, 1498, 748, 373, 184, 91, 43]
for i = 1:7
    @test size(ml.levels[i].A, 1) == s[i]
    @test nnz(ml.levels[i].A) == n[i]
end
@test size(ml.final_A, 1) == 7
@test nnz(ml.final_A) == 19

A = load("randlap.jld")["G"]
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
@test round(AMG.operator_complexity(ml), 3) ≈ 1.142
@test round(AMG.grid_complexity(ml), 3) ≈ 1.190


end

@testset "Solver" begin
A = poisson(1000)
A = float.(A)
ml = ruge_stuben(A)
x = solve(ml, A * ones(1000))
@test sum(abs2, x - ones(1000)) < 1e-10

A = load("randlap.jld")["G"]
ml = ruge_stuben(A)
x = solve(ml, A * ones(100))
@test sum(abs2, x - zeros(100)) < 1e-10

end

end
