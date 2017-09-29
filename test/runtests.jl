using AMG
using Base.Test
using JLD

graph = load("test.jld")["G"]
ref_S = load("ref_S_test.jld")["G"]
ref_split = readdlm("ref_split_test.txt")

@testset "Strength of connection" begin

# classical strength of connection
A = poisson(5)
S = strength_of_connection(Classical(0.2), A)
@test full(S) == [ 1.0  0.5  0.0  0.0  0.0
                   0.5  1.0  0.5  0.0  0.0
                   0.0  0.5  1.0  0.5  0.0
                   0.0  0.0  0.5  1.0  0.5
                   0.0  0.0  0.0  0.5  1.0 ]
S = strength_of_connection(Classical(0.25), graph)
diff = S - ref_S
@test maximum(diff) < 1e-10

end

@testset "Splitting" begin

# Ruge-Stuben splitting
S = poisson(7)
@test split_nodes(RS(), S) == [0, 1, 0, 1, 0, 1, 0]
@show "buzz"
srand(0)
S = sprand(10,10,0.1); S = S + S'
@test split_nodes(RS(), S) ==  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

a = load("thing.jld")["G"]
S = AMG.strength_of_connection(AMG.Classical(0.25), a)
@test split_nodes(RS(), S) == [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
1, 0]

@test split_nodes(RS(), ref_S) == Int.(vec(ref_split))

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
@test length(ml.levels) == 8
s = [1000, 500, 250, 125, 62, 31, 15, 7]
n = [2998, 1498, 748, 373, 184, 91, 43, 19]
for i = 1:8
    @test size(ml.levels[i].A, 1) == s[i]
    @test nnz(ml.levels[i].A) == n[i]
end
end

@testset "Solver" begin
A = poisson(1000)
A = float.(A)
ml = ruge_stuben(A)
x = solve(ml, A * ones(1000))
@test sum(abs2, x - ones(1000)) < 1e-10
end
