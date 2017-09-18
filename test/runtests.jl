using AMG
using Base.Test

@testset "Strength of connection" begin

# classical strength of connection
A = poisson(5)
X = strength_of_connection(Classical(0.2), A)
@test full(X) == [ 1.0  0.5  0.0  0.0  0.0
                   0.5  1.0  0.5  0.0  0.0
                   0.0  0.5  1.0  0.5  0.0
                   0.0  0.0  0.5  1.0  0.5
                   0.0  0.0  0.0  0.5  1.0 ]

end

@testset "Splitting" begin

# Ruge-Stuben splitting
S = poisson(7)
@test split_nodes(RS(), S) == [0, 1, 0, 1, 0, 1, 0]

srand(0)
S = sprand(10,10,0.1); S = S + S'
@test split_nodes(RS(), S) ==  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]

end

@testset "Interpolation" begin

# Direct Interpolation
using AMG
A = poisson(5)
splitting = [1,0,1,0,1]
P, R = AMG.direct_interpolation(A, A, splitting)
@test P ==  [ 1.0  0.0  0.0
              0.5  0.5  0.0
              0.0  1.0  0.0
              0.0  0.5  0.5
              0.0  0.0  1.0 ]

end
