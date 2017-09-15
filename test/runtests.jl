using AMG
using Base.Test

# classical strength of connection
A = poisson(5)
@test full(classical(A, 0.2)) == [ 1.0  0.5  0.0  0.0  0.0
                                   0.5  1.0  0.5  0.0  0.0
                                   0.0  0.5  1.0  0.5  0.0
                                   0.0  0.0  0.5  1.0  0.5
                                   0.0  0.0  0.0  0.5  1.0 ]

# Ruge-Stuben splitting
S = poisson(7)
S = S - spdiagm(diag(S)) #FIXME
@test RS(S) == [0, 1, 0, 1, 0, 1, 0]

srand(0)
S = sprand(10,10,0.1); S = S + S'
S = S - spdiagm(diag(S))
@test RS(S) ==  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
