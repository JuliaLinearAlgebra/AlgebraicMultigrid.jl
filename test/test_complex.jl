@testset "Complex-valued problems" begin
    A = poisson((5, 5))
    Ac = A .* 1/√2 + A .* im/√2

    Random.seed!(1337)
    u = rand(Complex{Float64}, 5*5)
    b = Ac*u

    rs = ruge_stuben(Ac)
    usolrs = AlgebraicMultigrid._solve(rs, b)

    @test usolrs ≈ u

    @test_throws ErrorException smoothed_aggregation(Ac)
end
