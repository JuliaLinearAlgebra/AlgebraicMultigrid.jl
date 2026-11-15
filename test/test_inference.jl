@testset "Type stability" begin
    A = poisson((20, 20))
    b = rand(size(A, 1))
    Ac = poisson(ComplexF64, (20, 20))
    bc = rand(ComplexF64, size(Ac, 1))

    for (A, b, mls) in ((A, b, (ruge_stuben(A), smoothed_aggregation(A))),
                        (Ac, bc, (ruge_stuben(Ac),)))
        for ml in mls
            # Preconditioner application, called once per Krylov iteration
            p = aspreconditioner(ml)
            x = zero(b)
            @test (@inferred ldiv!(x, p, b)) === x
            @inferred p \ b
            for cycle in (AlgebraicMultigrid.V(), AlgebraicMultigrid.W(), AlgebraicMultigrid.F())
                @inferred AlgebraicMultigrid._solve(ml, b, cycle)
            end
        end
    end

    # CommonSolve interface
    @inferred solve(A, b, RugeStubenAMG())
    @inferred solve(A, b, SmoothedAggregationAMG())
    @inferred solve(Ac, bc, RugeStubenAMG())
    # `log = true` returns `(x, residuals)`
    x, residuals = solve(A, b, RugeStubenAMG(); log = true)
    @test x isa Vector{Float64}
    @test residuals isa Vector{Float64}

    # `AMGSolver{T}(ml, b)` infers the hierarchy type.
    ml = ruge_stuben(A)
    @test AlgebraicMultigrid.AMGSolver{Float64}(ml, b) isa AlgebraicMultigrid.AMGSolver{Float64}
end

# A preconditioner is applied once per Krylov iteration, so a cycle should not
# allocate. The default `QRSolver` coarse solve allocates inside SuiteSparse;
# with `Pinv` the whole cycle runs in the preallocated workspace.
@testset "Allocation-free cycles" begin
    for T in (Float64, Float32)
        A = poisson(T, (40, 40))
        b = rand(T, size(A, 1))
        for ml in (ruge_stuben(A; coarse_solver = AlgebraicMultigrid.Pinv),
                   smoothed_aggregation(A; coarse_solver = AlgebraicMultigrid.Pinv))
            for cycle in (AlgebraicMultigrid.V(), AlgebraicMultigrid.W(), AlgebraicMultigrid.F())
                p = aspreconditioner(ml, cycle)
                x = zero(b)
                ldiv!(x, p, b)
                @test (@allocated ldiv!(x, p, b)) == 0
            end
        end
    end
end
