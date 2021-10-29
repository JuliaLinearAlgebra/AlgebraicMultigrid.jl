    import AlgebraicMultigrid: ruge_stuben, smoothed_aggregation,
    poisson, aspreconditioner

import IterativeSolvers: cg

function test_cycles()

    A = poisson((50,50))
    b = A * ones(size(A,1))

    reltol = 1e-8

    for method in [ruge_stuben, smoothed_aggregation]
        ml = method(A)

        for cycle in [AlgebraicMultigrid.V(),AlgebraicMultigrid.W(),AlgebraicMultigrid.F()]
            x,convhist = solve(ml, b, cycle; reltol = reltol, log = true)

            @debug "number of iterations for $cycle using $method: $(length(convhist))"
            @test norm(b - A*x) < reltol * norm(b)
        end

        for cycle in [AlgebraicMultigrid.V(),AlgebraicMultigrid.W(),AlgebraicMultigrid.F()]
            p = aspreconditioner(ml,cycle)

            x,convhist = cg(A, b, Pl = p; reltol = reltol, log = true)
            @debug "CG: number of iterations for $cycle using $method: $(convhist.iters)"
            @test norm(b - A*x) <= reltol * norm(b)
        end
    end
end
