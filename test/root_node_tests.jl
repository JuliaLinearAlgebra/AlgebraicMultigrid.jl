using AlgebraicMultigrid
import AlgebraicMultigrid as AMG

## Energy Prolongation Tests
@testset "Energy Prolongation" begin
    @testset "Energy is reduced" begin
        A = float.(poisson(50))
        n = size(A, 1)
        B = ones(n)

        # Build strength, splitting, aggregation, tentative P
        S, _ = AMG.SymmetricStrength()(A)
        splitting = AMG.RS()(copy(S))
        AggOp, c_map, nc = AMG.root_node_aggregation(S, splitting)
        T, Bc = fit_candidates(AggOp, B)

        # Compute energy before smoothing: trace(T' * A * T)
        energy_before = tr(Matrix(T' * A * T))

        # Apply energy prolongation
        P = AMG.energy_prolongation_smoother(A, T, S, splitting)
        energy_after = tr(Matrix(P' * A * P))

        @test energy_after < energy_before
    end

    @testset "Null space is approximately preserved" begin
        A = float.(poisson(50))
        n = size(A, 1)
        B = ones(n)

        S, _ = AMG.SymmetricStrength()(A)
        splitting = AMG.RS()(copy(S))
        AggOp, c_map, nc = AMG.root_node_aggregation(S, splitting)
        T, Bc = fit_candidates(AggOp, B)

        P = AMG.energy_prolongation_smoother(A, T, S, splitting)

        # C-point rows exactly preserve the null space
        # F-point rows are smoothed, so overall null space is approximately preserved
        @test norm(P * Bc - B) / norm(B) < 0.1
    end

    @testset "C-point rows are unchanged" begin
        A = float.(poisson(50))
        n = size(A, 1)
        B = ones(n)

        S, _ = AMG.SymmetricStrength()(A)
        splitting = AMG.RS()(copy(S))
        AggOp, c_map, nc = AMG.root_node_aggregation(S, splitting)
        T, Bc = fit_candidates(AggOp, B)

        P = AMG.energy_prolongation_smoother(A, T, S, splitting)

        # C-point rows should be identical in T and P
        for i = 1:n
            if splitting[i] == AMG.C_NODE
                for col = 1:nc
                    @test AMG._get_entry(P, i, col) ≈ AMG._get_entry(T, i, col) atol=1e-14
                end
            end
        end
    end
end

## Root-Node AMG Solver Tests
@testset "Root-Node AMG" begin
    @testset "Scalar-valued problem" begin
        A = float.(poisson(100))
        b = A * ones(100)
        x = solve(A, b, RootNodeAMG(), reltol=1e-10)
        @test norm(A * x - b) / norm(b) < 1e-8
    end
end
