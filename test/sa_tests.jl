import AlgebraicMultigrid: scale_cols_by_largest_entry!,
            SymmetricStrength, poisson, StandardAggregation
function symmetric_soc(A::SparseMatrixCSC{T,V}, θ) where {T,V}
    D = abs.(diag(A))
    i,j,v = findnz(A)
    mask = i .!= j
    DD = D[i] .* D[j]
    mask = mask .& (abs.(v.^2) .>= (θ * θ * DD))

    i = i[mask]
    j = j[mask]
    v = v[mask]

    S = sparse(i,j,v, size(A)...) + spdiagm(0=>D)

    for i = 1:size(S.nzval,1)
        S.nzval[i] = abs(S.nzval[i])
    end

    scale_cols_by_largest_entry!(S)

    S
end

# Set up tests
function test_symmetric_soc()

    cases = generate_matrices()

    for matrix in cases
        for θ in (0.0, 0.1, 0.5, 1., 10.)
            ref_matrix = symmetric_soc(matrix, θ)
            calc_matrix, _ = SymmetricStrength(θ)(matrix)

            @test sum(abs2, ref_matrix - calc_matrix) < 1e-6
        end
    end
end

function generate_matrices()

    cases = []

    # Random matrices
    seed!(0)
    for T in (Float32, Float64)

        for s in [2, 3, 5]
            push!(cases, sprand(T, s, s, 1.))
        end

        for s in [2, 3, 5, 7, 10, 11, 19]
            push!(cases, poisson(T, s))
        end
    end

    cases
end

# Implementation of Algorithm 5.1 from ,,Algebraic Multigrid by Smoothed
# Aggregation for Second and Fourth Order Elliptic Problems'' by Vanek et al.
#
# Note: isolated nodes are not aggregated
function stand_agg(C, ϵ=0)
    n = size(C, 1)

    # FIXME maneuvering around the implementation begin CSC but pretending to be CSR and
    #       the fact that the implementation can only handle symmetric matrices while
    #       this test covers has non-symmetric ones...
    NϵT(C, i, ϵ) = [j for j in 1:size(C, 2) if abs(C[i, j]) > ϵ * sqrt(C[i,i]*C[j,j])]
    Nϵ(C, i, ϵ) = [j for j in 1:size(C, 2) if abs(C[j, i]) > ϵ * sqrt(C[i,i]*C[j,j])]
    R = Set([i for i in 1:n if Nϵ(C, i, 0.0) != [i] || NϵT(C, i, 0.0) != [i]])

    j = 0
    Cpts = Int[]

    aggregates = -ones(Int, n)

    # Pass 1
    for i = 1:n
        Ni = Set(Nϵ(C, i, ϵ))
        if issubset(Ni, R)
            push!(Cpts, i)
            setdiff!(R, Ni)
            for x in Ni
                aggregates[x] = j
            end
            j += 1
        end
    end

    # Pass 2
    old_R = copy(R)
    for i = 1:n
        if i ∉ R
            continue
        end

        best_strength = -Inf
        best_candidate = 0
        for j in nzrange(C, i)
            x = C.rowval[j]
            if x ∉ old_R && best_strength < C.nzval[j]
                best_strength = C.nzval[j]
                best_candidate = x
            end
        end
        if best_candidate > 0
            aggregates[i] = aggregates[best_candidate]
            setdiff!(R, i)
        end
    end

    # Pass 3
    for i = 1:n
        if i ∉ R
            continue
        end

        Ni = intersect(Set(Nϵ(C, i, ϵ)), R)
        push!(Ni, i)  # always include the seed node itself
        push!(Cpts, i)
        setdiff!(R, Ni)
        for x in Ni
            aggregates[x] = j
        end
        j += 1
    end

    mask = aggregates .> -1
    I = aggregates[mask] .+ 1
    J = collect(1:n)[mask]
    V = ones(length(I))
    return sparse(I, J, V, maximum(I; init=0), n)
end



# Corner case tests for StandardAggregation
function test_standard_aggregation_corner_cases()

    # Off-by-one in aggregate_size: a 4-node chain whose strength matrix has no
    # diagonal entries should form 2 aggregates of size 2 when min_aggregate_size=2.
    # The buggy code (aggregate_size starts at 0) rejects the first candidate because
    # it only counts the 1 neighbour and misses the seed, collapsing everything into
    # one aggregate instead.
    S_chain = sparse([1,2,2,3,3,4],[2,1,3,2,4,3], ones(Float64,6), 4, 4)
    AggOp_chain = StandardAggregation(2)(S_chain)
    @test size(AggOp_chain, 1) == 2                          # exactly 2 aggregates
    @test all(vec(sum(AggOp_chain, dims=1)) .== 1)           # every node in exactly one

    # Disconnected graph: two independent 3-node chains.
    # Both components must be fully aggregated without mixing.
    rows_d = [1,2,2,3,4,5,5,6]; cols_d = [2,1,3,2,5,4,6,5]
    S_disc = sparse(rows_d, cols_d, ones(Float64,8), 6, 6) + spdiagm(0 => ones(6))
    ref_disc  = stand_agg(S_disc)
    calc_disc = StandardAggregation()(S_disc)
    @test sum(abs2, calc_disc - ref_disc) < 1e-6

    # All nodes isolated (diagonal-only strength matrix) → nothing is aggregated.
    S_iso = spdiagm(0 => ones(Float64, 5))
    @test sum(abs2, StandardAggregation()(S_iso) - stand_agg(S_iso)) < 1e-6
    @test nnz(StandardAggregation()(S_iso)) == 0

    # Empty 0×0 strength matrix must not crash (guards minimum() on empty array).
    S_empty = spzeros(Float64, 0, 0)
    AggOp_empty = StandardAggregation()(S_empty)
    @test size(AggOp_empty) == (0, 0)

    # smoothed_aggregation on a large diagonal matrix (all nodes isolated at every level)
    # must return a valid 1-level hierarchy rather than crashing at solve time.
    A_diag = spdiagm(0 => 2.0 * ones(Float64, 20))
    ml_diag = smoothed_aggregation(A_diag)
    @test length(ml_diag) == 1
    @test size(ml_diag.final_A) == (20, 20)

end

# Standard aggregation tests
function test_standard_aggregation()

    cases = generate_matrices()

    for matrix in cases
        for θ in (0.0, 0.02, 0.1, 1.)
            # We have to symmetrize the matrix for this functions below to work as expected (since some matrices are non-symmetric)
            C = symmetric_soc(matrix + matrix', θ)
            calc_matrix = StandardAggregation()(C)
            ref_matrix = stand_agg(C)
            @test sum(abs2, ref_matrix - calc_matrix) < 1e-6
        end
    end

end

# Test fit_candidates
function test_fit_candidates()

    cases = generate_fit_candidates_cases()

    for (i, (AggOp, fine_candidates)) in enumerate(cases)

        mask_candidates!(AggOp, fine_candidates)

        Q, coarse_candidates = fit_candidates(AggOp, fine_candidates)

        @test isapprox(fine_candidates, Q * coarse_candidates)
        @test isapprox(Q * (Q' * fine_candidates), fine_candidates)
    end
end
function mask_candidates!(A,B)
    B[(diff(A.colptr) .== 0)] .= 0
end

function generate_fit_candidates_cases()
    cases = []

    for T in (Float32, Float64)

        # One candidate
        AggOp = SparseMatrixCSC(2, 5, collect(1:6),
                        [1,1,1,2,2], ones(T,5))
        B =  ones(T,5)
        push!(cases, (AggOp, B))

        AggOp = SparseMatrixCSC(2, 5, collect(1:6),
                        [2,2,1,1,1], ones(T,5))
        B = ones(T, 5)
        push!(cases, (AggOp, B))

        AggOp = SparseMatrixCSC(3, 9, collect(1:10),
                        [1,1,1,2,2,2,3,3,3], ones(T, 9))
        B = ones(T, 9)
        push!(cases, (AggOp, B))

        AggOp = SparseMatrixCSC(3, 9, collect(1:10),
                        [3,2,1,1,2,3,2,1,3], ones(T,9))
        B = T.(collect(1:9))
        push!(cases, (AggOp, B))
    end

    cases
end

# Test approximate spectral radius
function test_approximate_spectral_radius()

    cases = []
    seed!(0)

    push!(cases, [2. 0.
                  0. 1.])

    push!(cases, [-2. 0.
                   0  1])

    push!(cases, [100.   0.  0.
                    0. 101.  0.
                    0.   0. 99.])

    for i in 2:5
        push!(cases, rand(i,i))
    end

    for A in cases
        E,V = (eigen(A)...,)
        E = abs.(E)
        largest_eig = findall(E .== maximum(E))[1]
        expected_eig = E[largest_eig]

        @test isapprox(approximate_spectral_radius(A), expected_eig)
    end

    # Symmetric matrices
    for A in cases
        A = A + A'
        @static if VERSION < v"0.7-"
            E,V = eig(A)
        else
            E,V = (eigen(A)...,)
        end
        E = abs.(E)
        largest_eig = findall(E .== maximum(E))[1]
        expected_eig = E[largest_eig]

        @test isapprox(approximate_spectral_radius(A), expected_eig)
    end
end

# Test Gauss Seidel
function test_gauss_seidel()
    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(ForwardSweep())
    s(A, x, b)
    @test sum(abs2, x - zeros(N)) < 1e-8

    N = 3
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(ForwardSweep())
    s(A, x, b)
    @test sum(abs2, x - [1.0/2.0, 5.0/4.0, 5.0/8.0]) < 1e-8

    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(BackwardSweep())
    s(A, x, b)
    @test sum(abs2, x - zeros(N)) < 1e-8

    N = 3
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(BackwardSweep())
    s(A, x, b)
    @test sum(abs2, x - [1.0/8.0, 1.0/4.0, 1.0/2.0]) < 1e-8

    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = eltype(A).([10.])
    s = GaussSeidel(ForwardSweep())
    s(A, x, b)
    @test sum(abs2, x - [5.]) < 1e-8

    N = 3
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = eltype(A).([10., 20., 30.])
    s = GaussSeidel(ForwardSweep())
    s(A, x, b)
    @test sum(abs2, x - [11.0/2.0, 55.0/4, 175.0/8.0]) < 1e-8

    N = 100
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = ones(eltype(A), N)
    b = zeros(eltype(A), N)
    s1 = GaussSeidel(ForwardSweep(), 200)
    s1(A, x, b)
    resid1 = norm(A*x,2)
    x = ones(eltype(A), N)
    s2 = GaussSeidel(BackwardSweep(), 200)
    s2(A, x, b)
    resid2 = norm(A*x,2)
    @test resid1 < 0.01 && resid2 < 0.01
    @test isapprox(resid1, resid2)

end

# Test jacobi smooth prolongator with local weighting
function test_jacobi_prolongator()
    A = poisson(100)
    T = poisson(100)
    x = JacobiProlongation(4/3)(A, T, 1, 1)
    ref = include("ref_R.jl")
    @test sum(abs2, x - ref) < 1e-6
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

    @testset "Jacobi Prolongation" begin
        test_jacobi_prolongator()
    end

    @testset "Int32 support" begin
        a = sparse(Int32.(1:10), Int32.(1:10), rand(10))
        @inferred smoothed_aggregation(a)
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

    @testset "Standard Aggregation Corner Cases" begin
        test_standard_aggregation_corner_cases()
    end

    @testset "Fit Candidates" begin
        test_fit_candidates()
    end

    @testset "Approximate Spectral Radius" begin
        test_approximate_spectral_radius()
    end

    @testset "Jacobi Prolongation" begin
        test_jacobi_prolongator()
    end

    @testset "Int32 support" begin
        a = sparse(Int32.(1:10), Int32.(1:10), rand(10))
        @inferred smoothed_aggregation(a)
    end
end