import AMG: scale_cols_by_largest_entry!, strength_of_connection, 
            SymmetricStrength, poisson
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

    scale_cols_by_largest_entry!(S)

    for i = 1:size(S.nzval,1)
        S.nzval[i] = abs(S.nzval[i])
    end

    S
end

# Set up tests
function test_symmetric_soc()

    cases = generate_matrices()

    for matrix in cases
        for θ in (0.0, 0.1, 0.5, 1., 10.)
            ref_matrix = symmetric_soc(matrix, θ)
            calc_matrix = strength_of_connection(SymmetricStrength(θ), matrix)

            @test sum(abs2, ref_matrix - calc_matrix) < 1e-6
        end
    end
end

function generate_matrices()
    
    cases = []

    # Random matrices
    srand(0)
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

function stand_agg(C)
    n = size(C, 1)

    R = Set(1:n)
    j = 0
    Cpts = Int[]

    aggregates = -ones(Int, n)

    # Pass 1
    for i = 1:n
        Ni = union!(Set(C.rowval[nzrange(C, i)]), Set(i))
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
        if ! (i in R)
            continue
        end

        for x in C.rowval[nzrange(C, i)]
            if !(x in old_R)
                aggregates[i] = aggregates[x]
                setdiff!(R, i)
                break
            end
        end
    end

    # Pass 3
    for i = 1:n
        if !(i in R)
            continue
        end
        Ni = union(Set(C.rowval[nzrange(C,i)]), Set(i))
        push!(Cpts, i)

        for x in Ni
            if x in R
                aggregates[x] = j
            end
            j += 1
        end
    end

    @assert length(R) == 0

    Pj = aggregates .+ 1
    Pp = collect(1:n+1)
    Px = ones(eltype(C), n)

    SparseMatrixCSC(maximum(aggregates .+ 1), n, Pp, Pj, Px)
end

# Standard aggregation tests
function test_standard_aggregation()

    cases = generate_matrices()

    for matrix in cases
        for θ in (0.0, 0.1, 0.5, 1., 10.)
            C = symmetric_soc(matrix, θ)
            calc_matrix = aggregation(StandardAggregation(), matrix)
            ref_matrix = stand_agg(matrix)
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
    srand(0)

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
import AMG: gs!, relax!
function test_gauss_seidel()
    
    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(ForwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - zeros(N)) < 1e-8

    N = 3 
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(ForwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - [1.0/2.0, 5.0/4.0, 5.0/8.0]) < 1e-8

    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(BackwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - zeros(N)) < 1e-8

    N = 3 
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = zeros(N)
    s = GaussSeidel(BackwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - [1.0/8.0, 1.0/4.0, 1.0/2.0]) < 1e-8

    N = 1
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = eltype(A).([10.])
    s = GaussSeidel(ForwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - [5.]) < 1e-8

    N = 3 
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = eltype(A).(collect(0:N-1))
    b = eltype(A).([10., 20., 30.])
    s = GaussSeidel(ForwardSweep())
    relax!(s, A, x, b)
    @test sum(abs2, x - [11.0/2.0, 55.0/4, 175.0/8.0]) < 1e-8

    N = 100
    A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
    x = ones(eltype(A), N)
    b = zeros(eltype(A), N)
    s1 = GaussSeidel(ForwardSweep(), 200)
    relax!(s1, A, x, b)
    resid1 = norm(A*x,2)
    x = ones(eltype(A), N)
    s2 = GaussSeidel(BackwardSweep(), 200)
    relax!(s2, A, x, b)
    resid2 = norm(A*x,2)
    @test resid1 < 0.01 && resid2 < 0.01
    @test isapprox(resid1, resid2)

end

# Test jacobi smooth prolongator with local weighting
function test_jacobi_prolongator()
    A = poisson(100)
    T = poisson(100)
    x = smooth_prolongator(JacobiProlongation(4/3), A, T, 1, 1)
    ref = load("ref_R.jld")["G"]
    @test sum(abs2, x - ref) < 1e-6
end

# Issue #24
function nodes_not_agg()
    A = load("onetoall.jld")["G"]
    ml = smoothed_aggregation(A)
    @test size(ml.levels[2].A) == (11,11)
    @test size(ml.final_A) == (2,2)
end

# Issue 26
import AMG: relax!
function test_symmetric_sweep()
    A = poisson(10)
    s = GaussSeidel(SymmetricSweep(), 4)
    x = ones(size(A,1))
    b = zeros(size(A,1))
    relax!(s, A, x, b)
    @test sum(abs2, x - [0.176765; 0.353529; 0.497517; 0.598914; 
                            0.653311; 0.659104; 0.615597; 0.52275; 
                            0.382787; 0.203251]) < 1e-6
        
end

