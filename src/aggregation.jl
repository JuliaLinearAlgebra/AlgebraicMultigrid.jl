struct JacobiProlongation{T}
    ω::T
end

struct DiagonalWeighting
end
struct LocalWeighting
end

function (j::JacobiProlongation)(A, T, S, B, degree = 1, weighting = LocalWeighting())
    D_inv_S = weight(weighting, A, j.ω)
    P = T
    for i = 1:degree
        P = P - (D_inv_S * P)
    end
    P
end

function weight(::DiagonalWeighting, S, ω)
    D_inv = 1 ./ diag(S)
    D_inv_S = scale_rows(S, D_inv)
    (eltype(S)(ω) / approximate_spectral_radius(D_inv_S)) * D_inv_S
    # (ω) * D_inv_S
end

function weight(::LocalWeighting, S, ω)
    #=D = abs.(S) * ones(eltype(S), size(S, 1))
    D_inv = 1 ./ D[find(D)]
    D_inv_S = scale_rows(S, D_inv)
    eltype(S)(ω) * D_inv_S=#
    D = zeros(eltype(S), size(S,1))
    for i = 1:size(S, 1)
        for j in nzrange(S, i)
            row = S.rowval[j]
            val = S.nzval[j]
            D[row] += abs(val)
        end
    end
    for i = 1:size(D, 1)
        if D[i] != 0
            D[i] = 1/D[i]
        end
    end
    D_inv_S = scale_rows(S, D)
    # eltype(S)(ω) * D_inv_S
    rmul!(D_inv_S, eltype(S)(ω))
end

function scale_rows!(ret, S, v)
    n = size(S, 1)
    for i = 1:n
        for j in nzrange(S, i)
            row = S.rowval[j]
            ret.nzval[j] *= v[row]
        end
    end
    ret
end
scale_rows(S, v) = scale_rows!(deepcopy(S), S,  v)

function smoothed_aggregation(_A::Union{Symmetric, Hermitian}, args...; kwargs...)
    A, symmetry = get_symmetry_and_data(_A)
    return smoothed_aggregation(A, args...; symmetry, kwargs...)
end

function smoothed_aggregation(A::TA,
                        ::Type{Val{bs}}=Val{1};
                        B = nothing,
                        symmetry = HermitianSymmetry(),
                        strength = SymmetricStrength(),
                        aggregate = StandardAggregation(),
                        smooth = JacobiProlongation(4.0/3.0),
                        presmoother = GaussSeidel(),
                        postsmoother = GaussSeidel(),
                        improve_candidates = GaussSeidel(iter=4),
                        max_levels = 10,
                        max_coarse = 10,
                        diagonal_dominance = false,
                        keep = false,
                        verbose = false,
                        coarse_solver = Pinv, kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    @timeit_debug "prologue" begin

    n = size(A, 1)
    B = isnothing(B) ? ones(T,n) : copy(B)
    @assert size(A, 1) == size(B, 1)

    levels = Vector{Level{TA, TA, Adjoint{T, TA}}}()
    bsr_flag = false
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    end

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        @timeit_debug "extend_hierarchy!" A, B, bsr_flag = extend_hierarchy_sa!(levels, strength, aggregate, smooth,
                                improve_candidates, diagonal_dominance,
                                keep, A, B, presmoother, postsmoother, symmetry, bsr_flag, verbose)
        size(A, 1) == 0 && break
        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        residual!(w, size(A, 1))
    end

    @timeit_debug "coarse solver setup" cs = coarse_solver(A)
    ml = MultiLevel(levels, A, cs,  presmoother, postsmoother, w)

    if verbose
        @info ml
    end

    return ml
end

function extend_hierarchy_sa!(levels, strength, aggregate, smooth,
                            improve_candidates, diagonal_dominance,
                            keep, A, B, presmoother, postsmoother,
                            symmetry, bsr_flag, verbose)

    # Calculate strength of connection matrix
    @timeit_debug "strength" if symmetry isa HermitianSymmetry
        S, _T = strength(A, bsr_flag)
    else
        S, _T = strength(adjoint(A), bsr_flag)
    end

    # Aggregation operator
    @timeit_debug "aggregation" AggOp = aggregate(S)

    # Improve candidates
    b = zeros(size(A,1),size(B,2))
    @timeit_debug "improve candidates" improve_candidates(A, B, b)
    @timeit_debug "fit candidates" T, B = fit_candidates(AggOp, B)

    @timeit_debug "restriction setup" begin
        P = smooth(A, T, S, B)
        R = construct_R(symmetry, P)
    end

    @timeit_debug "RAP" RAP = R * A * P

    @timeit_debug "smoother setup" begin
        pre = setup_smoother(presmoother, A, symmetry)
        post = setup_smoother(postsmoother, A, symmetry)
        push!(levels, Level(A, P, R, pre, post))
    end

    bsr_flag = true

    # RAP is the coarse matrix and B is the coarse null space
    RAP, B, bsr_flag
end
construct_R(::HermitianSymmetry, P) = P'
construct_R(::NoSymmetry, P) = P'

function fit_candidates(AggOp, B::AbstractVector; tol=1e-10)
    A = adjoint(AggOp)
    n_fine, n_coarse = size(A)
    n_col = n_coarse

    R = zeros(eltype(B), n_coarse)
    Qx = zeros(eltype(B), nnz(A))
    # copy!(Qx, B)
    for i = 1:size(Qx, 1)
        Qx[i] = B[i]
    end
    # copy!(A.nzval, B)
    for i = 1:n_col
        for j in nzrange(A,i)
            row = A.rowval[j]
            A.nzval[j] = B[row]
        end
    end
    k = 1
    for i = 1:n_col
        norm_i = norm_col(A, Qx, i)
        threshold_i = tol * norm_i
        if norm_i > threshold_i
            scale = 1 / norm_i
            R[i] = norm_i
        else
            scale = 0
            R[i] = 0
        end
        for j in nzrange(A, i)
            row = A.rowval[j]
            A.nzval[j] *= scale
        end
    end

    # SparseMatrixCSC(size(A)..., A.colptr, A.rowval, Qx), R
    A, R
end

function fit_candidates(AggOp, B::AbstractMatrix; tol=1e-10)
    A = adjoint(AggOp)
    n_fine, m = size(B)
    n_fine2, n_agg = size(A)
    @assert n_fine2 == n_fine
    n_coarse = m * n_agg
    T = eltype(B)
    Qs = spzeros(T, n_fine, n_coarse) # TODO use CSR here as the algorithm becomes easier
    R = zeros(T, n_coarse, m)

    for agg in 1:n_agg
        rows = A.rowval[A.colptr[agg]:A.colptr[agg+1]-1]
        M = @view B[rows, :]     # size(rows) × m

        # TODO the code below can be optimized
        F = qr(M)
        r = min(length(rows), m)
        Qfull = Matrix(F.Q)
        Qj = Qfull[:, 1:r]
        Rj = F.R

        offset = (agg - 1) * m

        for local_i in 1:length(rows), local_j in 1:r
            val = Qj[local_i, local_j]
            if abs(val) >= tol
                Qs[rows[local_i], offset+local_j] = val
            end
        end

        R[offset+1:offset+r, :] .= Rj[1:r, :]
    end

    dropzeros!(Qs)
    return Qs, R
end

function norm_col(A, Qx, i)
    s = zero(eltype(A))
    for j in nzrange(A, i)
        if A.rowval[j] > length(Qx)
            val = 1
        else
            val = Qx[A.rowval[j]]
        end
        # val = A.nzval[A.rowval[j]]
        s += val*val
    end
    sqrt(s)
end
