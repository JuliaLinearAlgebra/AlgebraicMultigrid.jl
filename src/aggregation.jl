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
                                keep, A, B, symmetry, bsr_flag, verbose)
        size(A, 1) == 0 && break
        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        residual!(w, size(A, 1))
    end

    @timeit_debug "coarse solver setup" cs = coarse_solver(A)
    @timeit_debug "ml setup" ml = MultiLevel(levels, A, cs, presmoother, postsmoother, w)

    if verbose
        @info ml
    end

    return ml
end

struct HermitianSymmetry
end

function extend_hierarchy_sa!(levels, strength, aggregate, smooth,
                            improve_candidates, diagonal_dominance,
                            keep, A, B,
                            symmetry, bsr_flag, verbose = false)

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

    push!(levels, Level(A, P, R))

    bsr_flag = true

    RAP, B, bsr_flag
end
construct_R(::HermitianSymmetry, P) = P'

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
