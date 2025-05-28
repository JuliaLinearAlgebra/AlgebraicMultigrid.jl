function smoothed_aggregation(A::TA, _B=nothing,
    ::Type{Val{bs}}=Val{1};
    symmetry=HermitianSymmetry(),
    strength=SymmetricStrength(),
    aggregate=StandardAggregation(),
    smooth=JacobiProlongation(4.0 / 3.0),
    presmoother=GaussSeidel(),
    postsmoother=GaussSeidel(),
    improve_candidates=GaussSeidel(iter=4),
    max_levels=10,
    max_coarse=10,
    diagonal_dominance=false,
    keep=false,
    coarse_solver=Pinv, kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n = size(A, 1)
    B = isnothing(_B) ? ones(T, n, 1) : copy(_B)
    @assert size(A, 1) == size(B, 1)

    #=max_levels, max_coarse, strength =
        levelize_strength_or_aggregation(max_levels, max_coarse, strength)
    max_levels, max_coarse, aggregate =
        levelize_strength_or_aggregation(max_levels, max_coarse, aggregate)

    improve_candidates =
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)=#
    # str = [stength for _ in 1:max_levels - 1]
    # agg = [aggregate for _ in 1:max_levels - 1]
    # sm = [smooth for _ in 1:max_levels]

    levels = Vector{Level{TA,TA,Adjoint{T,TA}}}()
    bsr_flag = false
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A, B, bsr_flag = extend_hierarchy!(levels, strength, aggregate, smooth,
            improve_candidates, diagonal_dominance,
            keep, A, B, symmetry, bsr_flag)
        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        #=if size(A, 1) <= max_coarse
            break
        end=#
        residual!(w, size(A, 1))
    end
    #=A, B = extend_hierarchy!(levels, strength, aggregate, smooth,
                            improve_candidates, diagonal_dominance,
                            keep, A, B, symmetry)=#
    MultiLevel(levels, A, coarse_solver(A), presmoother, postsmoother, w)
end

struct HermitianSymmetry
end

function extend_hierarchy!(levels, strength, aggregate, smooth,
    improve_candidates, diagonal_dominance, keep,
    A, B,
    symmetry, bsr_flag)

    # Calculate strength of connection matrix
    if symmetry isa HermitianSymmetry
        S, _T = strength(A, bsr_flag)
    else
        S, _T = strength(adjoint(A), bsr_flag)
    end

    # Aggregation operator
    AggOp = aggregate(S)
    # b = zeros(eltype(A), size(A, 1))

    # Improve candidates
    b = zeros(size(A, 1))
    improve_candidates(A, B, b)
    T, B = fit_candidates(AggOp, B)

    P = smooth(A, T, S, B)
    R = construct_R(symmetry, P)
    push!(levels, Level(A, P, R))

    A = R * A * P

    dropzeros!(A)

    bsr_flag = true

    A, B, bsr_flag
end
construct_R(::HermitianSymmetry, P) = P'

function fit_candidates(AggOp, B; tol=1e-10)
    A = adjoint(AggOp)
    n_fine, m = size(B)
    n_fine2, n_agg = size(A)
    @assert n_fine2 == n_fine
    n_coarse = m * n_agg
    T = eltype(B)
    Qs = spzeros(T, n_fine, n_coarse)
    R = zeros(T, n_coarse, m)

    for agg in 1:n_agg
        rows = A.rowval[A.colptr[agg]:A.colptr[agg+1]-1]
        M = @view B[rows, :]     # size(rows) × m


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
        dropzeros!(Qs)

        R[offset+1:offset+r, :] .= Rj[1:r, :]
    end

    return Qs, R
end
