function smoothed_aggregation(A::TA, B = nothing,
                        ::Type{Val{bs}}=Val{1};
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
                        coarse_solver = Pinv, kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n = size(A, 1)
    B = isnothing(B) ? ones(T,n,1) : B
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

    levels = Vector{Level{TA, TA, Adjoint{T, TA}}}()
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
    b = zeros(size(A,1))
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

function fit_candidates(AggOp, B, tol = 1e-10)

    A = adjoint(AggOp)
    @show AggOp |> size
    n_fine, m      = size(B)       
    n_fine2, n_agg = size(A)     
    @assert n_fine2 == n_fine
    
    n_coarse = m * n_agg
    T = eltype(B)
    Qs = spzeros(T, n_fine, n_coarse)
    R  = zeros(T, n_coarse, m)

    for agg in 1:n_agg
        # fine‐node indices in this aggregate
        rows = A.rowval[A.colptr[agg] : A.colptr[agg+1]-1]

        # local near‐nullspace block (length(rows)×m)
        M = @view B[rows, :]

        # thin QR ⇒ Qj(length(rows)×m), Rj(m×m)
        Qj, Rj = qr(M)

        # offset in global Qs/R
        offset = (agg - 1) * m

        # scatter dense Qj into sparse Qs
        for local_i in 1:length(rows), local_j in 1:m
            val = Qj[local_i, local_j]
            if abs(val) >= tol
                Qs[rows[local_i], offset + local_j] = val
            end
        end
        dropzeros!(Qs)
        # stack Rj into R
        R[offset+1 : offset+m, :] .= Rj
    end

    return Qs, R
end

function qr(A, tol = 1e-10)
    T = eltype(A)
    m, n = size(A)
    Q = similar(A)               # m×n, will hold the orthonormal vectors
    R = zeros(T, n, n)           # n×n upper triangular

    for j in 1:n
        # start with the j-th column of A
        v = @view A[:, j]              # creates a copy of the column

        # subtract off components along previously computed q_i
        for i in 1:j-1
            q = @view Q[:,i] 
            r_ij = q ⋅ v
            R[i, j] = r_ij > tol ? r_ij : zero(T)
            v = v -  R[i, j] * q
        end

        # normalize to get q_j
        R[j, j] = norm(v)
        @assert R[j, j] > tol     "Matrix is rank-deficient at column $j"
        Q[:, j] = v / R[j, j]
    end

    return Q, R
end
