function smoothed_aggregation(A::SparseMatrixCSC{T,V},
                        symmetry = HermitianSymmetry(),
                        strength = SymmetricStrength(),
                        aggregate = StandardAggregation(),
                        smooth = JacobiProlongation(4.0/3.0),
                        presmoother = GaussSeidel(),
                        postsmoother = GaussSeidel(),
                        improve_candidates = GaussSeidel(4),
                        max_levels = 10,
                        max_coarse = 10,
                        diagonal_dominance = false,
                        keep = false,
                        coarse_solver = Pinv()) where {T,V}


    n = size(A, 1)
    # B = kron(ones(n, 1), eye(1))
    B = ones(T,n)

    #=max_levels, max_coarse, strength =
        levelize_strength_or_aggregation(max_levels, max_coarse, strength)
    max_levels, max_coarse, aggregate =
        levelize_strength_or_aggregation(max_levels, max_coarse, aggregate)

    improve_candidates =
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)=#
    # str = [stength for _ in 1:max_levels - 1]
    # agg = [aggregate for _ in 1:max_levels - 1]
    # sm = [smooth for _ in 1:max_levels]

    levels = Vector{Level{T,V}}()
    bsr_flag = false

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A, B, bsr_flag = extend_hierarchy!(levels, strength, aggregate, smooth,
                                improve_candidates, diagonal_dominance,
                                keep, A, B, symmetry, bsr_flag)
        #=if size(A, 1) <= max_coarse
            break
        end=#
    end
    #=A, B = extend_hierarchy!(levels, strength, aggregate, smooth,
                            improve_candidates, diagonal_dominance,
                            keep, A, B, symmetry)=#
    MultiLevel(levels, A, presmoother, postsmoother)
end

struct HermitianSymmetry
end

function extend_hierarchy!(levels, strength, aggregate, smooth,
                            improve_candidates, diagonal_dominance, keep,
                            A, B,
                            symmetry, bsr_flag)

    # Calculate strength of connection matrix
    S = strength_of_connection(strength, A, bsr_flag)

    # Aggregation operator
    AggOp = aggregation(aggregate, S)
    # b = zeros(eltype(A), size(A, 1))

    # Improve candidates
    b = zeros(size(A,1))
    relax!(improve_candidates, A, B, b)
    T, B = fit_candidates(AggOp, B)

    P = smooth_prolongator(smooth, A, T, S, B)
    R = construct_R(symmetry, P)
    push!(levels, Level(A, P, R))

    A = R * A * P

    dropzeros!(A)

    bsr_flag = true

    A, B, bsr_flag
end
construct_R(::HermitianSymmetry, P) = P'

function fit_candidates(AggOp, B, tol = 1e-10)

    A = AggOp.'
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
            # Qx[row] *= scale
            #@show k
            # Qx[k] *= scale
            # k += 1
            A.nzval[j] *= scale
        end
    end

    # SparseMatrixCSC(size(A)..., A.colptr, A.rowval, Qx), R
    A, R
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
