function smoothed_aggregation(A,
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
                        coarse_solver = Pinv())


    n = size(A, 1)
    # B = kron(ones(n, 1), eye(1))
    B = ones(n)

    #=max_levels, max_coarse, strength =
        levelize_strength_or_aggregation(max_levels, max_coarse, strength)
    max_levels, max_coarse, aggregate =
        levelize_strength_or_aggregation(max_levels, max_coarse, aggregate)

    improve_candidates =
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)=#
    # str = [stength for _ in 1:max_levels - 1]
    # agg = [aggregate for _ in 1:max_levels - 1]
    # sm = [smooth for _ in 1:max_levels]

    levels = Vector{Level{eltype(A), Int64}}()

    while length(levels) < max_levels && size(A, 1) > max_coarse
        A, B = extend_hierarchy!(levels, strength, aggregate, smooth,
                                improve_candidates, diagonal_dominance,
                                keep, A, B, symmetry)
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
                            symmetry)

    # Calculate strength of connection matrix
    S = strength_of_connection(strength, A)

    # Aggregation operator
    AggOp = aggregation(aggregate, S)
    AggOp = float.(AggOp)

    b = zeros(eltype(A), size(A, 1))

    # Improve candidates
    relax!(improve_candidates, A, B, b)

    T, B = fit_candidates(AggOp, B)

    P = smooth_prolongator(smooth, A, T, S, B)
    R = construct_R(symmetry, P)
    push!(levels, Level(A, P, R))

    A = R * A * P

    A, B
end
construct_R(::HermitianSymmetry, P) = P'

function fit_candidates(AggOp, B, tol = 1e-10)

    N_coarse, N_fine = size(AggOp)

    K1 = Int(size(B, 1) / N_fine)
    K2 = size(B, 2)

    A = AggOp.'

    n_coarse = size(A, 1)
    n_fine = size(A, 2)

    # R = zeros(eltype(B), N_coarse, K2, K2)
    R = zeros(eltype(B), N_coarse)
    # Qx = zeros(eltype(B), nnz(AggOp), K1, K2)
    Qx = zeros(eltype(B), nnz(A), K1)


    R = vec(R)
    Qx = vec(Qx)

    # n_row = N_fine
    # n_col = N_coarse

    # BS = K1 * K2

    #=for i = 1:n_col
        Ax_start = 1 + BS * A.colptr[i]

        for j in nzrange(A, i)
            B_start = 1 + BS * A.rowval[j]
            B_end = B_start + BS
            @show B_start
            @show B_end
            for ind in B_start:B_end
                A.nzval[ind + Ax_start] = B[ind]
            end
            Ax_start += BS
        end
    end=#
    # A.nzval .= B

    for i = 1:n_fine
        norm_i = norm(A[:,i]) # Replace by more efficient norm
        threshold_i = tol * norm_i
        if norm_i > threshold_i
            scale = 1 / norm_i
            R[i] = norm_i
        end
        for j in nzrange(A, i)
            A.nzval[j] *= scale
        end
        #=col_start = A.colptr[i]
        col_end = A.colptr[i+1]

        Ax_start = 1 + BS * col_start
        Ax_end = 1 + BS * col_end
        R_start = 1 + i * K2 * K2

        for bj = 1:K2
            norm_j = zero(eltype(A))
            Ax_col = Ax_start + bj
            while Ax_col < Ax_end
                norm_j += norm(A.nzval[Ax_col])
                Ax_col += K2
            end
            norm_j = sqrt(norm_j)

            threshold_j = tol * norm_j

            for bi = 1:bj
                dot_prod = zero(eltype(A))

                Ax_bi = Ax_start + bj
                Ax_bj = Ax_start + bj
                while Ax_bi < Ax_end
                    dot_prod += dot(A.nzval[Ax_bj], A.nzval[Ax_bi])
                    Ax_bi    += K2
                    Ax_bj    += K2
                end

                Ax_bi = Ax_start + bi;
                Ax_bj = Ax_start + bj;
                while Ax_bi < Ax_end
                    A.nzval[Ax_bj] -= dot_prod * A.nzval[Ax_bi]
                    Ax_bi  += K2
                    Ax_bj  += K2
                end

                R[R_start + K2 * bi + bj] = dot_prod
            end

            norm_j = zero(eltype(A))
            Ax_bj = Ax_start + bj
            while Ax_bj < Ax_end
                norm_j += norm(A.nzval[Ax_bj])
                Ax_bj  += K2
                norm_j = sqrt(norm_j)
            end

            if norm_j > threshold_j
                scale = 1 / norm_j
                R[R_start + K2 * bj + bj] = norm_j
            else
                scale = zero(eltype(A))
                R[R_start + K2 * bj + bj] = 0
            end

            Ax_bj = Ax_start + bj
            while Ax_bj < Ax_end
                A.nzval[Ax_bj] *= scale
                Ax_bj  += K2
            end
        end=#
    end

    #Q = SparseMatrixCSC(N_coarse, N_fine,
                        #.colptr, A.rowval, Qx)

    #R = reshape(R, N_coarse, K2)
    A, R
end
