"""
    EnergyProlongation

Energy-minimizing prolongation smoother for Root-Node AMG.
Uses Jacobi-like smoothing (similar to SA's JacobiProlongation) applied
only to F-point rows, while C-point (root node) rows are held fixed.
The sparsity pattern is implicitly expanded through the smoothing iterations.
"""
struct EnergyProlongation
    maxiter::Int
    omega::Float64
end
EnergyProlongation() = EnergyProlongation(4, 4.0/3.0)

function (ep::EnergyProlongation)(A, T, S, B, splitting, c_map)
    energy_prolongation_smoother(A, T, S, splitting;
                                 maxiter=ep.maxiter, omega=ep.omega)
end

"""
    energy_prolongation_smoother(A, T, S, splitting; maxiter=4, omega=4/3)

Smooth the tentative prolongator `T` by minimizing energy.
Applies Jacobi-like smoothing iterations (P = P - ω D⁻¹_S P) to F-point rows
only, while preserving C-point rows from the tentative prolongator.
The sparsity pattern naturally expands through the smoothing.
"""
function energy_prolongation_smoother(A::SparseMatrixCSC{Tv,Ti},
                                      T::SparseMatrixCSC{Tv,Ti},
                                      S::SparseMatrixCSC,
                                      splitting;
                                      maxiter::Int=4, omega::Real=4.0/3.0) where {Tv,Ti}
    n = size(A, 1)
    D_inv_S = weight(LocalWeighting(), A, omega)

    P = T
    for iter = 1:maxiter
        # Compute the Jacobi update: delta = D_inv_S * P
        delta = D_inv_S * P

        # Apply update P = P - delta, but only for F-point rows
        P_new = P - delta

        # Restore C-point rows to their original values from T
        _restore_c_rows!(P_new, T, splitting)

        P = P_new
    end

    return P
end

"""
Restore C-point rows of `P` to their values from `T`.
"""
function _restore_c_rows!(P::SparseMatrixCSC{Tv,Ti}, T::SparseMatrixCSC{Tv,Ti},
                           splitting) where {Tv,Ti}
    # Zero out all C-point rows in P
    for col = 1:size(P, 2)
        for idx in nzrange(P, col)
            row = P.rowval[idx]
            if splitting[row] == C_NODE
                P.nzval[idx] = zero(Tv)
            end
        end
    end

    # Copy C-point rows from T into P
    for col = 1:size(T, 2)
        for idx in nzrange(T, col)
            row = T.rowval[idx]
            if splitting[row] == C_NODE
                _add_entry!(P, row, col, T.nzval[idx])
            end
        end
    end
end

function _get_entry(P::SparseMatrixCSC, row, col)
    for idx in nzrange(P, col)
        if P.rowval[idx] == row
            return P.nzval[idx]
        end
    end
    return zero(eltype(P))
end

function _add_entry!(P::SparseMatrixCSC, row, col, val)
    for idx in nzrange(P, col)
        if P.rowval[idx] == row
            P.nzval[idx] += val
            return
        end
    end
end

"""
Drop zero columns from prolongator P and corresponding rows from B_coarse.
Returns (P_new, B_new).
"""
function _drop_zero_columns(P::SparseMatrixCSC{Tv,Ti}, B_coarse) where {Tv,Ti}
    nc = size(P, 2)
    keep = falses(nc)
    for col = 1:nc
        for idx in nzrange(P, col)
            if P.nzval[idx] != zero(Tv)
                keep[col] = true
                break
            end
        end
    end
    all(keep) && return P, B_coarse

    keep_idx = findall(keep)
    P_new = P[:, keep_idx]
    B_new = isa(B_coarse, AbstractMatrix) ? B_coarse[keep_idx, :] : B_coarse[keep_idx]
    return P_new, B_new
end

"""
    root_node_aggregation(S, splitting)

Form aggregates around C-points (root nodes).
Each C-point is the root of its aggregate. F-points join the aggregate
of their strongest C-point neighbor.

Returns `(AggOp, c_map, n_coarse)`:
- `AggOp`: n_coarse × n_fine aggregation operator
- `c_map`: maps fine node index to coarse node index (0 for F-points)
- `n_coarse`: number of coarse nodes (= number of C-points)
"""
function root_node_aggregation(S::SparseMatrixCSC{Tv,Ti}, splitting) where {Tv,Ti}
    n = size(S, 1)

    # Number the C-points
    n_coarse = 0
    c_map = zeros(Ti, n)
    for i = 1:n
        if splitting[i] == C_NODE
            n_coarse += 1
            c_map[i] = n_coarse
        end
    end

    # Assign each node to an aggregate
    aggregate = zeros(Ti, n)

    # C-points get their own aggregate
    for i = 1:n
        if splitting[i] == C_NODE
            aggregate[i] = c_map[i]
        end
    end

    # F-points join the aggregate of their strongest C-point neighbor
    for i = 1:n
        splitting[i] == C_NODE && continue
        best_val = zero(Tv)
        best_agg = zero(Ti)
        for j in nzrange(S, i)
            row = S.rowval[j]
            val = S.nzval[j]
            if splitting[row] == C_NODE && val > best_val
                best_val = val
                best_agg = c_map[row]
            end
        end
        aggregate[i] = best_agg
    end

    # Build sparse aggregation operator (n_coarse × n_fine)
    I = Ti[]
    J = Ti[]
    V = Tv[]
    for i = 1:n
        if aggregate[i] > 0
            push!(I, aggregate[i])
            push!(J, Ti(i))
            push!(V, one(Tv))
        end
    end

    AggOp = sparse(I, J, V, n_coarse, n)
    return AggOp, c_map, n_coarse
end

function root_node_amg(A::TA,
                    ::Type{Val{bs}}=Val{1};
                    B = nothing,
                    symmetry = HermitianSymmetry(),
                    strength = SymmetricStrength(),
                    CF = RS(),
                    smooth = EnergyProlongation(),
                    presmoother = GaussSeidel(),
                    postsmoother = GaussSeidel(),
                    improve_candidates = GaussSeidel(iter=4),
                    max_levels = 10,
                    max_coarse = 10,
                    keep = false,
                    verbose = false,
                    coarse_solver = Pinv, kwargs...) where {T,V,bs,TA<:SparseMatrixCSC{T,V}}

    n = size(A, 1)
    B = isnothing(B) ? ones(T, n) : copy(B)
    @assert size(A, 1) == size(B, 1)

    levels = Vector{Level{TA, TA, Adjoint{T, TA}}}()
    bsr_flag = false
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A, B, bsr_flag = extend_hierarchy_rn!(levels, strength, CF, smooth,
                                improve_candidates, keep, A, B,
                                symmetry, bsr_flag, verbose)
        size(A, 1) == 0 && break
        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        residual!(w, size(A, 1))
    end

    cs = coarse_solver(A)
    ml = MultiLevel(levels, A, cs, presmoother, postsmoother, w)

    if verbose
        @info ml
    end

    return ml
end

function extend_hierarchy_rn!(levels, strength, CF, smooth,
                              improve_candidates, keep, A, B,
                              symmetry, bsr_flag, verbose = false)

    # Strength of connection
    if symmetry isa HermitianSymmetry
        S, _T = strength(A, bsr_flag)
    else
        S, _T = strength(adjoint(A), bsr_flag)
    end

    # C/F splitting
    S_copy = copy(S)
    remove_diag!(S_copy)
    splitting = RS_CF_splitting(S_copy, adjoint(S_copy))

    # Root-node aggregation
    AggOp, c_map, n_coarse = root_node_aggregation(S, splitting)

    # Improve candidates
    b = zeros(size(A, 1), size(B, 2))
    improve_candidates(A, B, b)
    T_tent, B_coarse = fit_candidates(AggOp, B)

    # Energy-minimizing prolongation
    P = smooth(A, T_tent, S, B_coarse, splitting, c_map)
    P, B_coarse = _drop_zero_columns(P, B_coarse)
    R = construct_R(symmetry, P)

    RAP = R * A * P

    push!(levels, Level(A, P, R))

    bsr_flag = true

    RAP, B_coarse, bsr_flag
end
