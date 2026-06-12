"""
    StandardAggregation()

Implementation of Algorithm 5.1 from ,,Algebraic Multigrid by Smoothed
Aggregation for Second and Fourth Order Elliptic Problems'' by Vanek et al. (1996).

Note that isolated nodes are not aggregated.
"""
struct StandardAggregation
end

function (::StandardAggregation)(S::SparseMatrixCSC{T,R}) where {T,R}
    n = size(S, 1)
    x = zeros(R, n)

    next_aggregate = 1

    # Pass 1: Tentative aggregation
    for i = 1:n
        if x[i] != 0
            continue
        end

        has_agg_neighbors = false
        has_neighbors = false

        for j in nzrange(S, i)
            row = S.rowval[j]
            if row != i
                has_neighbors = true
                if x[row] != 0
                    has_agg_neighbors = true
                    break
                end
            end
        end

        if !has_neighbors # Mark isolated node
            x[i] = -n
        elseif !has_agg_neighbors
            x[i] = next_aggregate
            for j in nzrange(S, i)
                row = S.rowval[j]
                if row != i
                    x[row] = next_aggregate
                end
            end

            next_aggregate += 1
        end
    end

    # Pass 2: Enlarge tentative aggregates
    for i = 1:n
        # Skip marked node
        if x[i] != 0
            continue
        end

        s_best = zero(eltype(S))
        x_best = 0
        for j in nzrange(S, i)
            row = S.rowval[j]
            x_row = x[row]
            s_candidate = S.nzval[j]
            if x_row > 0 && s_candidate > s_best # Assigned and stronger than previous best
                s_best = s_candidate
                x_best = x_row
            end
        end
        if x_best > 0
            x[i] = -x_best
        end
    end

    # Record which nodes are still unaggregated after Pass 1 and Pass 2 *before*
    # the 0-based shift below.  After the shift, accepted aggregate-0 nodes also
    # have x[i] == 0, so we cannot use x[i] == 0 as an "unaggregated" sentinel
    # inside the subsequent Pass 3 loop.
    unagg = x .== 0

    # Shift all Pass 1 / Pass 2 assignments from 1-indexed to 0-indexed.
    next_aggregate -= 1
    for i = 1:n
        xi = x[i]
        if xi > 0
            x[i] = xi - 1
        elseif xi == -n
            x[i] = -1
        elseif xi < 0  # Pass 2 tentative: x[i] = -x_best
            x[i] = -xi - 1
        end
        # unagg[i] == true nodes keep x[i] == 0; handled below.
    end

    # Pass 3: form new aggregates from nodes that were not reached in Pass 1 or 2.
    # Each seed i pulls in every unaggregated neighbour (identified via `unagg`,
    # not x[row] == 0, to avoid confusion with the 0-indexed aggregate-0).
    for i = 1:n
        unagg[i] || continue

        x[i] = next_aggregate

        for j in nzrange(S, i)
            row = S.rowval[j]
            if unagg[row]
                x[row] = next_aggregate
                unagg[row] = false
            end
        end
        unagg[i] = false
        next_aggregate += 1
    end

    M, N = (n, next_aggregate)

    # Aggregation of leftovers
    if isempty(x) || minimum(x) == -1
        mask = x .!= -1
        I = collect(R, 1:n)[mask]
        J = x[mask] .+ R(1)
        #J = x[mask] + 1
        V = ones(eltype(S), length(J))
        AggOp = sparse(J,I,V,N,M)
    else
        Tp = collect(R, 1:n+1)
        x .= x .+ R(1)
        #x .= x .+ 1
        Tx = ones(eltype(S), length(x))
        AggOp = SparseMatrixCSC(N, M, Tp, x, Tx)
    end

    AggOp
end
