struct StandardAggregation
    min_aggregate_size::Int
end

StandardAggregation() = StandardAggregation(1)

function (agg::StandardAggregation)(S::SparseMatrixCSC{T,R}) where {T,R}
    n = size(S, 1)
    x = zeros(R, n)
    y = zeros(R, n)

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
            aggregate_size = 0
            for j in nzrange(S, i)
                row = S.rowval[j]
                x[row] = next_aggregate
                aggregate_size += 1
            end

            # Reject aggregate if it is too small
            if aggregate_size < agg.min_aggregate_size
                x[i] = 0
                for j in nzrange(S, i)
                    row = S.rowval[j]
                    x[row] = 0
                end
            else
                y[next_aggregate] = i
                next_aggregate += 1
            end
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

    # Shift assignments by 1 and apply the temporary assignments
    next_aggregate -= 1
    for i = 1:n
        xi = x[i]
        if xi != 0
            if xi > 0
                x[i] = xi - 1
            elseif xi == -n
                x[i] = -1
            else
                x[i] = -xi - 1
            end
            continue
        end

        x[i] = next_aggregate
        y[next_aggregate + 1] = i

        for j in nzrange(S, i)
            row = S.rowval[j]

            if x[row] == 0
                x[row] = next_aggregate
            end
        end

        next_aggregate += 1
    end

    y = y[1:next_aggregate]
    M, N = (n, next_aggregate)

    # Pass 3: Aggregation of leftovers
    if minimum(x) == -1
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
