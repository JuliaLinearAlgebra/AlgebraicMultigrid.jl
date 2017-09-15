function classical(A::SparseMatrixCSC, θ::Float64)

    I = Int[]
    J = Int[]
    V = Float64[]

    m, n = size(A)

    for i = 1:n
        neighbors = A[:,i]
        m = find_max_off_diag(neighbors, i)
        threshold = θ * m
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if abs(val) >= threshold
                push!(I, row)
                push!(J, i)
                push!(V, abs(val))
            end
        end
    end
    S = sparse(I, J, V)

    scale_cols_by_largest_entry(S)
end

function find_max_off_diag(neighbors, col)
    max_offdiag = 0
    for (i,v) in enumerate(neighbors)
        if col != i
            max_offdiag = max(max_offdiag, abs(v))
        end
    end
    max_offdiag
end

function scale_cols_by_largest_entry(A::SparseMatrixCSC)

    m,n = size(A)

    I = zeros(Int, size(A.nzval))
    J = similar(I)
    V = zeros(size(A.nzval))

    k = 1
    for i = 1:n
        m = maximum(A[:,i])
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            I[k] = row
            J[k] = i
            V[k] = val / m
            k += 1
        end
    end

    sparse(I,J,V)
end
