abstract type Strength end
struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function strength_of_connection{T}(c::Classical{T}, A::SparseMatrixCSC)

    θ = c.θ
    I = Int[]
    J = Int[]
    V = Float64[]

    m, n = size(A)

    for i = 1:n
        neighbors = A[:,i]
        _m = find_max_off_diag(neighbors, i)
        threshold = θ * _m
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
    S = sparse(I, J, V, m, n)

    scale_cols_by_largest_entry!(S)

    S'
end

function find_max_off_diag(neighbors, col)
    maxval = zero(eltype(neighbors))
    for i in 1:length(neighbors.nzval)
        maxval = max(maxval, ifelse(neighbors.nzind[i] == col, 0, abs(neighbors.nzval[i])))
    end
    return maxval
end

function scale_cols_by_largest_entry!(A::SparseMatrixCSC)

    n = size(A, 1)
    for i = 1:n
        _m = maximum(A[:,i])
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end

    A
end
