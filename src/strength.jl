abstract type Strength end
struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function strength_of_connection{T, Ti, Tv}(c::Classical{T}, A::SparseMatrixCSC{Tv, Ti})

    θ = c.θ

    m, n = size(A)
    nz = nnz(A)
    I = zeros(Ti, nz)
    J = zeros(Ti, nz)
    V = zeros(float(Tv), nz)
    k = 1

    for i = 1:n
        #neighbors = A[:,i]
        #_m = find_max_off_diag(neighbors, i)
        _m = find_max_off_diag(A, i)
        threshold = θ * _m
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if abs(val) >= threshold
                if row != i
                    I[k] = row
                    J[k] = i
                    V[k] = abs(val)
                    k += 1
                end
            end

            if row == i
                I[k] = row
                J[k] = i
                V[k] = val
                k += 1
            end
        end
    end
    deleteat!(I, k:nz)
    deleteat!(J, k:nz)
    deleteat!(V, k:nz)

    S = sparse(I, J, V, m, n)

    scale_cols_by_largest_entry!(S)


    S', S
end

#=function find_max_off_diag(neighbors, col)
    maxval = zero(eltype(neighbors))
    for i in 1:length(neighbors.nzval)
        maxval = max(maxval, ifelse(neighbors.nzind[i] == col, 0, abs(neighbors.nzval[i])))
    end
    return maxval
end=#
function find_max_off_diag(A, i)
    m = 0
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        if row != i
            m = max(m, abs(val))
        end
    end
    m
end

function find_max(A, i)
    m = zero(eltype(A))
    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]
        m = max(m, abs(val))
    end
    m
end

function scale_cols_by_largest_entry!(A::SparseMatrixCSC)

    n = size(A, 1)
    for i = 1:n
        #_m = maximum(A[:,i])
        _m = find_max(A, i)
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end

    A
end
