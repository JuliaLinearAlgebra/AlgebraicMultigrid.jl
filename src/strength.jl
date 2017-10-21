abstract type Strength end
struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function strength_of_connection{T, Ti, Tv}(c::Classical{T}, A::SparseMatrixCSC{Tv, Ti})

    θ = c.θ

    m, n = size(A)
    S = deepcopy(A)

    for i = 1:n
        _m = find_max_off_diag(A, i)
        threshold = θ * _m
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]

            if row != i
                if abs(val) >= threshold
                    S.nzval[j] = abs(val)
                else
                    S.nzval[j] = 0
                end
            end

        end
    end

    dropzeros!(S)

    scale_cols_by_largest_entry!(S)

    S', S
end

function find_max_off_diag(A, i)
    m = zero(eltype(A))
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
        m = max(m, val)
    end
    m
end

function scale_cols_by_largest_entry!(A::SparseMatrixCSC)

    n = size(A, 1)
    for i = 1:n
        _m = find_max(A, i)
        for j in nzrange(A, i)
            A.nzval[j] /= _m
        end
    end
    A
end
