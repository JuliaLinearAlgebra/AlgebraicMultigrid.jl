abstract type Strength end
struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function strength_of_connection(c::Classical, 
                A::SparseMatrixCSC{Tv,Ti}) where {Ti,Tv}

    θ = c.θ

    m, n = size(A)
    T = copy(A')

    for i = 1:n
        _m = find_max_off_diag(T, i)
        threshold = θ * _m
        for j in nzrange(T, i)
            row = T.rowval[j]
            val = T.nzval[j]

            if row != i
                if abs(val) >= threshold
                    T.nzval[j] = abs(val)
                else
                    T.nzval[j] = 0
                end
            end

        end
    end

    dropzeros!(T)

    scale_cols_by_largest_entry!(T)

    copy(T'), T
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

struct SymmetricStrength{T} <: Strength
    θ::T
end
SymmetricStrength() = SymmetricStrength(0.)

function strength_of_connection(s::SymmetricStrength{T}, A, bsr_flag = false) where {T}

    θ = s.θ

    if bsr_flag && θ == 0
        S = SparseMatrixCSC(size(A)...,
                    A.colptr, A.rowval, ones(eltype(A), size(A.rowval)))
        return S
    else
        S = deepcopy(A)
    end
    n = size(A, 1)
    diags = Vector{eltype(A)}(undef, n)

    for i = 1:n
        diag = zero(eltype(A))
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if row == i
                diag += val
            end
        end
        diags[i] = norm(diag)
    end

    for i = 1:n
        eps_Aii = θ * θ * diags[i]
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if row != i
                if val*val < eps_Aii * diags[row]
                    S.nzval[j] = 0
                end
            end
        end
    end

    dropzeros!(S)

    S.nzval .= abs.(S.nzval)
    scale_cols_by_largest_entry!(S)    

    S
end
