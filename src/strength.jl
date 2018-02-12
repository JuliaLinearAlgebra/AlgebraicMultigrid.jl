abstract type Strength end
struct Classical{T} <: Strength
    θ::T
end
Classical(;θ = 0.25) = Classical(θ)

function strength_of_connection(c::Classical{T}, 
                A::SparseMatrixCSC{Tv,Ti}) where {T,Ti,Tv}

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

struct SymmetricStrength{T} <: Strength
    θ::T
end
SymmetricStrength() = SymmetricStrength(0.)

function strength_of_connection{T}(s::SymmetricStrength{T}, A, bsr_flag = false)

    θ = s.θ

    if bsr_flag && θ == 0
        S = SparseMatrixCSC(size(A)...,
                    A.colptr, A.rowval, ones(eltype(A), size(A.rowval)))
        return S
    else
        S = deepcopy(A)
    end
    n = size(A, 1)
    diags = Vector{eltype(A)}(n)

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

    # S.nzval .= abs.(S.nzval)
    #for i = 1:size(S.nzval, 1)
    #     S.nzval[i] = abs(S.nzval[i])
    #end

    scale_cols_by_largest_entry!(S)
    
    for i = 1:size(S.nzval, 1)
         S.nzval[i] = abs(S.nzval[i])
    end

    S
end
