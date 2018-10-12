struct Solver{S,T,P,PS}
    strength::S
    CF::T
    presmoother::P
    postsmoother::PS
    max_levels::Int64
    max_coarse::Int64
end

function ruge_stuben(_A::Union{TA, Symmetric{Ti, TA}, Hermitian{Ti, TA}}, 
                ::Type{Val{bs}}=Val{1};
                strength = Classical(0.25),
                CF = RS(),
                presmoother = GaussSeidel(),
                postsmoother = GaussSeidel(),
                max_levels = 10,
                max_coarse = 10,
                coarse_solver = Pinv) where {Ti,Tv,bs,TA<:SparseMatrixCSC{Ti,Tv}}

    s = Solver(strength, CF, presmoother,
                postsmoother, max_levels, max_levels)

    if _A isa Symmetric && Ti <: Real || _A isa Hermitian
        A = _A.data
        At = A
        symmetric = true
        @static if VERSION < v"0.7-"
            levels = Vector{Level{TA, TA}}()
        else
            levels = Vector{Level{TA, Adjoint{Ti, TA}, TA}}()
        end
    else
        symmetric = false
        A = _A
        At = adjoint(A)
        @static if VERSION < v"0.7-"
            levels = Vector{Level{TA, TA, TA}}()
        else
            levels = Vector{Level{TA, Adjoint{Ti, TA}, TA}}()
        end
    end
    w = MultiLevelWorkspace(Val{bs}, eltype(A))
    residual!(w, size(A, 1))

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A = extend_heirarchy!(levels, strength, CF, A, symmetric)
        coarse_x!(w, size(A, 1))
        coarse_b!(w, size(A, 1))
        residual!(w, size(A, 1))
    end

    MultiLevel(levels, A, coarse_solver(A), presmoother, postsmoother, w)
end

function extend_heirarchy!(levels, strength, CF, A::SparseMatrixCSC{Ti,Tv}, symmetric) where {Ti,Tv}
    if symmetric
        At = A
    else
        At = adjoint(A)
    end
    S, T = strength(At)
    splitting = CF(S)
    P, R = direct_interpolation(At, T, splitting)
    push!(levels, Level(A, P, R))
    A = R * A * P
end

function direct_interpolation(At, T, splitting)
    fill!(T.nzval, eltype(At)(1))
    T .= At .* T
    
    Pp = rs_direct_interpolation_pass1(T, splitting)
    Px, Pj, Pp = rs_direct_interpolation_pass2(At, T, splitting, Pp)
    R = SparseMatrixCSC(maximum(Pj), size(At, 1), Pp, Pj, Px)
    P = R'

    P, R
end

# calculates the number of nonzeros in each column of the interpolation matrix
function rs_direct_interpolation_pass1(T, splitting)
    n = size(T, 2)
    Bp = ones(Int, n+1)
    nnzplus1 = 1
    for i = 1:n
        if splitting[i] == C_NODE
            nnzplus1 += 1
        else
            for j in nzrange(T, i)
                row = T.rowval[j]
                if splitting[row] == C_NODE
                    nnzplus1 += 1
                end
            end
        end
        Bp[i+1] = nnzplus1
    end
    Bp
 end


function rs_direct_interpolation_pass2(At::SparseMatrixCSC{Tv,Ti},
                                                T::SparseMatrixCSC{Tv, Ti},
                                                splitting::Vector{Ti},
                                                Bp::Vector{Ti}) where {Tv,Ti}
                                                
    Bx = zeros(Tv, Bp[end] - 1)
    Bj = zeros(Ti, Bp[end] - 1)

    n = size(At, 1)

    for i = 1:n
        if splitting[i] == C_NODE
            Bj[Bp[i]] = i
            Bx[Bp[i]] = 1
        else
            sum_strong_pos = zero(Tv)
            sum_strong_neg = zero(Tv)
            for j in nzrange(T, i)
                row = T.rowval[j]
                sval = T.nzval[j]
                if splitting[row] == C_NODE
                    if sval < 0
                        sum_strong_neg += sval
                    else
                        sum_strong_pos += sval
                    end
                end
            end
            sum_all_pos = zero(Tv)
            sum_all_neg = zero(Tv)
            diag = zero(Tv)
            for j in nzrange(At, i)
                row = At.rowval[j]
                aval = At.nzval[j]
                if row == i
                    diag += aval
                else
                    if aval < 0
                        sum_all_neg += aval
                    else
                        sum_all_pos += aval
                    end
                end
            end

            if sum_strong_pos == 0
                beta = zero(diag)
                if diag >= 0
                    diag += sum_all_pos
                end
            else
                beta = sum_all_pos / sum_strong_pos
            end

            if sum_strong_neg == 0
                alpha = zero(diag)
                if diag < 0
                    diag += sum_all_neg
                end
            else
                alpha = sum_all_neg / sum_strong_neg
            end

            if isapprox(diag, 0, atol=eps(Tv))
                neg_coeff = Tv(0)
                pos_coeff = Tv(0)
            else
                neg_coeff = alpha / diag
                pos_coeff = beta / diag
            end

            nnz = Bp[i]
            for j in nzrange(T, i)
                row = T.rowval[j]
                sval = T.nzval[j]
                if splitting[row] == C_NODE
                    Bj[nnz] = row
                    if sval < 0
                        Bx[nnz] = abs(neg_coeff * sval)
                    else
                        Bx[nnz] = abs(pos_coeff * sval)
                    end
                    nnz += 1
                end
            end
        end
    end

    m = zeros(Ti, n)
    sum = zero(Ti)
    for i = 1:n
        m[i] = sum
        sum += splitting[i]
    end
    Bj .= m[Bj] .+ 1

    Bx, Bj, Bp
end
