struct Solver{S,T,P,PS}
    strength::S
    CF::T
    presmoother::P
    postsmoother::PS
    max_levels::Int64
    max_coarse::Int64
end

function ruge_stuben(A::SparseMatrixCSC{Ti,Tv};
                strength = Classical(0.25),
                CF = RS(),
                presmoother = GaussSeidel(),
                postsmoother = GaussSeidel(),
                max_levels = 10,
                max_coarse = 10) where {Ti,Tv}

    s = Solver(strength, CF, presmoother,
                postsmoother, max_levels, max_levels)

    levels = Vector{Level{Ti,Tv}}()

    while length(levels) + 1 < max_levels && size(A, 1) > max_coarse
        A = extend_heirarchy!(levels, strength, CF, A)
        #if size(A, 1) <= max_coarse
        #    break
        #end
    end
    MultiLevel(levels, A, presmoother, postsmoother)
end

function extend_heirarchy!(levels::Vector{Level{Ti,Tv}}, strength, CF, A::SparseMatrixCSC{Ti,Tv}) where {Ti,Tv}
    S, T = strength_of_connection(strength, A)
    splitting = split_nodes(CF, S)
    P, R = direct_interpolation(A, T, splitting)
    push!(levels, Level(A, P, R))
    A = R * A * P
end

function direct_interpolation(A, T, splitting)
    fill!(T.nzval, eltype(A)(1))
    T .= A .* T
    Pp = rs_direct_interpolation_pass1(T, splitting)
    Px, Pj, Pp = rs_direct_interpolation_pass2(A, T, splitting, Pp)

    R = SparseMatrixCSC(maximum(Pj), size(A, 1), Pp, Pj, Px)
    P = copy(R')

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
                if splitting[row] == C_NODE && row != i
                    nnzplus1 += 1
                end
            end
        end
        Bp[i+1] = nnzplus1
    end
    Bp
 end


function rs_direct_interpolation_pass2(A::SparseMatrixCSC{Tv,Ti},
                                                T::SparseMatrixCSC{Tv, Ti},
                                                splitting::Vector{Ti},
                                                Bp::Vector{Ti}) where {Tv,Ti}
                                                
    Bx = zeros(Tv, Bp[end] - 1)
    Bj = zeros(Ti, Bp[end] - 1)

    n = size(A, 1)

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
                if splitting[row] == C_NODE && row != i
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
            for j in nzrange(A, i)
                row = A.rowval[j]
                aval = A.nzval[j]
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
            alpha = sum_all_neg / sum_strong_neg
            beta = sum_all_pos / sum_strong_pos

            if sum_strong_pos == 0
                diag += sum_all_pos
                beta = zero(beta)
            end

            neg_coeff = -1 * alpha / diag
            pos_coeff = -1 * beta / diag

            nnz = Bp[i]

            for j in nzrange(T, i)
                row = T.rowval[j]
                sval = T.nzval[j]
                if splitting[row] == C_NODE && row != i
                    Bj[nnz] = row
                    if sval < 0
                        Bx[nnz] = neg_coeff * sval
                    else
                        Bx[nnz] = pos_coeff * sval
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
