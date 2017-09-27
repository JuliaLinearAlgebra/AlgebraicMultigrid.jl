struct Solver{S,T,P,PS}
    strength::S
    CF::T
    presmoother::P
    postsmoother::PS
    max_levels::Int64
    max_coarse::Int64
end

function ruge_stuben(A::SparseMatrixCSC;
                strength = Classical(0.25),
                CF = RS(),
                presmoother = GaussSeidel(),
                postsmoother = GaussSeidel(),
                max_levels = 10,
                max_coarse = 10)

    s = Solver(strength, CF, presmoother,
                postsmoother, max_levels, max_levels)

    levels = Vector{Level}()

    while length(levels) < max_levels
        A = extend_heirarchy!(levels, strength, CF, A)
        if size(levels[end].A, 1) < max_coarse
            break
        end
    end
    MultiLevel(levels, presmoother, postsmoother)
end

function extend_heirarchy!(levels::Vector{Level}, strength, CF, A)
    S = strength_of_connection(strength, A)
    splitting = split_nodes(CF, S)
    P, R = direct_interpolation(A, S, splitting)
    push!(levels, Level(A, P, R))
    A = R * A * P
end

function direct_interpolation{T,V}(A::T, S::T, splitting::Vector{V})

    fill!(S.nzval, 1.)
    S = A .* S
    Pp = rs_direct_interpolation_pass1(S, A, splitting)
    Pp = Pp .+ 1

    Px, Pj = rs_direct_interpolation_pass2(A, S, splitting, Pp)

    # Px .= abs.(Px)
    Pj = Pj .+ 1

    R = SparseMatrixCSC(maximum(Pj), size(A, 1), Pp, Pj, Px)
    P = R'

    P, R
end


function rs_direct_interpolation_pass1(S, A, splitting)

     Bp = zeros(Int, size(A.colptr))
     #=Sp = S.colptr
     Sj = S.rowval
     n_nodes = size(A, 1)
     nnz = 0
     for i = 1:n_nodes
         if splitting[i] == C_NODE
             nnz += 1
         else
            for jj = Sp[i]:Sp[i+1]
                jj > length(Sj) && continue
                if splitting[Sj[jj]] == C_NODE && Sj[jj] != i
                    nnz += 1
                end
            end
        end
         Bp[i+1] = nnz
     end=#
     n = size(A, 1)
     nnz = 0
     for i = 1:n
         if splitting[i] == C_NODE
             nnz += 1
         else
             for j in nzrange(S, i)
                 row = S.rowval[j]
                 if splitting[row] == C_NODE && row != i
                     nnz += 1
                end
             end
        end
        Bp[i+1] = nnz
     end
     Bp
 end


 function rs_direct_interpolation_pass2{Tv, Ti}(A::SparseMatrixCSC{Tv,Ti},
                                                S::SparseMatrixCSC{Tv,Ti},
                                                splitting::Vector{Ti},
                                                Bp::Vector{Ti})


    Bx = zeros(Float64, Bp[end])
    Bj = zeros(Ti, Bp[end])

    n = size(A, 1)

    for i = 1:n
        if splitting[i] == C_NODE
            Bj[Bp[i]] = i
            Bx[Bp[i]] = 1
        else
            sum_strong_pos = zero(Tv)
            sum_strong_neg = zero(Tv)
            for j in nzrange(S, i)
                row = S.rowval[j]
                sval = S.nzval[j]
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
                beta = 0
            end

            neg_coeff = -1 * alpha / diag
            pos_coeff = -1 * beta / diag

            nnz = Bp[i]

            for j in nzrange(S, i)
                row = S.rowval[j]
                sval = S.nzval[j]
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
    sum = 0
    for i = 1:n
        m[i] = sum
        sum += splitting[i]
    end
    for i = 1:Bp[n]
        Bj[i] == 0 && continue
        Bj[i] = m[Bj[i]]
    end

    #=Ap = A.colptr
    Aj = A.rowval
    Ax = A.nzval
    Sp = S.colptr
    Sj = S.rowval
    Sx = S.nzval
    Bj = zeros(Ti, Bp[end])
    Bx = zeros(Float64, Bp[end])
    n_nodes = size(A, 1)

    for i = 1:n_nodes
        if splitting[i] == C_NODE
            Bj[Bp[i]] = i
            Bx[Bp[i]] = 1
        else
            sum_strong_pos = 0
            sum_strong_neg = 0
            for jj = Sp[i]: Sp[i+1]
                jj > length(Sj) && continue
                if splitting[Sj[jj]] == C_NODE && Sj[jj] != i
                    if Sx[jj] < 0
                        sum_strong_neg += Sx[jj]
                    else
                        sum_strong_pos += Sx[jj]
                    end
                end
            end

            sum_all_pos = 0
            sum_all_neg = 0
            diag = 0
            for jj = Ap[i]:Ap[i+1]
                jj > length(Aj) && continue
                if Aj[jj] == i
                    @show Ax[jj]
                    diag += Ax[jj]
                else
                    if Ax[jj] < 0
                        sum_all_neg += Ax[jj]
                    else
                        sum_all_pos += Ax[jj]
                    end
                end
            end

            alpha = sum_all_neg / sum_strong_neg
            beta  = sum_all_pos / sum_strong_pos
            @show alpha
            @show beta
            @show diag

            if sum_strong_pos == 0
                diag += sum_all_pos
                beta = 0
            end

            neg_coeff = -1 * alpha / diag
            pos_coeff = -1 * beta / diag

            @show neg_coeff
            @show pos_coeff

            nnz = Bp[i]
            for jj = Sp[i]:Sp[i+1]
                jj > length(Sj) && continue
                if splitting[Sj[jj]] == C_NODE && Sj[jj] != i
                    Bj[nnz] = Sj[jj]
                    if Sx[jj] < 0
                        Bx[nnz] = neg_coeff * Sx[jj]
                    else
                        Bx[nnz] = pos_coeff * Sx[jj]
                    end
                    @show Bx[nnz]
                    nnz += 1
                end
            end
        end
    end

   m = zeros(Ti, n_nodes)
   sum = 0
   for i = 1:n_nodes
       m[i] = sum
       sum += splitting[i]
   end
   for i = 1:Bp[n_nodes]
       Bj[i] == 0 && continue
       Bj[i] = m[Bj[i]]
   end =#

   Bx, Bj, Bp
end
