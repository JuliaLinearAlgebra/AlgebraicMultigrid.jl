function adjoint(A)
    copy(A')
end

function approximate_spectral_radius(A, tol = 0.01,
                                        maxiter = 15, restart = 5)


    symmetric = false

    # Initial guess
    v0 = rand(eltype(A), size(A,2))
    maxiter = min(size(A, 1), maxiter)
    ev = zeros(eltype(A), maxiter)
    max_index = 0
    X = zeros(eltype(A), size(A,1), maxiter)

    for i in 1:restart+1
        evect, ev, H, V, flag =
                    approximate_eigenvalues(A, tol, maxiter,
                                            symmetric, v0)
        nvecs = size(ev, 1)
        # X = hcat(V[1:end-1]...)
        copy_V!(X, V)
        # m, max_index = findmax(abs.(ev))
        m, max_index = findmaxabs(ev)
        error = H[nvecs, nvecs-1] * evect[end, max_index]
        @views mul!(v0, X, evect[:, max_index])
        if (abs(error) / abs(ev[max_index]) < tol) || flag
            # v0 = X * evect[:, max_index]
            break
        end
    end
    ρ = abs(ev[max_index])
end

function findmaxabs(arr)
    m = abs(arr[1])
    m_i = 1
    for i = 2:size(arr, 1)
        x = abs(arr[i])
        if x > m
            m = x
            m_i  = i
        end
    end
    m, m_i
end

function copy_V!(X, V)
    n = size(V,1)
    for i = 1:n-1
        X[:,i] = V[i]
    end
end

function approximate_eigenvalues(A, tol, maxiter, symmetric, v0)

    # maxiter = min(size(A, 1), maxiter)
    v0 ./= norm(v0)
    H = zeros(eltype(A), maxiter + 1, maxiter)
    V = [v0]
    # V = Vector{Vector{eltype(A)}}(maxiter + 1)
    # V = zeros(size(A,1), maxiter)
    # V[1] = v0
    breakdown = find_breakdown(eltype(A))
    flag = false

    for j = 1:maxiter
        w = A * V[end]
        # V[j+1] = A * V[j]
        # w = V[j+1]
        # mul!(w, A, V[j])
        for (i,v) in enumerate(V)
        # for i = 1:j
            v = V[i]
            H[i,j] = dot(conj.(v), w)
            BLAS.axpy!(-H[i,j], v, w)
        end
        H[j+1,j] = norm(w)
        if H[j+1, j] < breakdown
            flag = true
            if H[j+1,j] != 0
                rmul!(w, 1/H[j+1,j])
                push!(V, w)
                break
            end
        end

        #w = w / H[j+1, j]
        rmul!(w, 1/H[j+1,j])
        push!(V, w)
    end
    Eigs, Vects = (eigen(H[1:maxiter, 1:maxiter], Matrix{eltype(A)}(I, maxiter, maxiter))...,)

    Vects, Eigs, H, V, flag
end

find_breakdown(::Type{Float64}) = eps(Float64) * 10^6
find_breakdown(::Type{Float32}) = eps(Float64) * 10^3

using Base.Threads
#=function mul!(α::Number, A::SparseMatrixCSC, B::StridedVecOrMat, β::Number, C::StridedVecOrMat)
    A.n == size(B, 1) || throw(DimensionMismatch())
    A.m == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        for col = 1:A.n
            αxj = α*B[col,k]
            for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end
spmatvec(A::SparseMatrixCSC{TA,S}, x::StridedVector{Tx}) where {TA,S,Tx} =
    (T = promote_type(TA, Tx); mul!(one(T), A, x, zero(T), similar(x, T, A.m)))=#

# export spmatvec

#=@static if haskey(ENV, "JULIA_NUM_THREADS")
    info("Initializing AMG to use $(nthreads()) threads")
    import Base: *
    function *(A::SparseMatrixCSC{T,V}, b::Vector{T}) where {T,V}
        m, n = size(A)
        ret = zeros(T, m)
        @threads for i = 1:n
            for j in nzrange(A, i)
                row = A.rowval[j]
                val = A.nzval[j]
                ret[row] += val * b[i]
            end
        end
        ret
    end
end=#
