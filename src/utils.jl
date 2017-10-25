function approximate_spectral_radius(A, tol = 0.01,
                                        maxiter = 15, restart = 5)


    symmetric = false

    # Initial guess
    v0 = rand(size(A,1))
    maxiter = min(size(A, 1), maxiter)
    ev = zeros(eltype(A), maxiter)
    max_index = 0
    X = zeros(size(A,1), maxiter)

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
        if abs(error) / abs(ev[max_index]) < tol || flag
            # v0 = X * evect[:, max_index]
            A_mul_B!(v0, X, evect[:, max_index])
            break
        else
            # v0 = X * evect[:, max_index]
            A_mul_B!(v0, X, evect[:, max_index])
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
    flag = false

    for j = 1:maxiter
        w = A * V[end]
        # V[j+1] = A * V[j]
        # w = V[j+1]
        # A_mul_B!(w, A, V[j])
        for (i,v) in enumerate(V)
        # for i = 1:j
            v = V[i]
            H[i,j] = dot(v, w)
            BLAS.axpy!(-H[i,j], v, w)
        end
        H[j+1,j] = norm(w)
        if H[j+1, j] < eps()
            flag = true
            if H[j+1,j] != 0
                scale!(w, 1/H[j+1,j])
                push!(V, w)
                break
            end
        end

        #w = w / H[j+1, j]
        scale!(w, 1/H[j+1,j])
        push!(V, w)
    end

    Eigs, Vects = eig(H[1:maxiter, 1:maxiter], eye(maxiter))

    Vects, Eigs, H, V, flag
end
