function approximate_spectral_radius(A, tol = 0.01,
                                        maxiter = 15, restart = 5)


    symmetric = false

    # Initial guess
    v0 = rand(size(A,1))
    ev = zeros(Complex{eltype(A)}, maxiter)
    max_index = 0

    for i in 1:restart+1
        evect, ev, H, V, flag =
                    approximate_eigenvalues(A, tol, maxiter,
                                            symmetric, v0)
        nvecs = size(ev, 1)
        m, max_index = findmax(abs.(ev))
        error = H[nvecs, nvecs-1] * evect[end, max_index]
        if abs(error) / abs(ev[max_index]) < tol || flag
            @show size(hcat(V[1:end-1]...))
            @show size(evect[:, max_index])
            v0 = hcat(V[1:end-1]...) * evect[:, max_index]
            break
        else
            v0 = hcat(V[1:end-1]...) * evect[:, max_index]
        end
    end

    ρ = abs(ev[max_index])

end

function approximate_eigenvalues(A, tol, maxiter, symmetric, v0)

    maxiter = min(size(A, 1), maxiter)
    v0 /= norm(v0)
    H = zeros(eltype(A), maxiter + 1, maxiter)
    V = [v0]
    flag = false

    for j = 1:maxiter
        w = A * V[end]
        for (i,v) in enumerate(V)
            H[i,j] = dot(v, w)
            Base.BLAS.axpy!(-H[i,j], v, w)
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

        w = w / H[j+1, j]
        push!(V, w)
    end

    Eigs, Vects = eig(H[1:maxiter, 1:maxiter])

    Vects, Eigs, H, V, flag
end
