abstract type Smoother end
abstract type Sweep end
struct SymmetricSweep <: Sweep
end
struct ForwardSweep <: Sweep
end
struct BackwardSweep <: Sweep
end
struct GaussSeidel{S} <: Smoother
    sweep::S
    iter::Int
end
GaussSeidel(iter = 1) = GaussSeidel(SymmetricSweep(), iter)
GaussSeidel(f::ForwardSweep) = GaussSeidel(f, 1)
GaussSeidel(b::BackwardSweep) = GaussSeidel(b, 1)
GaussSeidel(s::SymmetricSweep) = GaussSeidel(s, 1)

function (s::GaussSeidel{S})(A, x, b) where {S<:Sweep}
    for i in 1:s.iter
        if S === ForwardSweep || S === SymmetricSweep
            gs!(A, b, x, 1, 1, size(A, 1))
        end
        if S === BackwardSweep || S === SymmetricSweep
            gs!(A, b, x, size(A, 1), -1, 1)
        end
    end
end

function gs!(A, b, x, start, step, stop)
    n = size(A, 1)
    z = zero(eltype(A))
    for i = start:step:stop
        rsum = z
        d = z
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if i == row
                d = val
            else
                rsum += val * x[row]
            end
        end

        if d != 0
            x[i] = (b[i] - rsum) / d
        end
    end
end

struct Jacobi{T,TX} <: Smoother
    ω::T
    temp::TX
end
Jacobi(ω=0.5, x) = Jacobi{T,TX}(ω, similar(x))

function (jacobi::Jacobi)(A, x, b, ω, start, step, stop)

    one = one(eltype(A))
    temp = jacobi.temp

    @inbounds for col in 1:size(x, 2)
        for i = start:step:stop
            temp[i, col] = x[i, col]
        end

        for i = start:step:stop
            rsum = zero(eltype(A))
            diag = zero(eltype(A))

            for j in nzrange(A, i)
                row = A.rowval[j]
                val = A.nzval[j]

                if row == i
                    diag = val
                else
                    rsum += val * temp[row, col]
                end
            end

            if diag != 0
                x[i, col] = (one - ω) * temp[i, col] + ω * ((b[i, col] - rsum) / diag)
            end
        end
    end
end

struct JacobiProlongation{T}
    ω::T
end

struct DiagonalWeighting
end
struct LocalWeighting
end

function (j::JacobiProlongation)(A, T, S, B, degree = 1, weighting = LocalWeighting())
    D_inv_S = weight(weighting, A, j.ω)
    P = T
    for i = 1:degree
        P = P - (D_inv_S * P)
    end
    P
end

function weight(::DiagonalWeighting, S, ω)
    D_inv = 1 ./ diag(S)
    D_inv_S = scale_rows(S, D_inv)
    (eltype(S)(ω) / approximate_spectral_radius(D_inv_S)) * D_inv_S
    # (ω) * D_inv_S
end

function weight(::LocalWeighting, S, ω)
    #=D = abs.(S) * ones(eltype(S), size(S, 1))
    D_inv = 1 ./ D[find(D)]
    D_inv_S = scale_rows(S, D_inv)
    eltype(S)(ω) * D_inv_S=#
    D = zeros(eltype(S), size(S,1))
    for i = 1:size(S, 1)
        for j in nzrange(S, i)
            row = S.rowval[j]
            val = S.nzval[j]
            D[row] += abs(val)
        end
    end
    for i = 1:size(D, 1)
        if D[i] != 0
            D[i] = 1/D[i]
        end
    end
    D_inv_S = scale_rows(S, D)
    # eltype(S)(ω) * D_inv_S
    rmul!(D_inv_S, eltype(S)(ω))
end

#approximate_spectral_radius(A) =
#    eigs(A, maxiter = 15, tol = 0.01, ritzvec = false)[1][1] |> real

function scale_rows!(ret, S, v)
    n = size(S, 1)
    for i = 1:n
        for j in nzrange(S, i)
            row = S.rowval[j]
            ret.nzval[j] *= v[row]
        end
    end
    ret
end
scale_rows(S, v) = scale_rows!(deepcopy(S), S,  v)