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
GaussSeidel(; iter = 1) = GaussSeidel(SymmetricSweep(), iter)
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
    @inbounds for col in 1:size(x, 2)
        for i in start:step:stop
            rsum = z
            d = z
            for j in nzrange(A, i)
                row = A.rowval[j]
                val = A.nzval[j]
                d = ifelse(i == row, val, d)
                rsum += ifelse(i == row, z, val * x[row, col])
            end
            x[i, col] = ifelse(d == 0, x[i, col], (b[i, col] - rsum) / d)
        end
    end
end

struct Jacobi{T,TX} <: Smoother
    ω::T
    temp::TX
    iter::Int
end
Jacobi(ω, x::TX; iter=1) where {T, TX<:AbstractArray{T}} = Jacobi{T,TX}(ω, similar(x), iter)
Jacobi(x::TX, ω=0.5; iter=1) where {T, TX<:AbstractArray{T}} = Jacobi{T,TX}(ω, similar(x), iter)

function (jacobi::Jacobi)(A, x, b)

    ω = jacobi.ω
    one = Base.one(eltype(A))
    temp = jacobi.temp
    z = zero(eltype(A))

    for i in 1:jacobi.iter
        @inbounds for col = 1:size(x, 2)
            for i = 1:size(A, 1)
                temp[i, col] = x[i, col]
            end

            for i = 1:size(A, 1)
                rsum = z
                diag = z

                for j in nzrange(A, i)
                    row = A.rowval[j]
                    val = A.nzval[j]

                    diag = ifelse(row == i, val, diag)
                    rsum += ifelse(row == i, z, val * temp[row, col])
                end

                xcand = (one - ω) * temp[i, col] + ω * ((b[i, col] - rsum) / diag)
                x[i, col] = ifelse(diag == 0, x[i, col], xcand)
            end
        end
    end
end

#=
using KissThreading: tmap!

struct ParallelJacobi{T,TX} <: Smoother
    ω::T
    temp::TX
end
ParallelJacobi(ω, x::TX) where {T, TX<:AbstractArray{T}} = ParallelJacobi{T,TX}(ω, similar(x))
ParallelJacobi(x::TX, ω = 0.5) where {T, TX<:AbstractArray{T}} = ParallelJacobi{T,TX}(ω, similar(x))

struct ParallelTemp{TX, TI}
    temp::TX
    col::TI
end
(ptemp::ParallelTemp)(i) = ptemp.temp[i, ptemp.col]

struct ParallelJacobiMapper{TA, TX, TB, TTemp, TF, TI}
    A::TA
    x::TX
    b::TB
    temp::TTemp
    ω::TF
    col::TI
end
function (pjacobmapper::ParallelJacobiMapper)(i)
    A = pjacobmapper.A
    x = pjacobmapper.x
    b = pjacobmapper.b
    temp = pjacobmapper.temp
    ω = pjacobmapper.ω
    col = pjacobmapper.col

    one = Base.one(eltype(A))
    z = zero(eltype(A))
    rsum = z
    diag = z

    for j in nzrange(A, i)
        row = A.rowval[j]
        val = A.nzval[j]

        diag = ifelse(row == i, val, diag)
        rsum += ifelse(row == i, z, val * temp[row, col])
    end
    xcand = (one - ω) * temp[i, col] + ω * ((b[i, col] - rsum) / diag)
    
    return ifelse(diag == 0, x[i, col], xcand)
end

function (jacobi::ParallelJacobi)(A, x, b)
    ω = jacobi.ω
    temp = jacobi.temp
    for col = 1:size(x, 2)
        @views tmap!(ParallelTemp(temp, col), x[1:size(A, 1), col], 1:size(A, 1))
        @views tmap!(ParallelJacobiMapper(A, x, b, temp, ω, col), 
            x[1:size(A, 1), col], 1:size(A, 1))
    end
end
=#

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

struct SOR{S} <: Smoother
    ω::Float64
    sweep::S
    iter::Int
end

SOR(ω; iter = 1) = SOR(ω, SymmetricSweep(), iter)
SOR(ω, f::ForwardSweep) = SOR(ω, f, 1)
SOR(ω, b::BackwardSweep) = SOR(ω, b, 1)
SOR(ω, s::SymmetricSweep) = SOR(ω, s, 1)

function (sor::SOR{S})(A, x, b) where {S<:Sweep}
    for i in 1:sor.iter
        if S === ForwardSweep || S === SymmetricSweep
            sor_step!(A, b, x, sor.ω, 1, 1, size(A, 1))
        end
        if S === BackwardSweep || S === SymmetricSweep
            sor_step!(A, b, x, sor.ω, size(A, 1), -1, 1)
        end
    end
end

function sor_step!(A, b, x, ω, start, step, stop)
    n = size(A, 1)
    z = zero(eltype(A))
    @inbounds for col in 1:size(x, 2)
        for i in start:step:stop
            rsum = z
            d = z
            for j in nzrange(A, i)
                row = A.rowval[j]
                val = A.nzval[j]
                d = ifelse(i == row, val, d)
                rsum += ifelse(i == row, z, val * x[row, col])
            end
            x[i, col] = ifelse(d == 0, x[i, col], (1 - ω) * x[i, col] + (ω / d) * (b[i, col] - rsum))
        end
    end
end
