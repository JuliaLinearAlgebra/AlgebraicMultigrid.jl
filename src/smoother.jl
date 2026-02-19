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

# Inplace version
function (config::GaussSeidel)(A, x, b)
    s = setup_smoother(config, A, x)
    ldiv!(x, s, b)
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

    for _ in 1:jacobi.iter
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

struct SOR{S <: Sweep, T} <: Smoother
    ω::T
    sweep::S
    iter::Int
end

SOR(ω; iter = 1) = SOR(ω, SymmetricSweep(), iter)
SOR(ω, s::Sweep) = SOR(ω, f, 1)

# Inplace version
function (config::SOR)(A, x, b)
    s = setup_smoother(config, A, x)
    ldiv!(x, s, b)
end

# This is essentially the same as IterativeSolvers.jl/src/stationary_sparse.jl to iron out the interface
# TODO move this into a smoother package to be maintained separately.

struct DiagonalIndices{Tv, Ti <: Integer}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::Vector{Ti}
end

function DiagonalIndices(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    # Check square?
    diag = Vector{Ti}(undef, A.n)

    for col = 1 : A.n
        r1 = Int(A.colptr[col])
        r2 = Int(A.colptr[col + 1] - 1)
        r1 = searchsortedfirst(A.rowval, col, r1, r2, Base.Order.Forward)
        if r1 > r2 || A.rowval[r1] != col || iszero(A.nzval[r1])
            throw(LinearAlgebra.SingularException(col))
        end
        diag[col] = r1
    end

    DiagonalIndices(A, diag)
end

function LinearAlgebra.ldiv!(y::AbstractVecOrMat{Tv}, D::DiagonalIndices{Tv,Ti}, x::AbstractVecOrMat{Tv}) where {Tv,Ti}
    for system_index in size(x, 2)
        @inbounds for row = 1 : D.matrix.n
            y[row, system_index] = x[row, system_index] / D.matrix.nzval[D.diag[row]]
        end
    end
    y
end

@inline Base.getindex(d::DiagonalIndices, i::Int) = d.diag[i]

struct FastLowerTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct FastUpperTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct StrictlyUpperTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct StrictlyLowerTriangular{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

struct OffDiagonal{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
    diag::DiagonalIndices{Tv,Ti}
end

"""
Forward substitution for the FastLowerTriangular type
"""
function forward_sub!(F::FastLowerTriangular, x::AbstractVecOrMat)
    A = F.matrix

    T = eltype(A)
    for system_index in 1:size(x, 2)
        @inbounds for col = 1 : A.n
            # Solve for diagonal element
            idx = F.diag[col]
            d   = A.nzval[idx]
            x[col, system_index] /= d

            # Substitute next values involving x[col, system_index]
            for i = idx + 1 : (A.colptr[col + 1] - 1)
                x[A.rowval[i], system_index] -= A.nzval[i] * x[col, system_index]
            end
        end
    end

    x
end

"""
Forward substitution
"""
function forward_sub!(α, F::FastLowerTriangular, x::AbstractVecOrMat, β, y::AbstractVecOrMat)
    A = F.matrix

    T = eltype(A)
    for system_index in 1:size(x, 2)
        @inbounds for col = 1 : A.n
            # Solve for diagonal element
            idx = F.diag[col]
            d   = A.nzval[idx]
            x[col, system_index] = α * x[col, system_index] / d + β * y[col, system_index]

            # Substitute next values involving x[col, system_index]
            for i = idx + 1 : (A.colptr[col + 1] - 1)
                x[A.rowval[i], system_index] -= A.nzval[i] * x[col, system_index]
            end
        end
    end

    return x
end


"""
Backward substitution for the FastUpperTriangular type
"""
function backward_sub!(F::FastUpperTriangular, x::AbstractVecOrMat)
    A = F.matrix

    T = eltype(A)
    for system_index in axes(x, 2)
        @inbounds for col = A.n : -1 : 1
            # Solve for diagonal element
            idx = F.diag[col]
            d   = A.nzval[idx]
            x[col, system_index] /= d

            # Substitute next values involving x[col, system_index]
            for i = A.colptr[col] : idx - 1
                x[A.rowval[i], system_index] -= A.nzval[i] * x[col, system_index]
            end
        end
    end

    return x
end

function backward_sub!(α, F::FastUpperTriangular, x::AbstractVecOrMat, β, y::AbstractVecOrMat)
    A = F.matrix

    T = eltype(A)
    for system_index in axes(x, 2)
        @inbounds for col = A.n : -1 : 1
            # Solve for diagonal element
            idx = F.diag[col]
            d   = A.nzval[idx]
            x[col, system_index] = α * x[col, system_index] / A.nzval[idx] + β * y[col, system_index]

            # Substitute next values involving x[col, system_index]
            for i = A.colptr[col] : idx - 1
                x[A.rowval[i], system_index] -= A.nzval[i] * x[col, system_index]
            end
        end
    end

    return x
end

"""
Computes z := α * U * x + β * y. Because U is StrictlyUpperTriangular
one can set z = x and update x in-place as x := α * U * x + β * y.
"""
function gauss_seidel_multiply!(α, U::StrictlyUpperTriangular, x::AbstractVecOrMat, β, y::AbstractVecOrMat, z::AbstractVecOrMat)
    A = U.matrix

    for system_index in axes(x, 2)
        for col = 1 : A.n
            αx = α * x[col, system_index]
            diag_index = U.diag[col]
            @inbounds for j = A.colptr[col] : diag_index - 1
                z[A.rowval[j], system_index] += A.nzval[j] * αx
            end
            z[col, system_index] = β * y[col, system_index]
        end
    end

    return z
end

"""
Computes z := α * L * x + β * y. Because A is StrictlyLowerTriangular
one can set z = x and update x in-place as x := α * L * x + β * y.
"""
function gauss_seidel_multiply!(α, L::StrictlyLowerTriangular, x::AbstractVecOrMat, β, y::AbstractVecOrMat, z::AbstractVecOrMat)
    A = L.matrix

    for system_index in axes(x, 2)
        for col = A.n : -1 : 1
            αx = α * x[col, system_index]
            z[col, system_index] = β * y[col, system_index]
            @inbounds for j = L.diag[col] + 1 : (A.colptr[col + 1] - 1)
                z[A.rowval[j], system_index] += A.nzval[j] * αx
            end
        end
    end

    return z
end

struct ForwardGaussSeidelSmoother{Tv, Ti}
    U::StrictlyUpperTriangular{Tv, Ti}
    L::FastLowerTriangular{Tv, Ti}
    iter::Int
end

function setup_smoother(config::GaussSeidel{<:ForwardSweep}, A, x)
    D = DiagonalIndices(A)
    ForwardGaussSeidelSmoother(StrictlyUpperTriangular(A, D), FastLowerTriangular(A, D), config.iter)
end

function LinearAlgebra.ldiv!(x, s::ForwardGaussSeidelSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # x ← L \ (-U * x + b)
        gauss_seidel_multiply!(-one(T), s.U, x, one(T), b, x)
        forward_sub!(s.L, x)
    end
    return nothing
end

struct BackwardGaussSeidelSmoother{Tv, Ti}
    U::FastUpperTriangular{Tv, Ti}
    L::StrictlyLowerTriangular{Tv, Ti}
    iter::Int
end

function setup_smoother(config::GaussSeidel{<:BackwardSweep}, A, x)
    D = DiagonalIndices(A)
    BackwardGaussSeidelSmoother(FastUpperTriangular(A, D), StrictlyLowerTriangular(A, D), config.iter)
end

function LinearAlgebra.ldiv!(x, s::BackwardGaussSeidelSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # x ← U \ (-L * x + b)
        gauss_seidel_multiply!(-one(T), s.L, x, one(T), b, x)
        backward_sub!(s.U, x)
    end
    return nothing
end

struct SymmetricGaussSeidelSmoother{Tv, Ti}
    sL::StrictlyLowerTriangular{Tv, Ti}
    sU::StrictlyUpperTriangular{Tv, Ti}
    L::FastLowerTriangular{Tv, Ti}
    U::FastUpperTriangular{Tv, Ti}
    iter::Int
end

function setup_smoother(config::GaussSeidel{<:SymmetricSweep}, A, x)
    D = DiagonalIndices(A)
    SymmetricGaussSeidelSmoother(
        StrictlyLowerTriangular(A, D),
        StrictlyUpperTriangular(A, D),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        config.iter
    )
end

function LinearAlgebra.ldiv!(x, s::SymmetricGaussSeidelSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # x ← U \ (-L * x + b)
        gauss_seidel_multiply!(-one(T), s.sU, x, one(T), b, x)
        forward_sub!(s.L, x)

        # x ← L \ (-U * x + b)
        gauss_seidel_multiply!(-one(T), s.sL, x, one(T), b, x)
        backward_sub!(s.U, x)
    end
    return nothing
end

struct ForwardSORSmoother{Tv, Ti, vecT, numT}
    U::StrictlyUpperTriangular{Tv, Ti}
    L::FastLowerTriangular{Tv, Ti}
    tmp::vecT
    iter::Int
    ω::numT
end

function setup_smoother(config::SOR{<:ForwardSweep}, A, x)
    D = DiagonalIndices(A)
    ForwardSORSmoother(StrictlyUpperTriangular(A, D), FastLowerTriangular(A, D), similar(x), config.iter, config.ω)
end

function LinearAlgebra.ldiv!(x, s::ForwardSORSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # tmp = b - U * x
        gauss_seidel_multiply!(-one(T), s.U, x, one(T), b, s.tmp)

        # tmp = ω * inv(L) * tmp + (1 - ω) * x
        forward_sub!(s.ω, s.L, s.tmp, one(T) - s.ω, x)

        copy!(x, s.tmp)
    end
    return nothing
end

struct BackwardSORSmoother{Tv, Ti, vecT, numT}
    U::FastUpperTriangular{Tv, Ti}
    L::StrictlyLowerTriangular{Tv, Ti}
    tmp::vecT
    iter::Int
    ω::numT
end

function setup_smoother(config::SOR{<:BackwardSweep}, A, x)
    D = DiagonalIndices(A)
    BackwardSORSmoother(FastUpperTriangular(A, D), StrictlyLowerTriangular(A, D), similar(x), config.iter, config.ω)
end

function LinearAlgebra.ldiv!(x, s::BackwardSORSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # tmp = b - U * x
        gauss_seidel_multiply!(-one(T), s.L, x, one(T), b, s.tmp)

        # tmp = ω * inv(L) * tmp + (1 - ω) * x
        backward_sub!(s.ω, s.U, s.tmp, one(T) - s.ω, x)

        copy!(x, s.tmp)
    end
    return nothing
end

struct SymmetricSORSmoother{Tv, Ti, vecT, numT}
    sL::StrictlyLowerTriangular{Tv, Ti}
    sU::StrictlyUpperTriangular{Tv, Ti}
    L::FastLowerTriangular{Tv, Ti}
    U::FastUpperTriangular{Tv, Ti}
    tmp::vecT
    iter::Int
    ω::numT
end

function setup_smoother(config::SOR{<:SymmetricSweep}, A, x)
    D = DiagonalIndices(A)
    SymmetricSORSmoother(
        StrictlyLowerTriangular(A, D),
        StrictlyUpperTriangular(A, D),
        FastLowerTriangular(A, D),
        FastUpperTriangular(A, D),
        zeros(size(A, 2)),
        config.iter,
        config.ω,
    )
end

function LinearAlgebra.ldiv!(x, s::SymmetricSORSmoother, b)
    T = eltype(x)
    for i in 1:s.iter
        # tmp = b - U * x
        gauss_seidel_multiply!(-one(T), s.sU, x, one(T), b, s.tmp)

        # tmp = ω * inv(L) * tmp + (1 - ω) * x
        forward_sub!(s.ω, s.L, s.tmp, one(T) - s.ω, x)

        # tmp = b - U * x
        gauss_seidel_multiply!(-one(T), s.sL, x, one(T), b, s.tmp)

        # tmp = ω * inv(L) * tmp + (1 - ω) * x
        backward_sub!(s.ω, s.U, s.tmp, one(T) - s.ω, x)

        copy!(x, s.tmp)
    end
    return nothing
end
