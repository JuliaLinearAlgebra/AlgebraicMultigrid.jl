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

presmoother!(s, A, x, b) = smoother!(s, s.sweep, A, x, b)
postsmoother!(s, A, x, b) = smoother!(s, s.sweep, A, x, b)
relax!(s, A, x, b) = smoother!(s, s.sweep, A, x, b)

smoother!(s::GaussSeidel, ::ForwardSweep, A, x, b) =
                    gs!(A, b, x, 1, 1, size(A, 1))
                    # gauss_seidel!(x, A, b, maxiter = 1)
function smoother!(s::GaussSeidel, ::SymmetricSweep, A, x, b)
    for i in 1:s.iter
        smoother!(s, ForwardSweep(), A, x, b)
        smoother!(s, BackwardSweep(), A, x, b)
    end
end

smoother!(s::GaussSeidel, ::BackwardSweep, A, x, b) =
    gs!(A, b, x, size(A,1), -1, 1)


function gs!{T,Ti}(A::SparseMatrixCSC{T,Ti}, b::Vector{T}, x::Vector{T}, start, step, stop)
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

struct Jacobi{T} <: Smoother
    ω::T
end

function jacobi!(A, x, b, ω, start, step, stop)

    one = one(eltype(A))
    temp = similar(x)

    for i = start:step:stop
        temp[i] = x[i]
    end

    for i = start:step:stop
        rsum = zero(eltype(A))
        diag = zero(eltype(A))

        for j in nzrange(A, i)
            row = A.nzval[j]
            val = A.nzval[j]

            if row == i
                diag = val
            else
                rsum += val * temp[row]
            end
        end

        if diag != 0
            x[i] = (one - ω) * temp[i] + ω * ((b[i] - rsum) / diag)
        end
    end
end
