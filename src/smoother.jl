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
end
GaussSeidel() = GaussSeidel(SymmetricSweep())

presmoother!(s, A, x, b) = smoother!(s, s.sweep, A, x, b)
postsmoother!(s, A, x, b) = smoother!(s, s.sweep, A, x, b)

smoother!(s::GaussSeidel, ::ForwardSweep, A, x, b) =
                    gs!(A, b, x, 1, 1, size(A, 1))
                    # gauss_seidel!(x, A, b, maxiter = 1)
function smoother!(s::GaussSeidel, ::SymmetricSweep, A, x, b)
    smoother!(s, ForwardSweep(), A, x, b)
    smoother!(s, BackwardSweep(), A, x, b)
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
