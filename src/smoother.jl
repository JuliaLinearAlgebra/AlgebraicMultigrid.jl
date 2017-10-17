abstract type Smoother end
abstract type Sweep end
struct SymmetricSweep <: Sweep
end
struct ForwardSweep <: Sweep
end
struct GaussSeidel{S} <: Smoother
    sweep::S
end
GaussSeidel() = GaussSeidel(ForwardSweep())

presmoother!(s, A, x, b) = smoother(s, s.sweep, A, x, b)
postsmoother!(s, A, x, b) = smoother(s, s.sweep, A, x, b)

smoother(s::GaussSeidel, ::ForwardSweep, A, x, b) =
                    gs!(A, b, x)
                        #gauss_seidel!(x, A, b, maxiter = 1)

function gs!{T,Ti}(A::SparseMatrixCSC{T,Ti}, b::Vector{T}, x::Vector{T})
    n = size(A, 1)
    z = zero(eltype(A))
    for i = 1:n
        # rsum = calc_weighted_sum(A, x)
        rsum = z
        diag = z
        for j in nzrange(A, i)
            row = A.rowval[j]
            val = A.nzval[j]
            if i == row
                diag = val
            else
                rsum += val * x[row]
            end
        end
        if diag != 0
            x[i] = (b[i] - rsum) / diag
        end
    end
end
