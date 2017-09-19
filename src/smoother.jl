abstract type Smoother end
abstract type Sweep end
struct SymmetricSweep <: Sweep
end
struct ForwardSweep <: Sweep
end
struct GaussSeidel{S} <: Smoother
    sweep::S
end
GaussSeidel(;sweep = ForwardSweep()) = GaussSeidel(sweep)

presmoother!(s, A, x, b) = smoother(s, s.sweep, A, x, b)
postsmoother!(s, A, x, b) = smoother(s, s.sweep, A, x, b)

smoother(s::GaussSeidel, ::ForwardSweep, A, x, b) =
                        gauss_seidel!(x, A, b, maxiter = 1)
