import LinearAlgebra: \, *, ldiv!, mul!

struct Preconditioner
    ml::MultiLevel
end

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)

\(p::Preconditioner, b) = solve(p.ml, b, maxiter = 1, tol = 1e-12)
*(p::Preconditioner, b) = p.ml.levels[1].A * x

ldiv!(x, p::Preconditioner, b) = copyto!(x, p \ b)
mul!(b, p::Preconditioner, x) = mul!(b, p.ml.levels[1].A, x)
