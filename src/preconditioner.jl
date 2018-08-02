import Compat.LinearAlgebra: \, *, ldiv!, mul!

struct Preconditioner
    ml::MultiLevel
end

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)

@static if VERSION < v"0.7-"
    import Base: \, *, A_ldiv_B!, A_mul_B!
    A_ldiv_B!(x, p::Preconditioner, b) = copyto!(x, p \ b)
    A_mul_B!(b, p::Preconditioner, x) = A_mul_B!(b, p.ml.levels[1].A, x)
else
    import Compat.LinearAlgebra: \, *, ldiv!, mul!
    ldiv!(p::Preconditioner, b) = copyto!(b, p \ b)
    ldiv!(x, p::Preconditioner, b) = copyto!(x, p \ b)
    mul!(b, p::Preconditioner, x) = mul!(b, p.ml.levels[1].A, x)
end

\(p::Preconditioner, b) = solve(p.ml, b, maxiter = 1, tol = 1e-12)
*(p::Preconditioner, b) = p.ml.levels[1].A * x
