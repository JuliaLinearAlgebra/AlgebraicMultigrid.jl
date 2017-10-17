import Base: \, *, A_ldiv_B!, A_mul_B!

struct Preconditioner
    ml::MultiLevel
end

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)

\(p::Preconditioner, b) = p * b
*(p::Preconditioner, b) = solve(p.ml, b, 1, V(), 1e-12)

A_ldiv_B!(x, p::Preconditioner, b) = copy!(x, p \ b)
A_mul_B!(b, p::Preconditioner, x) = A_mul_B!(b, p.ml.levels[1].A, x)
