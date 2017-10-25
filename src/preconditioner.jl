import Base: \, *, A_ldiv_B!, A_mul_B!

struct Preconditioner
    ml::MultiLevel
end

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)
first_matrix(p::Preconditioner) = ifelse(isempty(p.ml.levels),
                                    p.ml.final_A, p.ml.levels[1].A)

\(p::Preconditioner, b) = solve(p.ml, b, 1, V(), 1e-12)
*(p::Preconditioner, b) = first_matrix(p) * x

A_ldiv_B!(x, p::Preconditioner, b) = copy!(x, p \ b)
A_mul_B!(b, p::Preconditioner, x) = A_mul_B!(b, first_matrix(p), x)
