struct Preconditioner{ML<:MultiLevel}
    ml::ML
    init::Symbol
end
Preconditioner(ml) = Preconditioner(ml, :zero)

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)

@static if VERSION < v"0.7-"
    import Base: \, *, A_ldiv_B!, A_mul_B!
    function A_ldiv_B!(x, p::Preconditioner, b)
        if p.init == :zero
            x .= 0
        else
            x .= b
        end
        solve!(x, p.ml, b, maxiter = 1, calculate_residual = false)
    end
    A_mul_B!(b, p::Preconditioner, x) = A_mul_B!(b, p.ml.levels[1].A, x)
else
    import Compat.LinearAlgebra: \, *, ldiv!, mul!
    ldiv!(p::Preconditioner, b) = copyto!(b, p \ b)
    function ldiv!(x, p::Preconditioner, b)
        if p.init == :zero
            x .= 0
        else
            x .= b
        end
        solve!(x, p.ml, b, maxiter = 1, calculate_residual = false)
    end
    mul!(b, p::Preconditioner, x) = mul!(b, p.ml.levels[1].A, x)
end

function \(p::Preconditioner, b)
    ldiv!(similar(b), p, b)
end
*(p::Preconditioner, b) = p.ml.levels[1].A * x
