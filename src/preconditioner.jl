struct Preconditioner{ML<:MultiLevel}
    ml::ML
    init::Symbol
end
Preconditioner(ml) = Preconditioner(ml, :zero)

aspreconditioner(ml::MultiLevel) = Preconditioner(ml)

import LinearAlgebra: \, *, ldiv!, mul!
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

function \(p::Preconditioner, b)
    ldiv!(similar(b), p, b)
end
*(p::Preconditioner, b) = p.ml.levels[1].A * x
