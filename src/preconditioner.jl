struct Preconditioner{ML<:MultiLevel,C<:Cycle}
    ml::ML
    init::Symbol
    cycle::C
end
Preconditioner(ml, cycle::Cycle) = Preconditioner(ml, :zero, cycle)

aspreconditioner(ml::MultiLevel, cycle::Cycle = V()) = Preconditioner(ml,cycle)

import LinearAlgebra: \, *, ldiv!, mul!
ldiv!(p::Preconditioner, b) = copyto!(b, p \ b)
function ldiv!(x, p::Preconditioner, b)
    if p.init == :zero
        x .= 0
    else
        x .= b
    end
    _solve!(x, p.ml, b, p.cycle, maxiter = 1, calculate_residual = false)
    # Return `x` explicitly: `_solve!` may also return `(x, residuals)` (with
    # `log = true`), and whether inference sees through that depends on
    # constant-propagation heuristics.
    return x
end
mul!(b, p::Preconditioner, x) = mul!(b, p.ml.levels[1].A, x)

function \(p::Preconditioner, b)
    ldiv!(similar(b), p, b)
end
