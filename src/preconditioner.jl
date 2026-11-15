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
    # Return `x` rather than rely on inference narrowing `_solve!`'s `log`-dependent return type.
    return x
end
mul!(b, p::Preconditioner, x) = mul!(b, length(p.ml) == 1 ? p.ml.final_A : p.ml.levels[1].A, x)

function \(p::Preconditioner, b)
    ldiv!(similar(b), p, b)
end
