struct LinearSolveAlgebraicMultigrid_JL{ML} <: SciMLBase.AbstractLinearAlgorithm
    multilevel::ML
end

function init_cacheval(alg::LinearSolveAlgebraicMultigrid_JL, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    lu!(convert(AbstractMatrix, A))
end
