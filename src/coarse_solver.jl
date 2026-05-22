
abstract type CoarseSolver end

"""
    Pinv{T} <: CoarseSolver

Moore-Penrose pseudo inverse coarse solver. Calls `pinv`
"""
struct Pinv{T} <: CoarseSolver
    pinvA::Matrix{T}
    Pinv{T}(A) where T = new{T}(pinv(Matrix(A)))
end
Pinv(A) = Pinv{eltype(A)}(A)
Base.show(io::IO, p::Pinv) = print(io, "Pinv")

(p::Pinv)(x, b) = mul!(x, p.pinvA, b)

# This one is used internally.
"""
    LinearSolveWrapperInternal <: CoarseSolver

Helper to allow the usage of LinearSolve.jl solvers for the coarse-level solve. Constructed via `LinearSolveWrapper`.
"""
struct LinearSolveWrapperInternal{LC <: LinearSolve.LinearCache} <: CoarseSolver
    linsolve::LC
    function LinearSolveWrapperInternal(A, alg::LinearSolve.SciMLLinearSolveAlgorithm)
        rhs_tmp = zeros(eltype(A), size(A,1))
        u_tmp   = zeros(eltype(A), size(A,2))
        linprob = LinearProblem(A, rhs_tmp; u0 = u_tmp, alias_A = false, alias_b = false)
        linsolve = init(linprob, alg)
        new{typeof(linsolve)}(linsolve)
    end
end

function (p::LinearSolveWrapperInternal{LC})(x, b) where {LC <: LinearSolve.LinearCache}
    for i ∈ 1:size(b, 2)
        # Update right hand side
        p.linsolve.b = b[:, i]
        # Solve for x and update
        x[:, i] = solve!(p.linsolve).u
    end
end

function Base.show(io::IO, ml::LinearSolveWrapperInternal)
    print(io, ml.linsolve.alg)
end


# This one simplifies passing of LinearSolve.jl algorithms into AlgebraicMultigrid.jl as coarse solvers.
"""
    LinearSolveWrapper <: CoarseSolver

Helper to allow the usage of LinearSolve.jl solvers for the coarse-level solve.
"""
struct LinearSolveWrapper{A <: LinearSolve.SciMLLinearSolveAlgorithm} <: CoarseSolver
    alg::A
end
(p::LinearSolveWrapper)(A::AbstractMatrix) = LinearSolveWrapperInternal(A, p.alg)


"""
    QRSolver{F} <: CoarseSolver

Coarse solver using Julia's built-in factorizations via `qr()`.
"""
struct QRSolver{F} <: CoarseSolver
    factorization::F
    function QRSolver(A)
        fact = qr(A)
        new{typeof(fact)}(fact)
    end
end
Base.show(io::IO, p::QRSolver) = print(io, "QRSolver")

function (solver::QRSolver)(x, b)
    # Handle multiple RHS efficiently
    for i ∈ 1:size(b, 2)
        # Use backslash - Julia's factorizations are optimized for this
        x[:, i] = solver.factorization \ b[:, i]
    end
end

# Guess the best coarse solver based on the matrix type
_default_coarse_solver(A) = QRSolver
