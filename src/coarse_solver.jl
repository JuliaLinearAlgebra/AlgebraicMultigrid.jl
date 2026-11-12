
abstract type CoarseSolver end

# Relative tolerance below which the coarse solvers treat singular values (`Pinv`) or
# diagonal entries of R (`QRSolver`) as zero. A singular problem, e.g. one with a pure
# Neumann boundary, produces a coarsest matrix that is only numerically singular
# (σmin/σmax ≈ 1e-16). With the factorizations' default tolerances, rounding decides
# whether that is recognized; when it is not, the coarse correction divides by ~1e-17
# and the cycle stalls.
coarse_rtol(::Type{T}) where {T} = sqrt(eps(real(float(T))))

"""
    Pinv{T} <: CoarseSolver

Moore-Penrose pseudo inverse coarse solver. Calls `pinv`, treating singular values
below `√eps · σmax` as zero so that numerically singular coarse matrices are handled
consistently.
"""
struct Pinv{T} <: CoarseSolver
    pinvA::Matrix{T}
    Pinv{T}(A) where T = new{T}(pinv(Matrix(A); rtol = coarse_rtol(T)))
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

Coarse solver using Julia's built-in factorizations via `qr()`. For sparse matrices
(SuiteSparse QR), columns whose norm drops below `√eps` times the largest column norm
during the factorization are treated as zero, so that numerically singular coarse
matrices are handled consistently.
"""
struct QRSolver{F} <: CoarseSolver
    factorization::F
    function QRSolver(A)
        fact = coarse_qr(A)
        new{typeof(fact)}(fact)
    end
end

coarse_qr(A) = qr(A)
function coarse_qr(A::SparseMatrixCSC{T}) where {T}
    maxcolnorm = maximum(j -> norm(view(nonzeros(A), nzrange(A, j))), 1:size(A, 2); init = zero(real(T)))
    return qr(A; tol = coarse_rtol(T) * maxcolnorm)
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
