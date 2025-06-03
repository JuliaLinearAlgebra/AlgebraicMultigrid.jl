struct Level{TA, TP, TR}
    A::TA
    P::TP
    R::TR
end

struct MultiLevel{S, Pre, Post, TA, TP, TR, TW}
    levels::Vector{Level{TA, TP, TR}}
    final_A::TA
    coarse_solver::S
    presmoother::Pre
    postsmoother::Post
    workspace::TW
end

struct MultiLevelWorkspace{TX, bs}
    coarse_xs::Vector{TX}
    coarse_bs::Vector{TX}
    res_vecs::Vector{TX}
end
function MultiLevelWorkspace(::Type{Val{bs}}, ::Type{T}) where {bs, T<:Number}
    if bs === 1
        TX = Vector{T}
    else
        TX = Matrix{T}
    end
    MultiLevelWorkspace{TX, bs}(TX[], TX[], TX[])
end
Base.eltype(w::MultiLevelWorkspace{TX}) where TX = eltype(TX)
blocksize(w::MultiLevelWorkspace{TX, bs}) where {TX, bs} = bs

function residual!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    if bs === 1
        push!(m.res_vecs, TX(undef, n))
    else
        push!(m.res_vecs, TX(undef, n, bs))
    end
end
function coarse_x!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    if bs === 1
        push!(m.coarse_xs, TX(undef, n))
    else
        push!(m.coarse_xs, TX(undef, n, bs))
    end
end
function coarse_b!(m::MultiLevelWorkspace{TX, bs}, n) where {TX, bs}
    if bs === 1
        push!(m.coarse_bs, TX(undef, n))
    else
        push!(m.coarse_bs, TX(undef, n, bs))
    end
end

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

# This one simplifies passing of LinearSolve.jl algorithms into AlgebraicMultigrid.jl as coarse solvers.
"""
    LinearSolveWrapper <: CoarseSolver

Helper to allow the usage of LinearSolve.jl solvers for the coarse-level solve.
"""
struct LinearSolveWrapper <: CoarseSolver
    alg::LinearSolve.SciMLLinearSolveAlgorithm
end
(p::LinearSolveWrapper)(A::AbstractMatrix) = LinearSolveWrapperInternal(A, p.alg)

Base.length(ml::MultiLevel) = length(ml.levels) + 1

function Base.show(io::IO, ml::MultiLevel)
    op = operator_complexity(ml)
    g = grid_complexity(ml)
    c = ml.coarse_solver
    total_nnz = nnz(ml.final_A)
    if !isempty(ml.levels)
        total_nnz += sum(nnz(level.A) for level in ml.levels)
    end
    lstr = ""
    if !isempty(ml.levels)
    for (i, level) in enumerate(ml.levels)
        lstr = lstr *
            @sprintf "   %2d   %10d   %10d [%5.2f%%]\n" i size(level.A, 1) nnz(level.A) (100 * nnz(level.A) / total_nnz)
    end
    end
    lstr = lstr *
        @sprintf "   %2d   %10d   %10d [%5.2f%%]" length(ml.levels) + 1 size(ml.final_A, 1) nnz(ml.final_A) (100 * nnz(ml.final_A) / total_nnz)

    opround = round(op, digits = 3)
    ground = round(g, digits = 3)

    str = """
    Multilevel Solver
    -----------------
    Operator Complexity: $opround
    Grid Complexity: $ground
    No. of Levels: $(length(ml))
    Coarse Solver: $c
    Level     Unknowns     NonZeros
    -----     --------     --------
    $lstr
    """
    print(io, str)
end

function operator_complexity(ml::MultiLevel)
    if !isempty(ml.levels)
        (sum(nnz(level.A) for level in ml.levels) +
                nnz(ml.final_A)) / nnz(ml.levels[1].A)
    else
        1.
    end
end

function grid_complexity(ml::MultiLevel)
    if !isempty(ml.levels)
        (sum(size(level.A, 1) for level in ml.levels) +
                size(ml.final_A, 1)) / size(ml.levels[1].A, 1)
    else
        1.
    end
end

abstract type Cycle end
struct V <: Cycle
end

struct W <: Cycle
end

struct F <: Cycle
end

"""
    _solve(ml::MultiLevel, b::AbstractArray, cycle, kwargs...)

Execute multigrid cycling.

Arguments
=========
* ml::MultiLevel - the multigrid hierarchy
* b::Vector - the right hand side
* cycle -  multigird cycle to execute at each iteration. Defaults to AMG.V()

Keyword Arguments
=================
* reltol::Float64 - relative tolerance criteria for convergence, the absolute tolerance will be `reltol * norm(b)`
* abstol::Float64 - absolute tolerance criteria for convergence
* maxiter::Int64 - maximum number of iterations to execute
* verbose::Bool - display residual at each iteration
* log::Bool - return vector of residuals along with solution

"""
function _solve(ml::MultiLevel, b::AbstractArray, args...; kwargs...)
    n = length(ml) == 1 ? size(ml.final_A, 1) : size(ml.levels[1].A, 1)
    V = promote_type(eltype(ml.workspace), eltype(b))
    x = zeros(V, size(b))
    return _solve!(x, ml, b, args...; kwargs...)
end
function _solve!(x, ml::MultiLevel, b::AbstractArray{T},
                                    cycle::Cycle = V();
                                    maxiter::Int = 100,
                                    abstol::Real = zero(real(eltype(b))),
                                    reltol::Real = sqrt(eps(real(eltype(b)))),
                                    verbose::Bool = false,
                                    log::Bool = false,
                                    calculate_residual = true, kwargs...) where {T}

    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A
    V = promote_type(eltype(A), eltype(b))
    log && (residuals = Vector{V}())
    normres = normb = norm(b)
    if normb != 0
        abstol = max(reltol * normb, abstol)
    end
    log && push!(residuals, normb)

    res = ml.workspace.res_vecs[1]
    itr = lvl = 1
    while itr <= maxiter && (!calculate_residual || normres > abstol)
        if length(ml) == 1
            ml.coarse_solver(x, b)
        else
            __solve!(x, ml, cycle, b, lvl)
        end
        if calculate_residual
            if verbose
                @printf "Norm of residual at iteration %6d is %.4e\n" itr normres
            end
            mul!(res, A, x)
            reshape(res, size(b)) .= b .- reshape(res, size(b))
            normres = norm(res)
            log && push!(residuals, normres)
        end
        itr += 1
    end

    # @show residuals
    log ? (x, residuals) : x
end

function __solve_next!(x, ml, cycle::V, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
end

function __solve_next!(x, ml, cycle::W, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
end

function __solve_next!(x, ml, cycle::F, b, lvl)
    __solve!(x, ml, cycle, b, lvl)
    __solve!(x, ml, V(), b, lvl)
end

function __solve!(x, ml, cycle::Cycle, b, lvl)
    A = ml.levels[lvl].A
    ml.presmoother(A, x, b)

    res = ml.workspace.res_vecs[lvl]
    mul!(res, A, x)
    reshape(res, size(b)) .= b .- reshape(res, size(b))

    coarse_b = ml.workspace.coarse_bs[lvl]
    mul!(coarse_b, ml.levels[lvl].R, res)

    coarse_x = ml.workspace.coarse_xs[lvl]
    coarse_x .= 0
    if lvl == length(ml.levels)
        ml.coarse_solver(coarse_x, coarse_b)
    else
        coarse_x = __solve_next!(coarse_x, ml, cycle, coarse_b, lvl + 1)
    end

    mul!(res, ml.levels[lvl].P, coarse_x)
    x .+= res

    ml.postsmoother(A, x, b)

    x
end

### CommonSolve.jl spec
struct AMGSolver{T}
    ml::MultiLevel
    b::Vector{T}
end

abstract type AMGAlg end

struct RugeStubenAMG  <: AMGAlg end

"""
    SmoothedAggregationAMG(B=nothing)

Smoothed Aggregation AMG (Algebraic Multigrid) algorithm configuration.

# Arguments
- `B::Union{AbstractArray, Nothing}`: The **near null space** in SA-AMG represents the low energy that cannot be attenuated by relaxtion, and thus needs to be perserved across the coarse grid. 

    - `B` can be:
        - a `Vector` (e.g., for scalar PDEs),
        - a `Matrix` (e.g., for vector PDEs or systems with multiple equations),
        - or `nothing`.

# Notes
If `B` is set to `nothing`, it will be internally defaulted to `B = ones(T, n)`, where `T = eltype(A)` and `n = size(A, 1)`.

# Examples

**Poisson equation (scalar PDE):**
```julia
n = size(A, 1)
B = ones(Float64, n)
amg = SmoothedAggregationAMG(B)
```

**Linear elasticity equation in 2d (vector PDE):**
```julia
n = size(A, 1)          # Ndof = 2 * number of nodes
B = zeros(Float64, n, 3) 
B[1:2:end,   :] = [1.0, 0.0, -y]
B[2:2:end, :] = [0.0, 1.0,  x]
amg = SmoothedAggregationAMG(B)
```
"""
struct SmoothedAggregationAMG  <: AMGAlg
    B::Union{<:AbstractArray,Nothing} # `B` can be `Vector`, `Matrix`, or `nothing`
    function SmoothedAggregationAMG(B::Union{AbstractArray,Nothing}=nothing)
        new(B)
    end
end

function solve(A::AbstractMatrix, b::Vector, s::AMGAlg, args...; kwargs...)
    solt = init(s, A, b, args...; kwargs...)
    solve!(solt, args...; kwargs...)
end
function init(::RugeStubenAMG, A, b, args...; kwargs...)
    AMGSolver(ruge_stuben(A; kwargs...), b)
end
function init(sa::SmoothedAggregationAMG, A, b; kwargs...)
    AMGSolver(smoothed_aggregation(A,sa.B; kwargs...), b)
end
function solve!(solt::AMGSolver, args...; kwargs...) 
    _solve(solt.ml, solt.b, args...; kwargs...)   
end
