struct Level{T,V}
    A::SparseMatrixCSC{T,V}
    P::SparseMatrixCSC{T,V}
    R::SparseMatrixCSC{T,V}
end

struct MultiLevel{S, Pre, Post, Ti, Tv}
    levels::Vector{Level{Ti,Tv}}
    final_A::SparseMatrixCSC{Ti,Tv}
    coarse_solver::S
    presmoother::Pre
    postsmoother::Post
end

abstract type CoarseSolver end
struct Pinv{T} <: CoarseSolver
    pinvA::Matrix{T}
    Pinv(A) = new{eltype(A)}(pinv(Matrix(A)))
end
(p::Pinv)(x, b) = mul!(x, p.pinvA, b)

MultiLevel(l::Vector{Level{Ti,Tv}}, A::SparseMatrixCSC{Ti,Tv}, presmoother, postsmoother) where {Ti,Tv} =
    MultiLevel(l, A, Pinv(A), presmoother, postsmoother)
Base.length(ml) = length(ml.levels) + 1

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
    str = """
    Multilevel Solver
    -----------------
    Operator Complexity: $(round(op, 3))
    Grid Complexity: $(round(g, 3))
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

"""
    solve(ml::MultiLevel, b::AbstractVector, cycle, kwargs...)

Execute multigrid cycling.

Arguments
=========
* ml::MultiLevel - the multigrid hierarchy
* b::Vector - the right hand side
* cycle -  multigird cycle to execute at each iteration. Defaults to AMG.V()

Keyword Arguments
=================
* tol::Float64 - tolerance criteria for convergence
* maxiter::Int64 - maximum number of iterations to execute
* verbose::Bool - display residual at each iteration
* log::Bool - return vector of residuals along with solution

"""
function solve(ml::MultiLevel, b::AbstractVector{T},
                                    cycle::Cycle = V();
                                    maxiter::Int = 100,
                                    tol::Float64 = 1e-5,
                                    verbose::Bool = false,
                                    log::Bool = false) where {T}
                                        
    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A                                   
    V = promote_type(eltype(A), eltype(b))
    x = zeros(V, size(b))
    tol = eltype(b)(tol)
    log && (residuals = Vector{V}())
    normres = normb = norm(b)
    if normb != 0
        tol *= normb
    end
    log && push!(residuals, normb)

    itr = lvl = 1
    while itr <= maxiter && normres > tol
        if length(ml) == 1
            ml.coarse_solver(x, b)
        else
            __solve!(x, ml, cycle, b, lvl)
        end
        normres = norm(b - A * x)
        log && push!(residuals, normres)
        itr += 1
    end

    # @show residuals
    log ? (x, residuals) : x
end
function __solve!(x, ml, v::V, b, lvl)

    A = ml.levels[lvl].A
    ml.presmoother(A, x, b)

    res = b - A * x
    coarse_b = ml.levels[lvl].R * res
    coarse_x = zeros(eltype(coarse_b), size(coarse_b))

    if lvl == length(ml.levels)
        ml.coarse_solver(coarse_x, coarse_b)
    else
        coarse_x = __solve!(coarse_x, ml, v, coarse_b, lvl + 1)
    end

    x .+= ml.levels[lvl].P * coarse_x

    ml.postsmoother(A, x, b)

    x
end
