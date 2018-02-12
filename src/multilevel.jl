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
struct Pinv <: CoarseSolver
end

MultiLevel{Ti,Tv}(l::Vector{Level{Ti,Tv}}, A::SparseMatrixCSC{Ti,Tv}, presmoother, postsmoother) =
    MultiLevel(l, A, Pinv(), presmoother, postsmoother)
Base.length(ml) = length(ml.levels) + 1

function Base.show(io::IO, ml::MultiLevel)
    op = operator_complexity(ml)
    g = grid_complexity(ml)
    c = ml.coarse_solver
    total_nnz = sum(nnz(level.A) for level in ml.levels) + nnz(ml.final_A)
    lstr = ""
    for (i, level) in enumerate(ml.levels)
        lstr = lstr *
            @sprintf "   %2d   %10d   %10d [%5.2f%%]\n" i size(level.A, 1) nnz(level.A) (100 * nnz(level.A) / total_nnz)
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
    (sum(nnz(level.A) for level in ml.levels) +
            nnz(ml.final_A)) / nnz(ml.levels[1].A)
end

function grid_complexity(ml::MultiLevel)
    (sum(size(level.A, 1) for level in ml.levels) +
            size(ml.final_A, 1)) / size(ml.levels[1].A, 1)
end

abstract type Cycle end
struct V <: Cycle
end

function solve{T}(ml::MultiLevel, b::Vector{T},
                                    maxiter = 100,
                                    cycle = V(),
                                    tol = 1e-5;
                                    verbose = false,
                                    log = false)
                                        
    A = length(ml) == 1 ? ml.final_A : ml.levels[1].A                                   
    V = promote_type(eltype(A), eltype(b))
    x = zeros(V, size(b))
    tol = eltype(b)(tol)
    residuals = Vector{V}()
    normb = norm(b)
    if normb != 0
        tol *= normb
    end
    push!(residuals, normb)

    lvl = 1
    while length(residuals) <= maxiter && residuals[end] > tol
        if length(ml) == 1
            x = coarse_solver(ml.coarse_solver, A, b)
        else
            x = __solve(cycle, ml, x, b, lvl)
        end
        push!(residuals, T(norm(b - A * x)))
    end

    @show residuals
    if log
        return x, residuals
    else
        return x
    end
end
function __solve(v::V, ml, x, b, lvl)

    A = ml.levels[lvl].A
    presmoother!(ml.presmoother, A, x, b)

    res = b - A * x
    coarse_b = ml.levels[lvl].R * res
    coarse_x = zeros(eltype(coarse_b), size(coarse_b))

    if lvl == length(ml.levels)
        coarse_x = coarse_solver(ml.coarse_solver, ml.final_A, coarse_b)
    else
        coarse_x = __solve(v, ml, coarse_x, coarse_b, lvl + 1)
    end

    x .+= ml.levels[lvl].P * coarse_x

    postsmoother!(ml.postsmoother, A, x, b)

    x
end

coarse_solver(::Pinv, A, b) = pinv(full(A)) * b
