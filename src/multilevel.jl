struct Level{Ti,Tv}
    A::SparseMatrixCSC{Ti,Tv}
    P::SparseMatrixCSC{Ti,Tv}
    R::SparseMatrixCSC{Ti,Tv}
end

struct MultiLevel{L, S, Pre, Post}
    levels::Vector{L}
    coarse_solver::S
    presmoother::Pre
    postsmoother::Post
end

abstract type CoarseSolver end
struct Pinv <: CoarseSolver
end

MultiLevel(l::Vector{Level}, presmoother, postsmoother; coarse_solver = Pinv()) =
    MultiLevel(l, coarse_solver, presmoother, postsmoother)

function Base.show(io::IO, ml::MultiLevel)
    op = operator_complexity(ml.levels)
    g = grid_complexity(ml.levels)
    c = ml.coarse_solver
    total_nnz = sum(nnz(level.A) for level in ml.levels)
    lstr = ""
    for (i, level) in enumerate(ml.levels)
        lstr = lstr *
            @sprintf "   %2d   %10d   %10d [%5.2f%%]\n" i size(level.A, 1) nnz(level.A) (100 * nnz(level.A) / total_nnz)
    end
    str = """
    Multilevel Solver
    -----------------
    Operator Complexity: $op
    Grid Complexity: $g
    No. of Levels: $(size(ml.levels, 1))
    Coarse Solver: $c
    Level     Unknowns     NonZeros
    -----     --------     --------
    $lstr
    """
    print(io, str)
end

function operator_complexity(ml::Vector{Level})
    sum(nnz(level.A) for level in ml) / nnz(ml[1].A)
end

function grid_complexity(ml::Vector{Level})
    sum(size(level.A, 1) for level in ml) / size(ml[1].A, 1)
end

abstract type Cycle end
struct V <: Cycle
end

function solve{T}(ml::MultiLevel, b::Vector{T}; maxiter = 100,
                                                cycle = V(),
                                                tol = 1e-5)
    x = zeros(T, size(b))
    residuals = Vector{T}()
    A = ml.levels[1].A
    normb = norm(b)
    push!(residuals, norm(b - A*x))

    lvl = 1
    while length(residuals) <= maxiter && residuals[end] > tol
        if length(ml.levels) == 1
            x = coarse_solver(ml.coarse_solver, A, b)
        else
            x = __solve(cycle, ml, x, b, lvl, residuals)
        end
    end
    x
end
function __solve{T}(v::V, ml, x::Vector{T}, b::Vector{T}, lvl, residuals)

    @show lvl
    A = ml.levels[lvl].A
    presmoother!(ml.presmoother, A, x, b)

    res = b - A * x
    @show norm(res)
    push!(residuals, norm(res))

    coarse_b = ml.levels[lvl].R * res

    if lvl == length(ml.levels) - 1
        coarse_x = coarse_solver(ml.coarse_solver, ml.levels[end].A, coarse_b)
    else
        coarse_x = __solve(v, ml, coarse_x, coarse_b, lvl + 1)
    end

    x .+= ml.levels[lvl].P * coarse_x

    postsmoother!(ml.postsmoother, A, x, b)

    x
end

coarse_solver(::Pinv, A, b) = pinv(full(A)) * b
