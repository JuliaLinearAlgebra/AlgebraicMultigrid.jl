struct Level{Ti,Tv}
    A::SparseMatrixCSC{Ti,Tv}
    P::SparseMatrixCSC{Ti,Tv}
    R::SparseMatrixCSC{Ti,Tv}
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
    (sum(nnz(level.A) for level in ml.levels) + nnz(ml.final_A)) / nnz(ml.levels[1].A)
end

function grid_complexity(ml::MultiLevel)
    (sum(size(level.A, 1) for level in ml.levels) + size(ml.final_A, 1)) / size(ml.levels[1].A, 1)
end

abstract type Cycle end
struct V <: Cycle
end

function solve{T}(ml::MultiLevel, b::Vector{T},
                                    maxiter = 100,
                                    cycle = V(),
                                    tol = 1e-5)
    x = zeros(T, size(b))
    residuals = Vector{T}()
    A = ml.levels[1].A
    normb = norm(b)
    if normb != 0
        tol *= normb
    end
    push!(residuals, T(norm(b - A*x)))

    lvl = 1
    while length(residuals) <= maxiter && residuals[end] > tol
        if length(ml.levels) == 1
            x = coarse_solver(ml.coarse_solver, A, b)
        else
            x = __solve(cycle, ml, x, b, lvl)
        end
        push!(residuals, T(norm(b - A * x)))
    end
    x
end
function __solve{T}(v::V, ml, x::Vector{T}, b::Vector{T}, lvl)

    A = ml.levels[lvl].A
    presmoother!(ml.presmoother, A, x, b)

    res = b - A * x
    coarse_b = ml.levels[lvl].R * res
    coarse_x = zeros(T, size(coarse_b))

    if lvl == length(ml.levels)
        coarse_x = coarse_solver(ml.coarse_solver, ml.final_A, coarse_b)
    else
        coarse_x = __solve(v, ml, coarse_x, coarse_b, lvl + 1)
    end

    x .+= ml.levels[lvl].P * coarse_x

    postsmoother!(ml.postsmoother, A, x, b)

    x
end

coarse_solver{Tv,Ti}(::Pinv, A::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv}) =
                                                        pinv(full(A)) * b
