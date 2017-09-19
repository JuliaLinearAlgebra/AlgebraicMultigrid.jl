struct Level{Ti,Tv}
    A::SparseMatrixCSC{Ti,Tv}
    P::SparseMatrixCSC{Ti,Tv}
    R::SparseMatrixCSC{Ti,Tv}
end

struct MultiLevel{L, S}
    levels::Vector{L}
    coarse_solver::S
end

abstract type CoarseSolver end
struct Pinv <: CoarseSolver
end
MultiLevel(l::Vector{Level}; coarse_solver = Pinv()) =
    MultiLevel(l, coarse_solver)

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
