"""
    SmoothedAggregationPreconBuilder(;blocksize=1, kwargs...)

Return callable object constructing a left smoothed aggregation algebraic multigrid preconditioner
to be used with the `precs` API of LinearSolve.
"""
struct SmoothedAggregationPreconBuilder{Tk,TB<:Union{Nothing,<:AbstractArray}}
    blocksize::Int
    B::TB # near null space basis
    kwargs::Tk
end

function SmoothedAggregationPreconBuilder(; blocksize = 1,B = nothing, kwargs...)
    return SmoothedAggregationPreconBuilder(blocksize,B, kwargs)
end

function (b::SmoothedAggregationPreconBuilder)(A::AbstractSparseMatrixCSC, p)
    return (aspreconditioner(smoothed_aggregation(SparseMatrixCSC(A),b.B, Val{b.blocksize}; b.kwargs...)), I)
end


"""
   RugeStubenPreconBuilder(;blocksize=1, kwargs...)

Return callable object constructing a  left algebraic multigrid preconditioner after Ruge & Stüben
to be used with the `precs` API of LinearSolve.
"""
struct RugeStubenPreconBuilder{Tk}
    blocksize::Int
    kwargs::Tk
end

function RugeStubenPreconBuilder(; blocksize = 1, kwargs...)
    return RugeStubenPreconBuilder(blocksize, kwargs)
end

function (b::RugeStubenPreconBuilder)(A::AbstractSparseMatrixCSC, p)
    return (aspreconditioner(ruge_stuben(SparseMatrixCSC(A), Val{b.blocksize}; b.kwargs...)), I)
end
