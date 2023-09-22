module AlgebraicMultigrid

using Reexport
using LinearAlgebra
using LinearSolve
using SparseArrays, Printf
@reexport import CommonSolve: solve, solve!, init
using Reexport

using LinearAlgebra: rmul!

include("utils.jl")
export approximate_spectral_radius

include("strength.jl")
export Classical, SymmetricStrength

include("splitting.jl")
export RS

include("gallery.jl")
export poisson

include("smoother.jl")
export GaussSeidel, SymmetricSweep, ForwardSweep, BackwardSweep,
        JacobiProlongation

include("multilevel.jl")
export RugeStubenAMG, SmoothedAggregationAMG

include("classical.jl")
export ruge_stuben

include("aggregate.jl")
export StandardAggregation

include("aggregation.jl")
export fit_candidates, smoothed_aggregation

include("preconditioner.jl")
export aspreconditioner

end # module
