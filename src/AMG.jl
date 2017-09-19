module AMG

import IterativeSolvers: gauss_seidel!

include("strength.jl")
export strength_of_connection, Classical

include("splitting.jl")
export split_nodes, RS

include("gallery.jl")
export poisson

include("smoother.jl")

include("multilevel.jl")
export solve

include("classical.jl")
export ruge_stuben

end # module
