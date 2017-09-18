module AMG

include("strength.jl")
export strength_of_connection, Classical

include("splitting.jl")
export split_nodes, RS

include("gallery.jl")
export poisson

include("classical.jl")

end # module
