module AMG

include("strength.jl")
export classical

include("splitting.jl")
export RS

include("gallery.jl")
export poisson

end # module
