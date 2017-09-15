module AMG

include("strength.jl")
export classical

include("mg.jl")
export RS

include("gallery.jl")
export poisson

end # module
