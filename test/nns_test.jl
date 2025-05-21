using AlgebraicMultigrid
using Test
import AlgebraicMultigrid as AMG
using SparseArrays, LinearAlgebra

## Test QR factorization
@testset "fit_candidates Unit tests" begin
    cases = Vector{Tuple{SparseMatrixCSC{Float64,Int}, Matrix{Float64}}}()

    # 1. Aggregates include all dofs, one candidate
    push!(cases, (
        sparse([1,2,3,4,5], [1,1,1,2,2], ones(5), 5, 2),
        ones(5, 1)
    ))
    push!(cases, (
        sparse([1,2,3,4,5], [2,2,1,1,1], ones(5), 5, 2),
        ones(5, 1)
    ))
    push!(cases, (
        sparse([1,2,3,4,5,6,7,8,9],
               repeat([1,2,3], inner=3),
               ones(9), 9, 3),
        ones(9, 1)
    ))
    push!(cases, (
        sparse([1,2,3,4,5,6,7,8,9], [3,2,1,1,2,3,2,1,3],
               ones(9), 9, 3),
        reshape(Float64.(0:8), 9, 1)
    ))

    # 2. Aggregates include all dofs, two candidates
    push!(cases, (
        sparse([1,2,3,4], [1,1,2,2], ones(4), 4, 2),
        hcat(ones(4), collect(0:3))
    ))
    push!(cases, (
        sparse([1,2,3,4,5,6,7,8,9], repeat([1,2,3], inner=3), ones(9), 9, 3),
        hcat(ones(9), collect(0:8))
    ))
    push!(cases, (
        sparse([1,2,3,4,5,6,7,8,9], [1,1,2,2,3,3,4,4,4], ones(9), 9, 4),
        hcat(ones(9), collect(0:8))
    ))

    # 3. Small norms
    push!(cases, (
        sparse([1,2,3,4], [1,1,2,2], ones(4), 4, 2),
        hcat(ones(4), 1e-20 .* collect(0:3))
    ))
    push!(cases, (
        sparse([1,2,3,4], [1,1,2,2], ones(4), 4, 2),
        1e-20 .* hcat(ones(4), collect(0:3))
    ))

    # Run tests
    for (AggOp, fine) in cases
        # mask dofs not in aggregation
        for i in 1:size(AggOp,1)
            if nnz(AggOp[i, :]) == 0
                fine[i, :] .= 0.0
            end
        end
        Q, R = fit_candidates(AggOp |> adjoint, fine)
        # fit exactly and via projection
        @test fine ≈ Q * R  
        @test fine ≈ Q * (Q' * fine) 
    end
end

## Test Convergance of AMG for linear elasticity
# Example test: https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/linear_elasticity/
using Ferrite, FerriteGmsh, SparseArrays

using Downloads: download
logo_mesh = "logo.geo"
asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

grid = togrid(logo_mesh);

addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6);

dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim; # vector valued interpolation

qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
qr_face = FacetQuadratureRule{RefTriangle}(1);

cellvalues = CellValues(qr, ip)
facetvalues = FacetValues(qr_face, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
close!(ch);

traction(x) = Vec(0.0, 20.0e3 * x[1]);

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    # Create a temporary array for the facet's local contributions to the external force vector
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        # Update the facetvalues to the correct facet number
        reinit!(facetvalues, facet)
        # Reset the temporary array for the next facet
        fill!(fe_ext, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(facetvalues, qp, cell_coordinates)
            tₚ = prescribed_traction(x)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end

Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]

Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus

C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2}));

function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, C)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        assemble_cell!(ke, cellvalues, C)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

function create_nns(dh)
    Ndof = ndofs(dh)
    grid = dh.grid
    B = zeros(Float64, Ndof, 3)
    B[1:2:end,1] .= 1 # x - translation
    B[2:2:end,2] .= 1 # y - translation

    # in-plane rotation (x,y) → (-y,x)
    coords = reduce(hcat,grid.nodes .|> (n -> n.x |> collect))' # convert nodes to 2d array
    y = coords[:,2]
    x = coords[:,1]
    B[1:2:end,3] .= -y
    B[2:2:end,3] .= x
    return B
end

A = allocate_matrix(dh)
assemble_global!(A, dh, cellvalues, C);

b = zeros(ndofs(dh))
B  = create_nns(dh)
assemble_external_forces!(b, dh, getfacetset(grid, "top"), facetvalues, traction);

apply!(A, b, ch)
x = A \ b;

x_amg,residuals = solve(A, b, SmoothedAggregationAMG(B);log=true,reltol = 1e-10);

ml = smoothed_aggregation(A)
@test A * x_amg ≈ b
