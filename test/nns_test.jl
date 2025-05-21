using AlgebraicMultigrid
using Test
import AlgebraicMultigrid as AMG
using SparseArrays, LinearAlgebra

## Test QR factorization
# tolerance and common dimensions
n_fine = 9
n_agg  = 3
rows   = collect(1:n_fine)
agg_id = repeat(1:n_agg, inner=n_fine ÷ n_agg)

# Agg_input is n_agg×n_fine; we then transpose it
Agg = sparse(agg_id, rows, ones(n_fine), n_agg, n_fine)

for m in (1, 2, 3)
    # build B (n_fine×m) for m = 1, 2, 3
    B = m == 1 ? ones(n_fine,1) :
        m == 2 ? hcat(ones(n_fine), collect(1:n_fine)) :
                 hcat(ones(n_fine), collect(1:n_fine), collect(1:n_fine).^2)

    Qs, R = AMG.fit_candidates(Agg, B)

    @test size(Qs) == (n_fine, m * n_agg)
    @test size(R)  == (m * n_agg, m)

    for agg in 1:n_agg
        idx = findall(x->x==agg, agg_id)
        M   = B[idx, :]
        Qj  = Array(Qs[idx, (agg-1)*m+1 : agg*m])
        Rj  = R[(agg-1)*m+1 : agg*m, :]

        @test isapprox(M, Qj * Rj; atol=1e-8)
        @test isapprox(Qj' * Qj, I(m);   atol=1e-8)
        @test istriu(Rj)
    end
end

@testset "QR examples tests" begin
    # Example 1: four nodes, two aggregates, constant vector
    AggOp = sparse([1,2,3,4], [1,1,2,2], [1,1,1,1], 4, 2) |> adjoint
    B1 = ones(4,1)
    Q1, R1 = fit_candidates(AggOp, B1)
    Q1a = [0.70710678 0.0;
            0.70710678 0.0;
            0.0        0.70710678;
            0.0        0.70710678]
    R1a = [1.41421356;
           1.41421356]
    @test Q1 ≈ Q1a
    @test R1 ≈ R1a

    # Example 2: constant vector + linear function
    B2 = Float64.([1 0;
          1 1;
          1 2;
          1 3])
    Q2, R2 = fit_candidates(AggOp, B2)
    Q2a = [ 0.70710678 -0.70710678  0.0         0.0;
            0.70710678  0.70710678  0.0         0.0;
            0.0         0.0         0.70710678 -0.70710678;
            0.0         0.0         0.70710678  0.70710678 ]
    R2a = [1.41421356 0.70710678;
           0.0        0.70710678;
           1.41421356 3.53553391;
           0.0        0.70710678]
    @test Q2 ≈ Q2a
    @test R2 ≈ R2a

    # Example 3: aggregation excludes third node
    AggOp3 = sparse([1,2,4], [1,1,2], [1,1,1], 4, 2) |> adjoint
    B3 = ones(4,1)
    Q3, R3 = fit_candidates(AggOp3, B3)
    Q3a = [0.70710678 0.0;
           0.70710678 0.0;
           0.0        0.0;
           0.0        1.0]
    R3a = [1.41421356;
           1.0]
    @test Q3 ≈ Q3a
    @test R3 ≈ R3a
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

#ml = smoothed_aggregation(A)
@test A * x_amg ≈ b



## DEBUG

ml = smoothed_aggregation(A,B)
aggregate = StandardAggregation()
improve_candidates = GaussSeidel(iter=4)



strength = SymmetricStrength()
S , _= strength(A, false)
AggOp = aggregate(S)
b = zeros(size(A,1))
improve_candidates(A, B, b)
T, B = fit_candidates(AggOp, B)



using DelimitedFiles



# write with comma delimiter
#writedlm("B_nns.csv", B, ',')
B_py = readdlm("B_nns_1.csv", ',', Float64)
