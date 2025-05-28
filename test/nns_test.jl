using AlgebraicMultigrid
import AlgebraicMultigrid as AMG
using Test
using SparseArrays, LinearAlgebra
using Ferrite, FerriteGmsh, SparseArrays
using Downloads: download

## Test QR factorization
@testset "fit_candidates unit test cases" begin
    cases = Vector{Tuple{SparseMatrixCSC{Float64,Int},Matrix{Float64}}}()

    # 1. Aggregates include all dofs, one candidate
    push!(cases, (
        sparse([1, 2, 3, 4, 5], [1, 1, 1, 2, 2], ones(5), 5, 2),
        ones(5, 1)
    ))
    push!(cases, (
        sparse([1, 2, 3, 4, 5], [2, 2, 1, 1, 1], ones(5), 5, 2),
        ones(5, 1)
    ))
    push!(cases, (
        sparse([1, 2, 3, 4, 5, 6, 7, 8, 9],
            repeat([1, 2, 3], inner=3),
            ones(9), 9, 3),
        ones(9, 1)
    ))
    push!(cases, (
        sparse([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 2, 1, 1, 2, 3, 2, 1, 3],
            ones(9), 9, 3),
        reshape(Float64.(0:8), 9, 1)
    ))

    # 2. Aggregates include all dofs, two candidates
    push!(cases, (
        sparse([1, 2, 3, 4], [1, 1, 2, 2], ones(4), 4, 2),
        hcat(ones(4), collect(0:3))
    ))
    push!(cases, (
        sparse([1, 2, 3, 4, 5, 6, 7, 8, 9], repeat([1, 2, 3], inner=3), ones(9), 9, 3),
        hcat(ones(9), collect(0:8))
    ))
    push!(cases, (
        sparse([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 2, 2, 3, 3, 4, 4, 4], ones(9), 9, 4),
        hcat(ones(9), collect(0:8))
    ))

    # 3. Small norms
    push!(cases, (
        sparse([1, 2, 3, 4], [1, 1, 2, 2], ones(4), 4, 2),
        hcat(ones(4), 1e-20 .* collect(0:3))
    ))
    push!(cases, (
        sparse([1, 2, 3, 4], [1, 1, 2, 2], ones(4), 4, 2),
        1e-20 .* hcat(ones(4), collect(0:3))
    ))

    # Run tests
    for (AggOp, fine) in cases
        # mask dofs not in aggregation
        for i in 1:size(AggOp, 1)
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

## Test Convergance of AMG for linear elasticity & bending beam
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
    B[1:2:end, 1] .= 1 # x - translation
    B[2:2:end, 2] .= 1 # y - translation

    # in-plane rotation (x,y) → (-y,x)
    coords = reduce(hcat, grid.nodes .|> (n -> n.x |> collect))' # convert nodes to 2d array
    y = coords[:, 2]
    x = coords[:, 1]
    B[1:2:end, 3] .= -y
    B[2:2:end, 3] .= x
    return B
end

function linear_elasticity_2d()
    # Example test: https://ferrite-fem.github.io/Ferrite.jl/stable/tutorials/linear_elasticity/
    logo_mesh = "logo.geo"
    asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
    isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

    grid = togrid(logo_mesh)
    addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
    addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
    addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6)

    dim = 2
    order = 1 # linear interpolation
    ip = Lagrange{RefTriangle,order}()^dim # vector valued interpolation

    qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
    qr_face = FacetQuadratureRule{RefTriangle}(1)

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
    close!(ch)

    traction(x) = Vec(0.0, 20.0e3 * x[1])
    Emod = 200.0e3 # Young's modulus [MPa]
    ν = 0.3        # Poisson's ratio [-]

    Gmod = Emod / (2(1 + ν))  # Shear modulus
    Kmod = Emod / (3(1 - 2ν)) # Bulk modulus

    C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2,2}))
    A = allocate_matrix(dh)
    assemble_global!(A, dh, cellvalues, C)

    b = zeros(ndofs(dh))
    B = create_nns(dh)
    assemble_external_forces!(b, dh, getfacetset(grid, "top"), facetvalues, traction)

    apply!(A, b, ch)

    return A, b, B # return the assembled matrix, force vector, and NNS matrix
end


## Helper functions for cantilever frame beam ##

# Element stiffness matrix
function frame_element_stiffness(EA, EI, le)
    l2 = le^2
    l3 = le^3
    Ke = zeros(6, 6)

    # Axial (u-u)
    Ke[1, 1] = EA / le
    Ke[1, 4] = -EA / le
    Ke[4, 1] = -EA / le
    Ke[4, 4] = EA / le

    # Bending (w, theta)
    Kb = EI / l3 * [
        12.0 6*le -12.0 6*le;
        6*le 4*l2 -6*le 2*l2;
        -12.0 -6*le 12.0 -6*le;
        6*le 2*l2 -6*le 4*l2
    ]
    idx = [2, 3, 5, 6]
    for i in 1:4, j in 1:4
        Ke[idx[i], idx[j]] = Kb[i, j]
    end
    return Ke
end


function create_nns_frame(x_coords::Vector{Float64}, dofmap::Vector{Int})
    # NNS matrix (3 columns)
    # i = 1 (u) : 1 0 -y
    # i = 2 (v) : 0 1 x
    # i = 3 (θ) : 0 0 1
    N = length(dofmap)
    B = zeros(N, 3)  # 3 rigid body modes: x-translation, y-translation, rotation

    for (i, dof) in enumerate(dofmap)
        node = div(dof - 1, 3) + 1
        offset = mod(dof - 1, 3)
        x = x_coords[node]
        y = 0.0  # 1D beam along x

        if offset == 0      # u (x-translation)
            B[i, 1] = 1.0              # translation in x
            B[i, 3] = -y               # rotation effect on u (−y)
        elseif offset == 1  # v (y-translation)
            B[i, 2] = 1.0              # translation in y
            B[i, 3] = x                # rotation effect on v (+x)
        elseif offset == 2  # θ (rotation DOF)
            B[i, 3] = 1.0              # direct contribution to rigid rotation
        end
    end

    return B
end

function cantilever_beam(P, E, A, I, L, n_elem)
    le = L / n_elem
    n_nodes = n_elem + 1
    dofs_per_node = 3  # u, w, theta
    n_dofs = n_nodes * dofs_per_node

    Ke = frame_element_stiffness(E * A, E * I, le)

    # Assemble global stiffness matrix
    row = Int[]
    col = Int[]
    val = Float64[]
    for e in 1:n_elem
        n1 = e
        n2 = e + 1
        dofmap = [
            3 * n1 - 2,  # u₁
            3 * n1 - 1,  # w₁
            3 * n1,      # θ₁
            3 * n2 - 2,  # u₂
            3 * n2 - 1,  # w₂
            3 * n2       # θ₂
        ]
        for i in 1:6, j in 1:6
            push!(row, dofmap[i])
            push!(col, dofmap[j])
            push!(val, Ke[i, j])
        end
    end
    A = sparse(row, col, val, n_dofs, n_dofs)
    # rhs
    b = zeros(n_dofs)
    force_dof = 3 * (n_nodes - 1) + 2  # w at last node
    b[force_dof] = P # Apply downward force at the last node
    # Boundary conditions: clamp left end
    fixed_dofs = [1, 2, 3]  # u₁, w₁, θ₁
    free_dofs = setdiff(1:n_dofs, fixed_dofs)
    A_free = A[free_dofs, free_dofs]
    b_free = b[free_dofs]

    # x-coordinates of nodes
    x_coords = [le * (i - 1) for i in 1:n_nodes]
    B = create_nns_frame(x_coords, free_dofs)

    return A_free, b_free, B
end

@testset "Mechanics test cases" begin
    @testset "Linear elasticity 2D" begin
        A, b, B = linear_elasticity_2d()

        x_nns, residuals_nns = solve(A, b, SmoothedAggregationAMG(B); log=true, reltol=1e-10)
        x_wonns, residuals_wonns = solve(A, b, SmoothedAggregationAMG(); log=true, reltol=1e-10)

        ml = smoothed_aggregation(A, B)
        @show ml

        println("No NNS: final residual at iteration ", length(residuals_wonns), ": ", residuals_nns[end])
        println("With NNS: final residual at iteration ", length(residuals_nns), ": ", residuals_wonns[end])


        #test QR factorization on linear elasticity
        aggregate = StandardAggregation()
        AggOp = aggregate(A)
        Q, R = fit_candidates(AggOp, B)
        # fit exactly and via projection
        @test B ≈ Q * R
        @test B ≈ Q * (Q' * B)

        # Check convergence
        @test !(A * x_wonns ≈ b)
        @test A * x_nns ≈ b

    end

    @testset "fit_candidates on cantilever frame beam" begin
        # Beam parameters
        P = -1000.0    # Applied force at the end of the beam
        n_elem = 10
        E = 210e9      # Young's modulus
        A = 1e-4       # Cross-section area (for axial)
        I = 1e-6       # Moment of inertia (for bending)
        L = 1.0        # Total length
        A, b, B = cantilever_beam(P, E, A, I, L, n_elem)
        # test solution
        # Analaytical solution for cantilever beam
        u = A \ b
        @test u[end-1] ≈ (P * L^3) / (3 * E * I) # vertical disp. at the end of the beam


        x_nns, residuals_nns = solve(A, b, SmoothedAggregationAMG(B); log=true, reltol=1e-10)
        x_wonns, residuals_wonns = solve(A, b, SmoothedAggregationAMG(); log=true, reltol=1e-10)

        println("No NNS: final residual at iteration ", length(residuals_wonns), ": ", residuals_nns[end])
        println("With NNS: final residual at iteration ", length(residuals_nns), ": ", residuals_wonns[end])


        # test QR factorization on bending beam
        # Aggregation
        aggregate = StandardAggregation()
        AggOp = aggregate(A)
        Q, R = fit_candidates(AggOp, B)
        @test B ≈ Q * R
        @test B ≈ Q * (Q' * B)

        # Check convergence
        @test !(A * x_wonns ≈ b)
        @test A * x_nns ≈ b
    end
end
