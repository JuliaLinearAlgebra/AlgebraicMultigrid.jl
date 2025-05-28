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
        # load linear elasticity test data
        @load "lin_elastic_2d.jld2" A b B
        A = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval)

        x_nns, residuals_nns = solve(A, b, SmoothedAggregationAMG(B); log=true, reltol=1e-10)
        x_wonns, residuals_wonns = solve(A, b, SmoothedAggregationAMG(); log=true, reltol=1e-10)

        ml = smoothed_aggregation(A, B)
        @show ml

        println("No NNS: final residual at iteration ", length(residuals_wonns), ": ", residuals_wonns[end])
        println("With NNS: final residual at iteration ", length(residuals_nns), ": ", residuals_nns[end])


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

        println("No NNS: final residual at iteration ", length(residuals_wonns), ": ", residuals_wonns[end])
        println("With NNS: final residual at iteration ", length(residuals_nns), ": ", residuals_nns[end])


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
