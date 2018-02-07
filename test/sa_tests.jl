import AMG: scale_cols_by_largest_entry!, strength_of_connection, 
            SymmetricStrength, poisson
function symmetric_soc(A::SparseMatrixCSC{T,V}, θ) where {T,V}
    D = abs.(diag(A))
    i,j,v = findnz(A)
    mask = i .!= j
    DD = D[i] .* D[j] 
    mask = mask .& (abs.(v.^2) .>= (θ * θ * DD))

    i = i[mask]
    j = j[mask]
    v = v[mask]

    S = sparse(i,j,v, size(A)...) + spdiagm(D)

    scale_cols_by_largest_entry!(S)

    for i = 1:size(S.nzval,1)
        S.nzval[i] = abs(S.nzval[i])
    end

    S
end

# Set up tests
function test_symmetric_soc()

    cases = generate_matrices()

    for matrix in cases
        for θ in (0.0, 0.1, 0.5, 1., 10.)
            ref_matrix = symmetric_soc(matrix, θ)
            calc_matrix = strength_of_connection(SymmetricStrength(θ), matrix)

            @test sum(abs2, ref_matrix - calc_matrix) < 1e-6
        end
    end
end

function generate_matrices()
    
    cases = []

    # Random matrices
    srand(0)
    for T in (Float32, Float64)
        
        for s in [2, 3, 5]
            push!(cases, sprand(T, s, s, 1.))
        end

        for s in [2, 3, 5, 7, 10, 11, 19]
            push!(cases, poisson(T, s))
        end
    end

    cases
end
