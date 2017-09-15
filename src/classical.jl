struct Solver{S,T,P,PS}
    strength::S
    CF::T
    presmoother::P
    postsmoother::PS
    max_levels::Int64
    max_coarse::Int64
end

function ruge_stuben(A::SparseMatrixCSC;
                strength = Classical(),
                CF = RS(),
                presmoother = GaussSiedel(),
                postsmoother = GaussSiedel(),
                max_levels = 10,
                max_coarse = 500)

        s = Solver(strength, CF, presmoother,
                    postsmoother, max_levels, max_levels)

        levels = [Level(A)]

        while length(levels) < max_levels && size(levels[end].A, 1)
            extend_heirarchy!(levels, strength, CF, A)
        end
end

function extend_heirarchy!(levels::Vector{Level}, strength, CF, A)

    S = strength_of_connection(strength, A)
    splitting = split_nodes(CF, S)
    P = direct_interpolation(A, S, splitting)
end
