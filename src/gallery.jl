poisson(T, n) = sparse(Tridiagonal(fill(T(-1), n-1),
                        fill(T(2), n), fill(T(-1), n-1)))
poisson(n) = poisson(Float64, n)

function stencil_grid(T,stencil,sz)
    # upper-bound for storage
    n = prod(sz) * sum(.!iszero,stencil)

    # indices and value of nonzero elements
    Si = zeros(Int,n)
    Sj = zeros(Int,n)
    Ss = zeros(T,n)

    linindices = LinearIndices(sz)
    nnz = 0

    stencil_sz = size(stencil)
    offset = CartesianIndex((stencil_sz .+ 1) .÷ 2)

    for i in CartesianIndices(sz)
        for k in CartesianIndices(stencil_sz)
            if stencil[k] != 0
                j = i + k - offset
                if checkbounds(Bool,linindices,j)
                    nnz = nnz + 1
                    Si[nnz] = linindices[i]
                    Sj[nnz] = linindices[j]
                    Ss[nnz] = stencil[k]
                end
            end
        end
    end

    sparse((@view Si[1:nnz]),
           (@view Sj[1:nnz]),
           (@view Ss[1:nnz]),
           prod(sz),prod(sz))
end



function poisson(T,sz::NTuple{N,Int}) where N
    #=
    In 2D stencil has the value:
    stencil = [0  -1   0;
               -1  4  -1;
               0  -1   0]

    =#
    stencil = zeros(T,ntuple(i -> 3,Val(N)))
    for i = 0:N-1
        Ipre = CartesianIndex(ntuple(l -> 2,i))
        Ipost = CartesianIndex(ntuple(l -> 2,N-i-1))
        stencil[Ipre, 1, Ipost] = -1
        stencil[Ipre, 3, Ipost] = -1
    end
    Icenter = CartesianIndex(ntuple(l -> 2,Val(N)))
    stencil[Icenter] = 2*N

    stencil_grid(T,stencil,sz)
end

poisson(sz::NTuple{N,Int}) where N = poisson(Float64,sz)
