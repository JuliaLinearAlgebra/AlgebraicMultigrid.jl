poisson(T, n) = sparse(Tridiagonal(fill(T(-1), n-1), 
                        fill(T(2), n), fill(T(-1), n-1)))
poisson(n) = poisson(Float64, n)
