poisson(n) = sparse(Tridiagonal(fill(-1, n-1), fill(2, n), fill(-1, n-1)))
