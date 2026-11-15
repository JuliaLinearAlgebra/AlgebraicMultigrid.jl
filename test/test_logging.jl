@testset "Logging and verbose output" begin
    A = poisson(100)
    b = ones(size(A, 1))

    @testset "Hierarchy logging ($build)" for build in (ruge_stuben, smoothed_aggregation)
        # Silent by default; the hierarchy is available as a debug message.
        @test_logs build(A)
        @test_logs (:debug, r"Multilevel Solver") match_mode=:any min_level=Base.CoreLogging.Debug build(A)
        # `verbose = true` shows it at Info level.
        @test_logs (:info, r"Multilevel Solver") build(A; verbose = true)
    end

    @testset "Coarsening failure warning ($build)" for build in (ruge_stuben, smoothed_aggregation)
        # No strong connections, so the input matrix cannot be coarsened at all.
        A_diag = spdiagm(0 => 2.0 * ones(20))
        ml = @test_logs (:warn, r"Could not coarsen the input matrix") build(A_diag)
        @test length(ml) == 1
    end

    @testset "Coarsening stall below the first level" begin
        # A stall deeper in the hierarchy only leaves a larger coarsest level: debug, not warn.
        @test_logs AlgebraicMultigrid._log_coarsening_stalled(A, 10, 3)
        @test_logs (:debug, r"Coarsening stalled") min_level=Base.CoreLogging.Debug AlgebraicMultigrid._log_coarsening_stalled(A, 10, 3)
    end

    @testset "Verbose residual trace" begin
        ml = ruge_stuben(A)
        residuals = nothing
        out = mktemp() do path, io
            redirect_stdout(io) do
                _, residuals = AlgebraicMultigrid._solve(ml, b; verbose = true, log = true, maxiter = 3)
            end
            flush(io)
            read(path, String)
        end
        lines = split(chomp(out), '\n')
        # Initial residual (iteration 0) plus one line per iteration, matching the log.
        @test length(lines) == length(residuals) == 4
        @test startswith(lines[1], "Norm of residual at iteration      0")
        @test startswith(lines[end], "Norm of residual at iteration      3")
        for (line, r) in zip(lines, residuals)
            @test parse(Float64, last(split(line))) ≈ r rtol = 1e-4
        end
    end

    @testset "Initial residual uses the initial guess" begin
        ml = ruge_stuben(A)
        x0 = fill(0.5, size(A, 1))
        _, residuals = AlgebraicMultigrid._solve!(copy(x0), ml, b; log = true, maxiter = 2)
        @test residuals[1] ≈ norm(b - A * x0)

        # An initial guess that already meets the tolerance needs no iterations.
        x_exact = A \ b
        x, residuals = AlgebraicMultigrid._solve!(copy(x_exact), ml, b; log = true)
        @test length(residuals) == 1
        @test x == x_exact
    end
end
