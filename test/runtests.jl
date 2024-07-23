module ReplicaSymmetryBreakingTests

using ReplicaSymmetryBreaking:
    ReplicaSymmetryBreaking,
    SKProblem,
    GaussLegendre,
    RSBWorkspace,
    initoptimizationstate,
    rsbobjective

using Test: @testset, @test, @inferred
using JET: @test_opt

include("qualityassurance.jl")

using .QualityAssurance: aqua_tests, explicit_import_tests

@testset "AllocCheck launcher" begin
    let code="using Pkg: Pkg; Pkg.update(); include(\"alloccheck/alloccheck.jl\")"
        alloccheck = addenv(
            Cmd(`$(Base.julia_cmd()) --eval "$code"`),
            Dict("JULIA_LOAD_PATH" => "@:alloccheck", "JULIA_PROJECT" => nothing),
        )
        @test success(run(alloccheck))
    end
end

@testset "Tests QA" begin
    explicit_import_tests(
        @__MODULE__, @__FILE__;
        ignore_imports=(:GaussLegendre, :RSBWorkspace, :initoptimizationstate, :rsbobjective),
    )
end

@testset "ReplicaSymmetryBreaking.jl" begin
    @testset "QA" begin
        aqua_tests(ReplicaSymmetryBreaking)
        explicit_import_tests(
            ReplicaSymmetryBreaking; ignore_accesses=(:LD_LBFGS, :successful_retcode)
        )
    end
    @testset "inference/allocations" begin
        let k=6, quad=GaussLegendre(15, 0.0, 7.5)
            @testset "β = $β" for β in (10.0, Inf)
                problem = SKProblem(; k, β)
                state = initoptimizationstate(problem)
                workspace = RSBWorkspace(problem, quad)
                p = (problem, workspace, quad)
                @inferred rsbobjective(state, p)
                @test_opt rsbobjective(state, p)
                @test 0 == @allocations rsbobjective(state, p)
            end
        end
    end
end

end  # module
