module ReplicaSymmetryBreakingAllocCheck

using ReplicaSymmetryBreaking:
    SKProblem,
    GaussLegendre,
    RSBWorkspace,
    initoptimizationstate,
    rsbobjective

using Test: @testset, @test
using AllocCheck: check_allocs

include("../qualityassurance.jl")

using .QualityAssurance: explicit_import_tests

@testset "AllocCheck QA" begin
    explicit_import_tests(
        @__MODULE__, @__FILE__;
        ignore_imports=(:GaussLegendre, :RSBWorkspace, :initoptimizationstate, :rsbobjective),
    )
end

@testset "AllocCheck" begin
    let k=6, quad=GaussLegendre(15, 0.0, 7.5)
        @testset "β = $β" for β in (10.0, Inf)
            problem = SKProblem(; k, β)
            state = initoptimizationstate(problem)
            workspace = RSBWorkspace(problem, quad)
            p = (problem, workspace, quad)
            @test let
                allocations = check_allocs(rsbobjective, typeof.((state, p)))
                if isempty(allocations)
                    true
                else
                    for allocation in allocations
                        @show allocation
                    end
                    false
                end
            end
        end
    end
end

end  # module
