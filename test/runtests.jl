module ReplicaSymmetryBreakingTests

using ReplicaSymmetryBreaking:
    ReplicaSymmetryBreaking,
    SKProblem,
    GaussLegendre,
    RSBWorkspace,
    initrsbweights,
    freeenergy!

using Test: @testset, @test, @inferred
using Aqua: Aqua
using ExplicitImports:
    check_no_implicit_imports,
    check_no_stale_explicit_imports,
    check_no_self_qualified_accesses,
    check_all_explicit_imports_via_owners,
    check_all_qualified_accesses_via_owners,
    check_all_explicit_imports_are_public,
    check_all_qualified_accesses_are_public
using JET: @test_opt

@testset "ReplicaSymmetryBreaking.jl" begin
    let mod=ReplicaSymmetryBreaking
        @testset "Aqua" Aqua.test_all(mod)
        @testset "ExplicitImports" begin
            @test check_no_implicit_imports(mod) === nothing
            @test check_no_stale_explicit_imports(mod) === nothing
            @test check_no_self_qualified_accesses(mod) === nothing
            @test check_all_explicit_imports_via_owners(mod) === nothing
            @test check_all_qualified_accesses_via_owners(mod) === nothing
            @test check_all_explicit_imports_are_public(mod) === nothing
            @test check_all_qualified_accesses_are_public(mod) === nothing
        end
    end
    let kRSB=6, quad=GaussLegendre(15, 0.0, 7.5)
        @testset "β = $β" for β in (10.0, Inf)
            problem = SKProblem(; kRSB, β)
            weights = initrsbweights(problem)
            workspace = RSBWorkspace(problem, quad)
            @testset "Inference/allocations" begin
                @inferred freeenergy!(workspace, problem, weights, quad)
                @test_opt freeenergy!(workspace, problem, weights, quad)
                @test 0 == @allocations freeenergy!(workspace, problem, weights, quad)
            end
        end
    end
end

end  # module

using Test: @testset, @test
using ExplicitImports:
    check_no_implicit_imports,
    check_no_stale_explicit_imports,
    check_no_self_qualified_accesses,
    check_all_explicit_imports_via_owners,
    check_all_explicit_imports_are_public,
    check_all_qualified_accesses_via_owners,
    check_all_qualified_accesses_are_public

@testset "ReplicaSymmetryBreakingTests" begin
    let mod=(ReplicaSymmetryBreakingTests, @__FILE__)
        @testset "ExplicitImports" begin
            @test check_no_implicit_imports(mod...) === nothing
            @test check_no_stale_explicit_imports(mod...) === nothing
            @test check_no_self_qualified_accesses(mod...) === nothing
            @test check_all_explicit_imports_via_owners(mod...) === nothing
            @test check_all_qualified_accesses_via_owners(mod...) === nothing
            let ignore=(:GaussLegendre, :RSBWorkspace, :initrsbweights, :freeenergy!)
                @test check_all_explicit_imports_are_public(mod...; ignore) === nothing
            end
            let ignore=(:test_all,)
                @test check_all_qualified_accesses_are_public(mod...; ignore) === nothing
            end
        end
    end
end
