module ReplicaSymmetryBreakingTests

using ReplicaSymmetryBreaking: ReplicaSymmetryBreaking

using Test: @testset, @test
using Aqua: Aqua
using ExplicitImports:
    check_no_implicit_imports,
    check_no_stale_explicit_imports,
    check_no_self_qualified_accesses,
    check_all_explicit_imports_via_owners,
    check_all_explicit_imports_are_public,
    check_all_qualified_accesses_via_owners,
    check_all_qualified_accesses_are_public

@testset "ReplicaSymmetryBreaking.jl" begin
    @testset "Aqua" Aqua.test_all(ReplicaSymmetryBreaking)
    @testset "ExplicitImports" begin
        @test check_no_implicit_imports(ReplicaSymmetryBreaking) === nothing
        @test check_no_stale_explicit_imports(ReplicaSymmetryBreaking) === nothing
        @test check_no_self_qualified_accesses(ReplicaSymmetryBreaking) === nothing
        @test check_all_explicit_imports_via_owners(ReplicaSymmetryBreaking) === nothing
        @test check_all_explicit_imports_are_public(ReplicaSymmetryBreaking) === nothing
        @test check_all_qualified_accesses_via_owners(ReplicaSymmetryBreaking) === nothing
        @test check_all_qualified_accesses_are_public(ReplicaSymmetryBreaking) === nothing
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
    @testset "ExplicitImports" begin
        @test check_no_implicit_imports(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_no_stale_explicit_imports(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_no_self_qualified_accesses(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_all_explicit_imports_via_owners(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_all_explicit_imports_are_public(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_all_qualified_accesses_via_owners(ReplicaSymmetryBreakingTests, @__FILE__) === nothing
        @test check_all_qualified_accesses_are_public(ReplicaSymmetryBreakingTests, @__FILE__; ignore=(:test_all,)) === nothing
    end
end
