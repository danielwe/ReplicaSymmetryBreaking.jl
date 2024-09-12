@testitem "Aqua.jl" begin
    using Aqua

    # The default ambiguity test applies to [mod, Base, Core], which is poisoned
    # beyond repair by imported packages. Run test_ambiguities separately instead.
    Aqua.test_all(ReplicaSymmetryBreaking; ambiguities=false)
    Aqua.test_ambiguities(ReplicaSymmetryBreaking)
end

@testitem "ExplicitImports.jl" begin
    using ExplicitImports

    const RSB = ReplicaSymmetryBreaking
    @test check_no_implicit_imports(RSB) === nothing
    @test check_no_stale_explicit_imports(RSB) === nothing
    @test check_no_self_qualified_accesses(RSB) === nothing
    @test check_all_explicit_imports_via_owners(RSB) === nothing
    @test check_all_qualified_accesses_via_owners(RSB) === nothing
    @test check_all_explicit_imports_are_public(RSB) === nothing
    let ignore=(
        :promote_op,
        :require_one_based_indexing,
        :RefValue,
        :threadpoolsize,
        :inactive,
        :inactive_type,
        :API,
        :fast_math!,
        :runtimeActivity!,
        :typeWarning!,
        :maxtypedepth!,
        # :maxtypeoffset!,
        # :looseTypeAnalysis!,
        # :inlineall!,
        # :strictAliasing!,
    )
        @test check_all_qualified_accesses_are_public(RSB; ignore) === nothing
    end
end
