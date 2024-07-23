module QualityAssurance

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

export aqua_tests, explicit_import_tests

function aqua_tests(mod)
    @testset "Aqua $(nameof(mod))" begin
        # The default ambiguity test applies to [mod, Base, Core], which is poisoned
        # beyond repair by imported packages
        Aqua.test_all(mod; ambiguities=false)
        Aqua.test_ambiguities(mod)
    end
    return nothing
end

function explicit_import_tests(mod, f=pathof(mod); ignore_imports=(), ignore_accesses=())
    if nameof(@__MODULE__) in names(mod; all=true)
        # we're checking this module or a module into which this one has been imported
        ignore_accesses = (ignore_accesses..., :test_all, :test_ambiguities)
    end
    @testset "ExplicitImports $(nameof(mod))" begin
        @test check_no_implicit_imports(mod, f) === nothing
        @test check_no_stale_explicit_imports(mod, f) === nothing
        @test check_no_self_qualified_accesses(mod, f) === nothing
        @test check_all_explicit_imports_via_owners(mod, f) === nothing
        @test check_all_qualified_accesses_via_owners(mod, f) === nothing
        @test check_all_explicit_imports_are_public(
            mod, f; ignore=ignore_imports
        ) === nothing
        @test check_all_qualified_accesses_are_public(
            mod, f; ignore=ignore_accesses
        ) === nothing
    end
    return nothing
end

end  # module
