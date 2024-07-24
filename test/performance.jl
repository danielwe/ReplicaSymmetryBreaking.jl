@testitem "AllocCheck.jl" begin
    code = "using Pkg; Pkg.update(); include(\"alloccheck/runtests.jl\")"
    cmd = addenv(
        Cmd(`$(Base.julia_cmd()) --eval "$code"`),
        Dict("JULIA_LOAD_PATH" => "@:alloccheck", "JULIA_PROJECT" => nothing),
    )
    @test success(run(cmd))
end

@testitem "inference & allocations" begin
    include("commontestsets.jl")
    using JET: @test_opt

    function testinferrednoallocs(f::F, args) where {F}
        @inferred f(args...)
        @test_opt f(args...)
        @test 0 == @allocations f(args...)
        return nothing
    end

    commontestsets(Dict(objective => testinferrednoallocs))
end
