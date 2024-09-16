@testitem "AllocCheck.jl" begin
    code = """
        using Pkg: Pkg
        Pkg.develop(path="..")
        Pkg.instantiate()
        try
            include("alloccheck/runtests.jl")
        finally
            redirect_stderr(devnull) do
                Pkg.rm("ReplicaSymmetryBreaking")
                rm("alloccheck/Manifest.toml")
            end
        end
    """
    cmd = addenv(
        Cmd(`$(Base.julia_cmd()) --eval "$code"`),
        Dict("JULIA_LOAD_PATH" => "@:alloccheck", "JULIA_PROJECT" => nothing),
    )
    @test success(run(cmd))
end

@testitem "inference & allocations" begin
    include("commontestsets.jl")
    using JET: @test_opt, @test_call, AnyFrameModule

    function testinferrednoallocs(f::F, args) where {F}
        result = f(args...)
        @test result == @inferred f(args...)
        @test_call f(args...)
        @test_opt f(args...)
        @test 0 == @allocations f(args...)
        return nothing
    end

    commontestrunner(objective, (testinferrednoallocs,))
end
