@testitem "symmetry" begin
    include("commontestsets.jl")

    commontestsets() do β, cutoff
        u0, problem = commonproblem(β)
        quadp, wsp = commonsetup(problem, cutoff)
        quadm, wsm = commonsetup(problem, -cutoff)
        @test objective(u0, problem, quadp, wsp) == objective(u0, problem, quadm, wsm)
    end
end
