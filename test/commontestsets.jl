using ReplicaSymmetryBreaking: GaussKronrod, initstate, Workspace, objective, SKProblem
using Test: @testset

function commontestsets(
    testfs; k=2, β=(10.0, Inf), quadn=256, quadcutoff=(50.0, Inf), quadscale=8.75
)
    @testset(
        "k = $k_, β = $β_, quadcutoff = $quadcutoff_, quadscale = $quadscale_",
        for k_ in k, β_ in β, quadcutoff_ in quadcutoff, quadscale_ in quadscale
            problem = SKProblem(; k=k_, β=β_)
            u = initstate(problem)
            quad = GaussKronrod(quadn, quadcutoff_; scale=quadscale_)
            ws = Workspace(u, problem, quad; ntasks=1)
            args = Dict(objective => (u, problem, quad, ws))
            for (f, testf) in pairs(testfs)
                testf(f, args[f])
            end
        end,
    )
    return nothing
end
