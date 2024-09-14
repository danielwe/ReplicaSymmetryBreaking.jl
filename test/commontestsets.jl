using ReplicaSymmetryBreaking: GaussKronrod, initstate, Workspace, objective, SKProblem
using Test: @testset

function commontestsets(testfs; β=(10.0, Inf), cutoff=(50.0, Inf))
    @testset "β = $β_, cutoff = $cutoff_" for β_ in β, cutoff_ in cutoff
        problem = SKProblem(; k=2, β=β_)
        quad = GaussKronrod(cutoff_, 128; tol=16.0^(-9))
        ws = Workspace(problem, quad; ntasks=1)
        u0 = initstate(problem)
        args = Dict(objective => (u0, problem, quad, ws))
        for (f, testf) in pairs(testfs)
            testf(f, args[f])
        end
    end
    return nothing
end
