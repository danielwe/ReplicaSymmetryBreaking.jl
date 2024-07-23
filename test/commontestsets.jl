using ReplicaSymmetryBreaking:
    SKProblem, GaussLegendre, RSBWorkspace, initoptimizationstate, rsbobjective
using Test: @testset

function commontestsets(testfs; k=6, quad=GaussLegendre(15, 0.0, 7.5), βs=(3.0, Inf))
    @testset "β = $β" for β in βs 
        problem = SKProblem(; k, β)
        state = initoptimizationstate(problem)
        workspace = RSBWorkspace(problem, quad)
        p = (problem, workspace, quad)
        args = Dict(rsbobjective => (state, p))
        for (f, testf) in pairs(testfs)
            testf(f, args[f])
        end
    end
    return nothing
end
