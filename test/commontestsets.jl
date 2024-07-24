using ReplicaSymmetryBreaking:
    SKProblem, RSBSolver, initstate, objective
using Test: @testset

function commontestsets(testfs; k=6, βs=(3.0, Inf), nquad=26, μmax=5.0)
    @testset "β = $β" for β in βs 
        problem = SKProblem(; k, β)
        solver = RSBSolver(problem; nquad, μmax)
        state = initstate(problem)
        args = Dict(objective => (state, solver))
        for (f, testf) in pairs(testfs)
            testf(f, args[f])
        end
    end
    return nothing
end
