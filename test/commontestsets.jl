using ReplicaSymmetryBreaking: GaussKronrod, initstate, Workspace, objective, SKProblem
using Test: @testset

function commontestsets(testf; β=(10.0, Inf), cutoff=(50.0, Inf))
    @testset "β = $β_, cutoff = $cutoff_" for β_ in β, cutoff_ in cutoff
        testf(β_, cutoff_)
    end
    return nothing
end

function commontestrunner(::typeof(objective), testfs; kwargs...)
    commontestsets(kwargs...) do β, cutoff
        u0, problem = commonproblem(β)
        quad, ws = commonsetup(problem, cutoff)
        for testf in testfs
            testf(objective, (u0, problem, quad, ws))
        end
    end
end

function commonproblem(β; k=2)
    problem = SKProblem(; k, β)
    u0 = initstate(problem)
    return (u0, problem)
end

function commonsetup(problem, cutoff; n=128, tol=16.0^(-9), ntasks=1)
    quad = GaussKronrod(cutoff, n; tol)
    ws = Workspace(problem, quad; ntasks)
    return (quad, ws)
end
