module ReplicaSymmetryBreaking

using Enzyme: Enzyme, EnzymeRules, Forward, Reverse, Active, Const, Duplicated, autodiff,
    autodiff_deferred, make_zero

# Enzyme.make_zero! is slow when recursing through nested data structures, so here's
# an alternative to which we will add methods for each relevant type
function _make_zero!(x::AbstractArray{T}) where {T}
    fill!(x, zero(T))
    return nothing
end

function __init__()
    Enzyme.API.fast_math!(false)
    Enzyme.API.runtimeActivity!(true)
    Enzyme.API.typeWarning!(true)
    Enzyme.API.maxtypedepth!(7)
    # Enzyme.API.maxtypeoffset!(2^12)
    # Enzyme.API.looseTypeAnalysis!(true)
    # Enzyme.API.inlineall!(true)
    # Enzyme.API.strictAliasing!(false)
    return nothing
end

using LinearAlgebra: dot, norm
using LogExpFunctions: log1pexp
using QuadGK: QuadGK, kronrod
using SpecialFunctions: erfc

using Ipopt: AddIpoptIntOption, AddIpoptNumOption, AddIpoptStrOption, CreateIpoptProblem,
    IpoptSolve, SetIntermediateCallback

export GaussKronrod, Orderparameters, freeenergy, state, initstate, Workspace, objective,
    optimize, SKProblem

include("abstract.jl")

include("utils.jl")
include("gaussians.jl")
include("gausskronrod.jl")

include("orderparameters.jl")
include("freeenergy.jl")

include("state.jl")
include("optimization.jl")

include("skproblem.jl")

using PrecompileTools: @compile_workload

@compile_workload begin
    for β in (10.0, Inf), cutoff in (50.0, Inf)
        problem = SKProblem(; k=2, β)
        quad = GaussKronrod(cutoff, 128; tol=16.0^(-9))
        u0 = initstate(problem)
        for ntasks in (1, 2, 4, 8)
            ws = Workspace(problem, quad; ntasks)
            nanobjective(u0, problem, quad, ws)
        end
    end
end

end  # module
