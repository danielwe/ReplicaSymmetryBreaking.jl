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

export GaussKronrod, Orderparameters, freeenergy, initstate, Workspace, objective, optimize,
    SKProblem

include("abstract.jl")

include("utils.jl")
include("gaussians.jl")
include("gausskronrod.jl")

include("orderparameters.jl")
include("freeenergy.jl")

include("state.jl")
include("optimization.jl")

include("skproblem.jl")

end  # module
