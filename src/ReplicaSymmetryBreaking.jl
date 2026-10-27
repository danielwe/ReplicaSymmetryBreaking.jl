module ReplicaSymmetryBreaking

using Enzyme:
    Enzyme, EnzymeRules,
    Reverse, ReverseSplitNoPrimal, ReverseSplitWidth,
    Annotation, Active, Const, Duplicated, DuplicatedNoNeed, MixedDuplicated,
    BatchDuplicated, BatchDuplicatedNoNeed, BatchMixedDuplicated,
    autodiff, autodiff_thunk, make_zero, make_zero!

function __init__()
    # Enzyme.Compiler.VERBOSE_ERRORS[] = true
    # Enzyme.Compiler.CheckNan[] = true
    Enzyme.API.fast_math!(false)
    # Enzyme.API.strong_zero!(true)
    # Enzyme.API.typeWarning!(true)
    # Enzyme.API.maxtypedepth!(7)
    # Enzyme.API.maxtypeoffset!(2^12)
    # Enzyme.API.looseTypeAnalysis!(true)
    # Enzyme.API.inlineall!(true)
    # Enzyme.API.strictAliasing!(false)
    return nothing
end

using LinearAlgebra: LinearAlgebra, dot, mul!, norm, eigvals!, svd, svd!, QRIteration
using LogExpFunctions: log1pexp, logaddexp
using SpecialFunctions: erfc
using QuadGK: kronrod, quadgk
using ApproxFun: ApproxFun, ApproxFunBase, Fun, Chebyshev, ChebyshevInterval, :(..), endpoints
using StaticArrays: MVector

using Ipopt: AddIpoptIntOption, AddIpoptNumOption, AddIpoptStrOption, CreateIpoptProblem,
    IpoptSolve, SetIntermediateCallback

# export GaussKronrod, QuadGKApproxFun, QuadGKPCHIP,
export QuadGK, ChebyshevInterpolator,
    rsborder, inversetemp, externalfield, blockparams, overlapparams, limits,
    # countintegrandcalls,
    Orderparameters, freeenergy, state,
    # initstate,
    initorderparameters,
    objective, optimize, #  optimize_sequence,
    orderparameters,
    SKProblem

# abstract types and interfaces
include("abstract.jl")

# free energy calculation
include("rsbsolver.jl")
include("freeenergy.jl")
include("transforms.jl")
include("gaussians.jl")

# quadratures and approximants
# include("gausskronrod.jl")
include("quadgk.jl")
include("chebyshev.jl")

# optimization
include("optimization.jl")
include("state.jl")
include("ipopt.jl")
include("exceptionhandling.jl")

# problem types
include("skproblem.jl")
include("cimproblem.jl")

# custom rules
include("rules.jl")

# using PrecompileTools: @compile_workload
#
# @compile_workload begin
#     # for β in (10.0, Inf), cutoff in (50.0, Inf)
#     #     problem = SKProblem(; k=2, β)
#     #     quad = GaussKronrod(cutoff, 128; tol=16.0^(-9))
#     #     u0 = initstate(problem)
#     #     for ntasks in (1, 2, 4, 8)
#     #         ws = Workspace(problem, quad; ntasks)
#     #         nanobjective(u0, problem, quad, ws)
#     #     end
#     # end
# end

end  # module
