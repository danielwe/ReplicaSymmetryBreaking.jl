module ReplicaSymmetryBreaking

using Enzyme:
    Enzyme, EnzymeRules,
    Reverse, ReverseSplitNoPrimal, ReverseSplitWidth,
    Annotation, Active, Const, Duplicated, DuplicatedNoNeed, MixedDuplicated,
    BatchDuplicated, BatchDuplicatedNoNeed, BatchMixedDuplicated,
    autodiff, autodiff_thunk, make_zero, make_zero!, Forward  #, set_runtime_activity

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
using LogExpFunctions: log1pexp
using SpecialFunctions: erfc
using QuadGK: QuadGK, kronrod, quadgk
using ApproxFun: ApproxFun, ApproxFunBase, Fun, Chebyshev, ChebyshevInterval, :(..), endpoints
# using PCHIPInterpolation
# using RationalFunctionApproximation
# using RationalFunctionApproximation: Approximation
# using ComplexRegions: Path, Segment
# import ScopedValues
# using StaticArrays: SVector
using StaticArrays: MVector

using Ipopt: AddIpoptIntOption, AddIpoptNumOption, AddIpoptStrOption, CreateIpoptProblem,
    IpoptSolve, SetIntermediateCallback

using FiniteDifferences: grad, central_fdm

# export GaussKronrod, QuadGKApproxFun, QuadGKPCHIP,
export QuadGKAAA,
    QuadGKChebyshev,
    getlimits,
    # getnodes,
    # countintegrandcalls,
    Orderparameters, freeenergy, state, initstate,
    # Workspace,
    objective, optimize, optimize_sequence, SKProblem

include("abstract.jl")
include("utils.jl")
include("transforms.jl")

include("orderparameters.jl")
include("skproblem.jl")
include("state.jl")

include("gaussians.jl")
# include("gausskronrod.jl")
# include("quadgkchebyshev.jl")
include("quadgkchebyshev2.jl")
# include("quadgkpchip.jl")
include("aaa.jl")
include("quadgkaaa.jl")
include("patches.jl")

include("freeenergy.jl")

include("optimization.jl")

using PrecompileTools: @compile_workload

@compile_workload begin
    for T in [
        Patches.FastTransforms.ChebyshevTransformPlan,
        Patches.FFTW.r2rFFTWPlan,
        SKProblem,
        QuadGKChebyshev,
        QuadGKAAA,
        GaussianLeakError,
        QuadGKChebyshevError,
        OrderparametersError,
    ]
        EnzymeRules.inactive_type(T)
    end
    # for β in (10.0, Inf), cutoff in (50.0, Inf)
    #     problem = SKProblem(; k=2, β)
    #     quad = GaussKronrod(cutoff, 128; tol=16.0^(-9))
    #     u0 = initstate(problem)
    #     for ntasks in (1, 2, 4, 8)
    #         ws = Workspace(problem, quad; ntasks)
    #         nanobjective(u0, problem, quad, ws)
    #     end
    # end
end

end  # module
