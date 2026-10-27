struct ChebyshevInterpolator{T,S,P} <: AbstractInterpolator{T}
    space::S
    knots::Vector{Vector{T}}
    plans::Vector{P}

    function ChebyshevInterpolator{T}() where {T}
        space = Chebyshev(ChebyshevInterval{T}())
        knots = collect.(ApproxFun.points.((space,), 2 .^ (6:12)))
        plans = ApproxFunBase.plan_transform!.((space,), knots)
        # plans = ApproxFunBase.plan_transform.((space,), knots)
        return new{T,typeof(space),eltype(plans)}(space, knots, plans)
    end
end

# private interface
_space(app::ChebyshevInterpolator) = app.space
Base.@propagate_inbounds _knots(app::ChebyshevInterpolator, i) = app.knots[i]
Base.@propagate_inbounds _plan(app::ChebyshevInterpolator, i) = app.plans[i]
_nplans(app::ChebyshevInterpolator) = length(app.plans)

EnzymeRules.inactive(::typeof(_space), args...) = true
EnzymeRules.inactive(::typeof(_knots), args...) = true
EnzymeRules.inactive(::typeof(_plan), args...) = true

# public interface
limits(app::ChebyshevInterpolator) = endpoints(_space(app).domain)

function approximate(fvectorized::F, app::ChebyshevInterpolator) where {F}
    return Fun(_space(app), chebcoeffs(fvectorized, app))
end

function chebcoeffs(fvec::F, app::ChebyshevInterpolator) where {F}
    # factor out the adaptive part and mark it inactive such that Enzyme can compile a
    # straight-line reverse pass, even though this comes at the expense of one extra
    # plan * fvec(knots) in the forward pass
    # if it weren't for Enzyme.jl#2304 we could use the custom rule commented below and
    # avoid this overhead
    i, c, L = _chebcoeffs_adapt(fvec, app)
    # coeffs = c  # if only the custom rule worked...
    coeffs = Enzyme.within_autodiff() ? _chebcoeffs_fixed(fvec, app, i) : c
    if L < length(coeffs)
        resize!(coeffs, L)
    end
    return coeffs
end

function _chebcoeffs_fixed(fvec::F, app::ChebyshevInterpolator, i) where {F}
    return _plan(app, i) * fvec(_knots(app, i), Descending())
end

function _chebcoeffs_adapt(fvec::F, app::ChebyshevInterpolator) where {F}
    for i in 1:_nplans(app)
        coeffs = _chebcoeffs_fixed(fvec, app, i)
        L = _countsignificantcoeffs(coeffs)
        (L > 0) && return i, coeffs, L
    end
    throw(ChebyshevConvergenceError())
end

EnzymeRules.inactive(::typeof(_chebcoeffs_adapt), args...) = true

function _countsignificantcoeffs(coeffs)
    # hardcoded heuristic that quite reliably obtains machine precision and seems robust;
    # naively cutting off at abs(coeff) ≈ eps(maxcoeff) could be rather arbitrary due to
    # coefficient noise/uncertainty being comparable to/larger than coefficient magnitude;
    # we therefore pick a threshold a few orders of magnitude larger than this and require
    # some fraction of the coefficients to be in the tail of coefficients smaller than this;
    # experience suggests that cutting off at an even number of coefficients sometimes
    # results in better behavior---maybe because basis functions can be grouped into
    # even-odd pairs and it's somehow better to keep both for a given fld(order, 2)? Or
    # maybe I'm just imagining things
    Base.require_one_based_indexing(coeffs)
    ntail(N_) = round(Int, 4log2(N_))  # length of tail given total length N_
    # ntail(N_) = min(div(N_, 5), 64)
    N = length(coeffs)
    @assert iseven(N)
    lastbulkindex = N - ntail(N)  # bulk == !tail
    tailthreshold = maximum(abs, coeffs) * (eps(eltype(coeffs))^(5 // 6))
    if any(i -> (abs(coeffs[i]) > tailthreshold), (lastbulkindex+1):N)
        # verify that all tail coeffs are smaller than threshold, otherwise return 0 to
        # indicate that an insufficient number of coefficients have been computed
        return 0
    end
    # move down in steps of two: once the resulting tail is noncompliant, return the
    # previous value, which is the minimal length that satisfies tail threshold criterion
    M = N - 2
    lastbulkindex_prev = lastbulkindex
    lastbulkindex = M - ntail(M)
    while lastbulkindex > 0
        if any(i -> (abs(coeffs[i]) > tailthreshold), (lastbulkindex+1):lastbulkindex_prev)
            return M + 2
        end
        M -= 2
        lastbulkindex_prev = lastbulkindex
        lastbulkindex = M - ntail(M)
    end
    throw(ChebyshevConvergenceError())  # something went wrong, unclear what, so let's throw
end

struct ChebyshevConvergenceError <: AbstractRSBException end

function Base.showerror(io::IO, ::ChebyshevConvergenceError)
    print(io, "Failed to compute Chebyshev interpolation to machine precision")
    return nothing
end

# # Enzyme rules that currently don't work correctly due to Enzyme.jl#2304
# function EnzymeRules.augmented_primal(
#     config::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(chebcoeffs)},
#     ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
#     f,
#     app::Const,
# )
#     needs_primal = EnzymeRules.needs_primal(config)
#     needs_shadow = EnzymeRules.needs_shadow(config)
#     tape_fval = EnzymeRules.overwritten(config)[2] ? deepcopy(f.val) : nothing
#     primal, shadow, tape = if needs_primal || needs_shadow
#         i, coeffs, _ = _chebcoeffs_adapt(f.val, app.val)
#         p = needs_primal ? coeffs : nothing
#         s = if needs_primal && needs_shadow
#             make_zero(coeffs)
#         elseif needs_shadow  # might as well reuse coeffs as shadow
#             make_zero!(coeffs)
#             coeffs
#         else
#             nothing
#         end
#         (p, s, (i, s, tape_fval))
#     else
#         (nothing, nothing, (nothing, nothing, tape_fval))
#     end
#     return EnzymeRules.AugmentedReturn(primal, shadow, tape)
# end
#
# function EnzymeRules.reverse(
#     config::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(chebcoeffs)},
#     ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
#     tape::Tuple{Int,AbstractVector{T},Any},
#     f::F,
#     app::Q
# ) where {F,T,Q<:Const{<:ChebyshevInterpolator{T}}}
#     i, shadow = tape
#     # XXX: workaround for Enzyme.jl#2304: would like to use the primal from the tape and
#     # accumulate into the shadow from the argument; they may in principle contain values
#     # with mismatched shapes, but that doesn't seem to be the case here
#     ff = EnzymeRules.overwritten(config)[2] ? mergeannotation(last(tape), f) : f
#     fwd, rev = autodiff_thunk(
#         ReverseSplitNoPrimal, Const{typeof(_chebcoeffs_fixed)}, Duplicated, F, Q, Const{Int}
#     )
#     thunktape, _, thunkshadow = fwd(Const(_chebcoeffs_fixed), ff, app, Const(i))
#     thunkshadow .= shadow
#     df = first(only(rev(Const(_chebcoeffs_fixed), ff, app, Const(i), thunktape)))
#     make_zero!(shadow)
#     return (df, nothing)
# end
#
# mergeannotation(newxval::T, ::Const{T}) where {T} = Const(newxval)::Const{T}
# mergeannotation(newxval::T, ::Active{T}) where {T} = Active(newxval)::Active{T}
# mergeannotation(newxval::T, x::Duplicated{T}) where {T} = Duplicated(newxval, x.dval)::Duplicated{T}
# mergeannotation(newxval::T, x::DuplicatedNoNeed{T}) where {T} = DuplicatedNoNeed(newxval, x.dval)::DuplicatedNoNeed{T}
# mergeannotation(newxval::T, x::MixedDuplicated{T}) where {T} = MixedDuplicated(newxval, x.dval)::MixedDuplicated{T}
