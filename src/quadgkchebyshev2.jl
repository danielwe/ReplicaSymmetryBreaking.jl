struct QuadGKChebyshev{T,stretch,S,P} <: AbstractRSBQuadGK{T,stretch}
    order::Int
    rtol::T
    space::S
    knots::Vector{Vector{T}}
    plans::Vector{P}
    function QuadGKChebyshev{T}(order; rtol=64eps(), stretch=:power) where {T}
        @assert rtol > 0
        space = Chebyshev(ChebyshevInterval{T}())
        knots = collect.(ApproxFun.points.(space, 2 .^ (6:12)))
        plans = ApproxFunBase.plan_transform!.(space, knots)
        # plans = ApproxFunBase.plan_transform.(space, knots)
        S, P = typeof(space), eltype(plans)
        return new{T,stretch,S,P}(order, rtol, space, knots, plans)
    end
end

QuadGKChebyshev(args...; kwargs...) = QuadGKChebyshev{Float64}(args...; kwargs...)

descendingknots(::QuadGKChebyshev) = true
getlimits(quad::QuadGKChebyshev) = endpoints(quad.space.domain)
getspace(quad::QuadGKChebyshev) = quad.space
getknots(quad::QuadGKChebyshev, i) = quad.knots[i]
getplan(quad::QuadGKChebyshev, i) = quad.plans[i]
nplans(quad::QuadGKChebyshev) = length(quad.plans)
EnzymeRules.inactive(::typeof(getspace), args...) = true
EnzymeRules.inactive(::typeof(getknots), args...) = true
EnzymeRules.inactive(::typeof(getplan), args...) = true

approximate(f::F, quad::QuadGKChebyshev) where {F} = Fun(getspace(quad), chebcoeffs(f, quad))

# factor out the adaptive part and mark it inactive such that Enzyme can compile a
# straight-line reverse pass, even though this comes at the expense of one extra
# plan * fvec(knots) in the forward pass
# wish we could use the custom rule from the commented-out section below to eliminate that
# overhead, but Enzyme.jl#2304 gets in the way
function chebcoeffs(fvec::F, quad::QuadGKChebyshev) where {F}
    i, c, L = chebcoeffs_adapt(fvec, quad)
    # coeffs = c  # if only the custom rule worked...
    coeffs = Enzyme.within_autodiff() ? chebcoeffs_fixed(fvec, quad, i) : c
    if L < length(coeffs)
        resize!(coeffs, L)
    end
    return coeffs
end

function chebcoeffs_fixed(fvec::F, quad::QuadGKChebyshev, i) where {F}
    return getplan(quad, i) * fvec(getknots(quad, i))
end

function chebcoeffs_adapt(fvec::F, quad::QuadGKChebyshev) where {F}
    for i in 1:nplans(quad)
        coeffs = chebcoeffs_fixed(fvec, quad, i)
        L = chebtrunc(coeffs)
        (L > 0) && return i, coeffs, L
    end
    throw(QuadGKChebyshevError())
end

EnzymeRules.inactive(::typeof(chebcoeffs_adapt), args...) = true

function chebtrunc(coeffs)
    Base.require_one_based_indexing(coeffs)
    ntail(N_) = round(Int, 4log2(N_))
    # ntail(N_) = min(div(N_, 5), 64)
    N = length(coeffs)
    @assert iseven(N)
    lastnontail = N - ntail(N)
    tailtol = maximum(abs, coeffs) * (eps(eltype(coeffs))^(5 // 6))
    if any(i -> (abs(coeffs[i]) > tailtol), (lastnontail+1):N)
        return 0
    end
    M = N - 2
    lastnontail_prev = lastnontail
    lastnontail = M - ntail(M)
    while lastnontail > 0
        if any(i -> (abs(coeffs[i]) > tailtol), (lastnontail+1):lastnontail_prev)
            return M + 2
        end
        M -= 2
        lastnontail_prev = lastnontail
        lastnontail = M - ntail(M)
    end
    throw(QuadGKChebyshevError())
end

struct QuadGKChebyshevError <: AbstractRSBException end

function Base.showerror(io::IO, ::QuadGKChebyshevError)
    print(io, "Failed to compute Chebyshev interpolation to machine precision")
    return nothing
end

# function EnzymeRules.augmented_primal(
#     config::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(chebcoeffs)},
#     ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
#     f,
#     quad::Const,
# )
#     needs_primal = EnzymeRules.needs_primal(config)
#     needs_shadow = EnzymeRules.needs_shadow(config)
#     tape_fval = EnzymeRules.overwritten(config)[2] ? deepcopy(f.val) : nothing
#     primal, shadow, tape = if needs_primal || needs_shadow
#         i, coeffs, _ = chebcoeffs_adapt(f.val, quad.val)
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
#     quad::Q
# ) where {F,T,Q<:Const{<:QuadGKChebyshev{T}}}
#     i, shadow = tape
#     # XXX: workaround for Enzyme.jl#2304: would like to use the primal from the tape and
#     # accumulate into the shadow from the argument; they may in principle contain values
#     # with mismatched shapes, but that doesn't seem to be the case here
#     ff = EnzymeRules.overwritten(config)[2] ? mergeannotation(last(tape), f) : f
#     fwd, rev = autodiff_thunk(
#         ReverseSplitNoPrimal, Const{typeof(chebcoeffs_fixed)}, Duplicated, F, Q, Const{Int}
#     )
#     thunktape, _, thunkshadow = fwd(Const(chebcoeffs_fixed), ff, quad, Const(i))
#     thunkshadow .= shadow
#     df = first(only(rev(Const(chebcoeffs_fixed), ff, quad, Const(i), thunktape)))
#     make_zero!(shadow)
#     return (df, nothing)
# end
#
# mergeannotation(newxval::T, ::Const{T}) where {T} = Const(newxval)::Const{T}
# mergeannotation(newxval::T, ::Active{T}) where {T} = Active(newxval)::Active{T}
# mergeannotation(newxval::T, x::Duplicated{T}) where {T} = Duplicated(newxval, x.dval)::Duplicated{T}
# mergeannotation(newxval::T, x::DuplicatedNoNeed{T}) where {T} = DuplicatedNoNeed(newxval, x.dval)::DuplicatedNoNeed{T}
# mergeannotation(newxval::T, x::MixedDuplicated{T}) where {T} = MixedDuplicated(newxval, x.dval)::MixedDuplicated{T}
