## Preallocated buffers for calculating the trace term
# struct Tracebuffers{T,Q<:Tuple}
#     kernelbuffers::Tuple{Vector{T},Vector{T}}
#     quadbuffers::Q
#
#     function Tracebuffers(quad::AbstractRSBQuadrature{T}; ntasks) where {T}
#         N = lengthnodes(quad)
#         kernelbuffers = (zeros(T, N), zeros(T, N))
#         quadbuffers = ntuple(_ -> createbuffer(quad), ntasks)
#         return new{T,typeof(quadbuffers)}(kernelbuffers, quadbuffers)
#     end
# end

@inline evaluate(f::F, x) where {F<:Function} = f(x)

## Free energy calculation
function freeenergy(
    problem::AbstractRSBProblem{T},
    quad::AbstractRSBQuadrature{T},
    op::Orderparameters{T},
    # buffers::Tracebuffers{T},
) where {T}
    return fieldterm(problem, op) + traceterm(problem, quad, op)
end

function traceterm(problem, quad, (; a, sqrtdq))
    transform = gettransform(quad)
    GJfirst(t) = G0(problem, apply(transform, t)) * jacobian(transform, t)
    isempty(a) && return -Gtrace(GJfirst, geth(problem), last(sqrtdq), quad)
    a1, sqrtdq1 = a[begin], sqrtdq[begin+1]
    GJ = Gmap(GJfirst, a1, sqrtdq1, quad)
    arest, sqrtdqrest = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
    for i in eachindex(arest, sqrtdqrest)
        GJ = Gmap(GJ, arest[i], sqrtdqrest[i], quad)
    end
    return -Gtrace(GJ, geth(problem), last(sqrtdq), quad)
end

function Gtrace(GJ::F, h, sqrtqlast, quad::AbstractRSBQuadrature{T}) where {F,T}
    transform = gettransform(quad)
    C = gaussnorm(sqrtqlast)
    return integralwithgaussian(abs(h), sqrtqlast, quad, transform) do t
        # NOTE: for GaussKronrod, `integrate` should simply pass integers to the integrand,
        # and a corresponding Transform type should be implemented
        μ = apply(transform, t)
        Jμ = jacobian(transform, t)
        GJμ = evaluate(GJ, t)
        return C * (GJμ + Jμ * μ) * wrappedgaussian(μ, h, sqrtqlast; normalized=false)
    end
end

function Gmap(GJ::F, ai, sqrtdqi, quad::AbstractRSBQuadrature{T}) where {F,T}
    scale = getscale(GJ, ai, sqrtdqi, quad)
    ttail_ = oneunit(T)
    Ginterp = approximate(quad) do ts
        reversed = descendingknots(quad)
        Gs = similar(ts)
        indices = eachindex(Gs, ts)
        for i in (reversed ? reverse(indices) : indices)
            t = ts[i]
            G = Gnext(t, GJ, ai, sqrtdqi, quad; scale)
            if G <= zero(G)
                tail = reversed ? @view(Gs[begin:i]) : @view(Gs[i:end])
                fill!(tail, zero(eltype(Gs)))
                ttail_ = t
                break
            end
            Gs[i] = G
        end
        return Gs
    end
    ttail = ttail_  # reassignment to avoid boxing in closures
    return t -> ((t < ttail) ? Ginterp(t) : zero(Base.promote_op(Ginterp, typeof(t))))
end

function getscale(GJ::F, ai, sqrtdqi, quad::AbstractRSBQuadrature{T}) where {F,T}
    return first(_Gnext(first(getlimits(quad)), GJ, ai, sqrtdqi, quad))
end

EnzymeRules.inactive(::typeof(getscale), args...) = true

function Gnext(t, GJ::F, ai, sqrtdqi, quad; kwargs...) where {F}
    Iμ, Jμ = _Gnext(t, GJ, ai, sqrtdqi, quad; kwargs...)
    return Jμ * log1p(Iμ / Jμ) / ai
end

function _Gnext(t, GJ::F, ai, sqrtdqi, quad; kwargs...) where {F}
    C = gaussnorm(sqrtdqi)
    transform = gettransform(quad)
    μ = apply(transform, t)
    isinf(μ) && return zero(C * μ), oneunit(μ)  # strictly speaking Jμ == Inf, but that's not relevant and would result in an unwelcome NaN
    Jμ = jacobian(transform, t)
    CJμ = C * Jμ
    dqi = sqrtdqi^2
    μpadq, μmadq = muladd(ai, dqi, μ), muladd(-ai, dqi, μ)
    # g2shift = -2(ai * μ)
    g2coeff = exp(-2(ai * μ))
    # I1 and I2 are analytically solved parts of the integral that we bake into the
    # integrand for correct quadrature error estimates
    Cerf = C * sqrt(oftype(sqrtdqi, π))  # sqrt of gaussian exponent scaling factor
    I1, I2 = -erfc(Cerf * μpadq) / 2, erfc(Cerf * μmadq) / 2  # I2 to be multiplied by exp(g2shift) via g2
    Iμ = integralwithsplitgaussian(
        abs(μmadq), μpadq, sqrtdqi, quad, transform; kwargs...
    ) do u
        nu = apply(transform, u)
        Jnu = jacobian(transform, u)
        Gnu = evaluate(GJ, u) / Jnu
        # gaussian envelope
        g1 = gaussian(nu - μpadq, sqrtdqi; normalized=false)
        # g2 = gaussian(nu + μmadq, sqrtdqi, g2shift; normalized=false)
        g2 = g2coeff * gaussian(nu + μmadq, sqrtdqi; normalized=false)
        # wrapped gaussian envelope
        g1w = gaussian(-nu - μpadq, sqrtdqi; normalized=false)
        # g2w = gaussian(-nu + μmadq, sqrtdqi, g2shift; normalized=false)
        g2w = g2coeff * gaussian(-nu + μmadq, sqrtdqi; normalized=false)
        return CJμ * Jnu * ((g1 + g2) * expm1(ai * Gnu) + (g1 + g1w) * I1 + (g2 + g2w) * I2)
    end
    return Iμ, Jμ
end

# # TODO: submit two issues
# # - Threads.@threads rule not thread safe does not ensure thread safety of the reverse pass
# # (see Gt_tcopies below for an example of what's required); this is the reason we implement
# # a custom rule here in the first place
# # - Wrong arguments passed to reverse rule when function called in loop where names
# # alternate like in traceterm (hence putting arguments on tape below)
# function EnzymeRules.augmented_primal(
#     ::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(Gmap!)},
#     ::Type{<:Const},
#     Gnext::Duplicated,
#     Gt::Duplicated,
#     ai::Active,
#     sqrtdqi::Active,
#     quad::Const,
#     quadbuffers::Const,
# )
#     Gnext_index_zero = Gmap!(Gnext.val, Gt.val, ai.val, sqrtdqi.val, quad.val, quadbuffers.val)
#     return EnzymeRules.AugmentedReturn(nothing, nothing, (Gnext_index_zero, Gnext, Gt))
# end
#
# function EnzymeRules.reverse(
#     ::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(Gmap!)},
#     ::Type{<:Const},
#     (Gnext_index_zero, Gnext, Gt),  # tape
#     ::Duplicated,
#     ::Duplicated,
#     ai::Active,
#     sqrtdqi::Active,
#     quad::Const,
#     quadbuffers::Union{Const,Duplicated},
# )
#     Gnext_index_last = Gnext_index_zero - 1
#     Gnext.dval[Gnext_index_zero:end] .= zero(eltype(Gnext.dval))
#     update_atol!(Gt.val, ai.val, sqrtdqi.val, quad.val, first(quadbuffers.val))
#     ntasks = length(quadbuffers.val)
#     dai, dsqrtdqi = if ntasks == 1
#         quadbuffer = if quadbuffers isa Const
#             Const(first(quadbuffers.val))
#         else
#             Duplicated(only(quadbuffers.val), only(quadbuffers.dval))
#         end
#         ret = autodiff(
#             Reverse,
#             Const(Gmap_inner!),
#             Const(firstindex(Gnext.val):Gnext_index_last),
#             Gnext,
#             Gt,
#             ai,
#             sqrtdqi,
#             quad,
#             quadbuffer,
#             Const(Val(false)),
#         )[1]
#         (ret[4], ret[5])
#     else
#         Gt_tcopies = ntuple(Val(ntasks - 1)) do _
#             @inline
#             Duplicated(Gt.val, make_zero(Gt.dval))
#         end
#         Gt_ts = (Gt, Gt_tcopies...)
#         dai_ts = Vector{typeof(ai.val)}(undef, ntasks)
#         dsqrtdqi_ts = Vector{typeof(ai.val)}(undef, ntasks)
#         Base.require_one_based_indexing(Gt_ts, dai_ts, dsqrtdqi_ts, quadbuffers.val)
#         Threads.@threads for i in 1:ntasks
#             start_t = (firstindex(Gnext.val) - 1) + i
#             quadbuffer_t = if quadbuffers isa Const
#                 Const(quadbuffers.val[i])
#             else
#                 Duplicated(quadbuffers.val[i], quadbuffers.dval[i])
#             end
#             ret_t = autodiff(
#                 Reverse,
#                 Const(Gmap_inner!),
#                 Const(start_t:length(quadbuffers.val):Gnext_index_last),
#                 Gnext,
#                 Gt_ts[i],
#                 ai,
#                 sqrtdqi,
#                 quad,
#                 quadbuffer_t,
#                 Const(Val(false))
#             )[1]
#             dai_ts[i] = ret_t[4]
#             dsqrtdqi_ts[i] = ret_t[5]
#         end
#         ntuple(Val(ntasks - 1)) do i
#             @inline
#             Gt_dVS = EnzymeRules.VectorSpace(Gt.dval; runtime_inactive=Val(false))
#             Gt_t_dVS = EnzymeRules.VectorSpace(Gt_tcopies[i].dval; runtime_inactive=Val(false))
#             Gt_dVS .+= Gt_t_dVS
#             nothing
#         end
#         (sum(dai_ts), sum(dsqrtdqi_ts))
#     end
#     return (nothing, nothing, dai, dsqrtdqi, nothing, nothing)
# end
