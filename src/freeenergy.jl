### free energy calculation
function freeenergy(
    problem::AbstractDisorderProblem{T}, quad::AbstractQuadrature{T}, op::Orderparameters{T}
) where {T}
    return fieldterm(problem, op) + traceterm(problem, quad, op)
end

# function traceterm(problem, quad, op)
#     tf = gettransform(quad)
#     GJfirst(t) = G0(problem, apply(tf, t)) * applyjacobian(tf, t)
#     # XXX: We currently do not support a nonzero mean coupling and magnetization
#     # heffective = externalfield(problem) + meancoupling(problem) * magnetization(op)
#     heffective = externalfield(problem)
#     a = blockparams(op)
#     sqrtdq = overlapparams(op)
#     # unroll the first iteration to distinguish GJfirst from GJ (different types)
#     trace = if isempty(a)
#         Gtrace(GJfirst, heffective, last(sqrtdq), quad)
#     else
#         a1, sqrtdq1 = a[begin], sqrtdq[begin+1]
#         GJ = Gmap(GJfirst, a1, sqrtdq1, quad)
#         abulk, sqrtdqbulk = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
#         for i in eachindex(abulk, sqrtdqbulk)
#             GJ = Gmap(GJ, abulk[i], sqrtdqbulk[i], quad)
#         end
#         Gtrace(GJ, heffective, last(sqrtdq), quad)
#     end
#     return -trace
# end

function traceterm(problem, quad, op)
    # XXX: We currently do not support a nonzero mean coupling/magnetization
    # heffective = externalfield(problem) + meancoupling(problem) * magnetization(op)
    heffective = externalfield(problem)
    qf = qfinal(op)
    tf = gettransform(quad)
    GJfirst(t) = G0(problem, apply(tf, t)) * applyjacobian(tf, t)
    trace = if length(op) > 0
        # unroll the first iteration to distinguish GJfirst from GJ (different types)
        (a_first, sqrtdq_first, b_first), op_rest = Iterators.peeled(op)
        GJ = Gmap(GJfirst, problem, a_first, sqrtdq_first, b_first, quad)
        for (ai, sqrtdqi, bi) in op_rest
            GJ = Gmap(GJ, problem, ai, sqrtdqi, bi, quad)
        end
        Gtrace(GJ, heffective, qf, quad)
    else
        Gtrace(GJfirst, heffective, qf, quad)
    end
    return -trace
end

function Gtrace(GJ::F, heffective, sqrtqlast, quad::AbstractQuadrature{T}) where {F,T}
    tf = gettransform(quad)
    Zinv = gaussnorm(sqrtqlast)
    return withgaussianinflectionpoints(abs(heffective), sqrtqlast) do ips
        @inline
        withgtsorted(ips, 2sqrtqlast) do gtips
            @inline
            bps = apply.((tf,), gtips)
            integratewithapproximant(GJ, quad, bps) do t, GJt
                μ, Jμ = apply(tf, t), applyjacobian(tf, t)
                GJμ = GJt
                wg = wrappedgaussian(μ, heffective, sqrtqlast; normalized=false)
                return Zinv * (GJμ + Jμ * μ) * wg
            end
        end
    end
end

# function Gmap(GJ::F, ai, sqrtdqi, quad::AbstractQuadrature{T}) where {F,T}
#     scale = getscale(GJ, ai, sqrtdqi, quad)
#     ttail_ref = Ref{T}(last(limits(quad)))  # TODO: can we make this conditional on approximantiscallable?
#     Ginterp = approximate(quad) do ts, order
#         Gs = similar(ts)
#         indices = eachindex(Gs, ts)
#         for i in ((order isa Descending) ? reverse(indices) : indices)
#             t = ts[i]
#             G = Gnext(t, GJ, ai, sqrtdqi, quad; scale)
#             if (order isa Union{Ascending,Descending}) && (G <= zero(G))
#                 # This optimization relies on gettransform(quad) not flipping the order of
#                 # the limits of integration, i.e., that it has a positive Jacobian. This is
#                 # guaranteed by the constructor in all transforms implemented.
#                 tail = (order isa Descending) ? @view(Gs[begin:i]) : @view(Gs[i:end])
#                 fill!(tail, zero(eltype(Gs)))
#                 ttail_ref[] = min(ttail_ref[], t)  # XXX: not thread safe; make atomic if approximate is made multithreaded
#                 break
#             end
#             Gs[i] = G
#         end
#         return Gs
#     end
#     return if approximantiscallable(quad)
#         ttail = ttail_ref[]  # reassignment to avoid boxing in closures
#         t -> ((t < ttail) ? Ginterp(t) : zero(Base.promote_op(Ginterp, typeof(t))))
#     else
#         Ginterp
#     end
# end

function Gmap(GJ::F, problem, quad::AbstractQuadrature{T}, op, i) where {F,T}
    scale = getscale(GJ, ai, sqrtdqi, quad)
    ttail_ref = Ref{T}(last(limits(quad)))  # TODO: can we make this conditional on approximantiscallable?
    Ginterp = approximate(quad) do ts, order
        Gs = similar(ts)
        indices = eachindex(Gs, ts)
        for i in ((order isa Descending) ? reverse(indices) : indices)
            t = ts[i]
            G = Gnext(t, GJ, ai, sqrtdqi, quad; scale)
            if (order isa Union{Ascending,Descending}) && (G <= zero(G))
                # This optimization relies on gettransform(quad) not flipping the order of
                # the limits of integration, i.e., that it has a positive Jacobian. This is
                # guaranteed by the constructor in all transforms implemented.
                tail = (order isa Descending) ? @view(Gs[begin:i]) : @view(Gs[i:end])
                fill!(tail, zero(eltype(Gs)))
                ttail_ref[] = min(ttail_ref[], t)  # XXX: not thread safe; make atomic if approximate is made multithreaded
                break
            end
            Gs[i] = G
        end
        return Gs
    end
    return if approximantiscallable(quad)
        ttail = ttail_ref[]  # reassignment to avoid boxing in closures
        t -> ((t < ttail) ? Ginterp(t) : zero(Base.promote_op(Ginterp, typeof(t))))
    else
        Ginterp
    end
end

function getscale(GJ::F, ai, sqrtdqi, quad::AbstractQuadrature{T}) where {F,T}
    return first(_Gnext(first(limits(quad)), GJ, ai, sqrtdqi, quad))
end

EnzymeRules.inactive(::typeof(getscale), args...) = true

function Gnext(t, GJ::F, ai, sqrtdqi, quad; kwargs...) where {F}
    Iμ, Jμ = _Gnext(t, GJ, ai, sqrtdqi, quad; kwargs...)
    return Jμ * log1p(Iμ / Jμ) / ai
end

function _Gnext(t, GJ::F, ai, sqrtdqi, quad; kwargs...) where {F}
    Zinv = gaussnorm(sqrtdqi)
    tf = gettransform(quad)
    μ = apply(tf, t)
    isinf(μ) && return zero(Zinv * μ), oneunit(μ)  # strictly speaking Jμ == Inf, but that's irrelevant and would result in a spurious NaN downstream
    Jμ = applyjacobian(tf, t)
    Jμ_Z = Zinv * Jμ
    dqi = sqrtdqi^2
    μpadq, μmadq = muladd(ai, dqi, μ), muladd(-ai, dqi, μ)
    g2coeff = exp(-2(ai * μ))
    # I1 and I2 are analytically solved parts of the integral that we bake into the
    # integrand for correct quadrature error estimates
    C = Zinv * sqrt(oftype(sqrtdqi, π))
    I1, I2 = -erfc(C * μpadq) / 2, erfc(C * μmadq) / 2  # to be multiplied by g2coeff via g2
    Iμ = withsplitgaussianinflectionpoints(abs(μmadq), μpadq, sqrtdqi) do ips
        @inline
        withgtsorted(ips, 2sqrtdqi) do gtips
            @inline
            bps = applyinverse.((tf,), gtips)
            integratewithapproximant(GJ, quad, bps; kwargs...) do u, GJu
                nu, Jnu = apply(tf, u), applyjacobian(tf, u)
                Gnu = GJu / Jnu
                # gaussian envelope
                g1 = gaussian(nu - μpadq, sqrtdqi; normalized=false)
                g2 = g2coeff * gaussian(nu + μmadq, sqrtdqi; normalized=false)
                # wrapped gaussian envelope
                g1wrapped = gaussian(-nu - μpadq, sqrtdqi; normalized=false)
                g2wrapped = g2coeff * gaussian(-nu + μmadq, sqrtdqi; normalized=false)
                I12 = (g1 + g1wrapped) * I1 + (g2 + g2wrapped) * I2
                return Jμ_Z * Jnu * (I12 + (g1 + g2) * expm1(ai * Gnu))
            end
        end
    end
    return Iμ, Jμ
end

function _Gnext_new(t, GJ::F, ai, sqrtdqi, quad; kwargs...) where {F}
    Zinv = gaussnorm(sqrtdqi)
    tf = gettransform(quad)
    μ = apply(tf, t)
    isinf(μ) && return zero(Zinv * μ), oneunit(μ)  # strictly speaking Jμ == Inf, but that's irrelevant and would result in a spurious NaN downstream
    Jμ = applyjacobian(tf, t)
    Jμ_Z = Zinv * Jμ
    dqi = sqrtdqi^2
    Iμ = withsplitgaussianinflectionpoints(abs(μ), sqrtdqi) do ips
        @inline
        withgtsorted(ips, 2sqrtdqi) do gtips
            @inline
            bps = applyinverse.((tf,), gtips)
            integratewithapproximant(GJ, quad, bps; kwargs...) do u, GJu
                nu, Jnu = apply(tf, u), applyjacobian(tf, u)
                Gnu = GJu / Jnu
                # gaussian envelope
                g = wrappedgaussian(nu, μ, sqrtdqi; normalized=false)
                return Jμ_Z * Jnu * g * expm1(ai * (Gnu + dℰ(rsbproblem, b + ai * dqi, μ, b, nu)))
            end
        end
    end
    return Iμ, Jμ
end

# TODO: figure out a better solution here; too much nesting and @inline required
@inline function withgtsorted(f::F, vs::NTuple{N,Any}, vmin) where {F,N}
    return (vmin <= first(vs)) ? f(vs) : withgtsorted(f, Base.tail(vs), vmin)
end
@inline withgtsorted(f::F, vs::NTuple{1}, vmin) where {F} = (vmin <= only(vs)) ? f(vs) : f(())

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
