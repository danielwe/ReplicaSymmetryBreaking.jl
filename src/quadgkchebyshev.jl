struct QuadGKApproxFun{T,C,P,N<:AbstractVector{T}} <: AbstractRSBQuadGK{T}
    space::C
    transform::P
    nodes::N
    order::Int
    scale::T
    stretchmode::Symbol
    rtol::T
    atol::Ref{T}

    function QuadGKApproxFun(
        nknots, order, scale::T; stretchmode=:power, rtol=64eps(T)
    ) where {T}
        # TODO: this only works for finite cutoff, need to figure out stretch
        # lb, ub = (cutoff >= 0) ? (zero(cutoff), cutoff) : (cutoff, zero(cutoff))
        space = Chebyshev(-oneunit(T) .. oneunit(T))
        knots = ApproxFun.points(space, nknots)
        transform = ApproxFunBase.plan_transform(space, knots)
        nodes = stretchmap.(reverse(knots), scale, stretchmode)
        C, P, N = typeof.((space, transform, nodes))
        return new{T,C,P,N}(space, transform, nodes, order, scale, stretchmode, rtol, Ref(zero(T)))
    end
end

cutoff((; scale)::QuadGKApproxFun{T}) where {T} = copysign(scale, convert(T, Inf))

function transform!!(values::AbstractVector{T}, quad::QuadGKApproxFun{T}) where {T}
    # TODO: write custom rule for in-place transform to avoid allocation here
    # currently importing rule through ChainRulesCore by adding some missing dispatches
    reverse!(values)
    coeffs = quad.transform * values
    checkapproxfunprecision!(coeffs)
    return coeffs
end

# preallocating QuadGK segbuf seems not worth it
createbuffer(::QuadGKApproxFun) = nothing

function integrate(f::F, quad::QuadGKApproxFun, coeffs::AbstractVector, midpoints::Tuple=(); kwargs...) where {F}
    (; scale, stretchmode, space) = quad
    return integrate(quad, invstretchmap.(midpoints, scale, stretchmode)) do t
        ν = stretchmap(t, scale, stretchmode)
        Gν = Fun(space, coeffs)(t)
        fν = f(ν, Gν)
        Jν = stretchjac(t, scale, stretchmode)
        return (t == oneunit(t)) ? (zero(fν) * zero(Jν)) : (fν * Jν)
    end
end

function integrate(integrand::F, quad::QuadGKApproxFun, midpoints::Tuple=()) where {F}
    (; order, space, rtol, atol) = quad
    lb, ub = endpoints(space.domain)
    norm(x) = LinearAlgebra.norm(x, Inf)
    return first(quadgk(integrand, lb, midpoints..., ub; order, rtol, atol=atol[], norm))
end

function checkapproxfunprecision!(coeffs)
    iN = lastindex(coeffs)
    last_ = abs(coeffs[iN])
    max_ = maximum(abs, coeffs)
    eps_ = eps(max_)
    if last_ >= 2eps_
        throw(QuadGKApproxFunPrecisionError(last_, max_, length(coeffs)))
    else
        for i in iN:-1:firstindex(coeffs)
            if 2abs(coeffs[i]) >= eps_  # last coeff to keep
                coeffs[(i+1):end] .= zero(eltype(coeffs))
                break
            end
        end
    end
    return nothing
end

struct QuadGKApproxFunPrecisionError{T} <: AbstractRSBException
    lastcoeff::T
    maxcoeff::T
    funorder::Int
end

function Base.showerror(
    io::IO, (; lastcoeff, maxcoeff, funorder)::QuadGKApproxFunPrecisionError
)
    print(io, "Inexact ApproxFun of order $funorder: ")
    print(io, "maxcoeff = $maxcoeff, lastcoeff = $lastcoeff; ")
    print(io, "lastcoeff / maxcoeff = $(lastcoeff / maxcoeff)")
    return nothing
end

# let quadgkcounter=ScopedValues.ScopedValue{Threads.Atomic{Int}}()
#     global function countintegrandcalls(f)
#         ncalls = Threads.Atomic{Int}(0)
#         returns = ScopedValues.@with quadgkcounter => ncalls f()
#         return ncalls[], returns
#     end
#
#     global function quadgkwrapped(f, lb, ub; atol, rtol, order, segbuf=nothing)
#         counter = ScopedValues.get(quadgkcounter)
#         return if isnothing(counter)
#             I, E = quadgk(f, lb, ub; atol, rtol, order, segbuf)
#             checkquadgk(I, E, atol, rtol, order)
#             I, E
#         else
#             I, E, count = quadgk_count(f, lb, ub; atol, rtol, order, segbuf)
#             checkquadgk(I, E, atol, rtol, order)
#             Threads.atomic_add!(something(counter), count)
#             I, E
#         end
#     end
#
#     function checkquadgk(I, E, atol, rtol, order)
#         tol = max(atol, rtol * norm(I))
#         if E >= tol
#             error(
#                 "quadgk did not converge\n"
#                 * "final value = $I\n"
#                 * "final error estimate = $E\n"
#                 * "atol = $atol\n"
#                 * "rtol = $rtol)\n"
#                 * "max(atol, rtol * norm(I)) = $tol\n"
#                 * "order = $order\n"
#                 * "integrand call count = $count"
#             )
#         end
#         return nothing
#     end
#
#     EnzymeRules.inactive(::typeof(checkquadgk), args...) = true
# end
