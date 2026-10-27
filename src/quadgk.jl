struct QuadGK{T,transformtype,A<:AbstractInterpolator{T}} <: AbstractQuadrature{T}
    order::Int
    rtol::T
    approximator::A

    function QuadGK(
        order, approximator::A; rtol=eps(T)^(5 // 6), transformtype::Symbol=:geometric
    ) where {T,A<:AbstractInterpolator{T}}
        if !(rtol > 0)
            throw(ArgumentError("require rtol > 0, got rtol = $rtol"))
        end
        return new{T,transformtype,A}(order, rtol, approximator)
    end
end

# getters required for declaring fields inactive even if they contain floats
# not needed for order::Int, but included anyway for interface consistency
_order(quad::QuadGK) = quad.order
_rtol(quad::QuadGK) = quad.rtol
_approximator(quad::QuadGK) = quad.approximator

EnzymeRules.inactive(::typeof(_rtol), args...) = true
EnzymeRules.inactive(::typeof(_approximator), args...) = true

limits(quad::QuadGK) = limits(_approximator(quad))

function gettransform(quad::QuadGK{T,transformtype}) where {T,transformtype}
    return IntervalToRplus{transformtype}(limits(_approximator(quad))..., one(T) / 2)
end

function approximate(gvectorized::G, quad::QuadGK) where {G}
    return approximate(gvectorized, _approximator(quad))
end

approximantiscallable(::QuadGK) = true

function integratewithapproximant(
    F::FF, gapprox::G, quad::QuadGK, args::Vararg{Any,N}; kwargs...
) where {FF,G,N}
    return integrate(t -> F(t, gapprox(t)), quad, args...; kwargs...)
end

function integrate(f::F, quad::QuadGK{T}, breakpoints=(); scale::T=zero(T)) where {F,T}
    a, b = limits(quad)
    order, rtol = _order(quad), _rtol(quad)
    atol = rtol * scale
    I, E = quadgk(f, a, breakpoints..., b; order, rtol, atol, norm=_norminf)
    # check convergence manually because quadgk will return without error or warning if
    # if maxevals is reached before convergence
    if !((E <= atol) || (E <= (rtol * _norminf(I))))
        throw(QuadGKConvergenceError())
    end
    return I
end

_norminf(x) = norm(x, Inf)

struct QuadGKConvergenceError <: AbstractRSBException end

function Base.showerror(io::IO, ::QuadGKConvergenceError)
    print(io, "QuadGK did not converge; reached max number of integrand evaluations")
    return nothing
end

