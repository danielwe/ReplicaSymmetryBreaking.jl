struct QuadGKPCHIP{T,R<:AbstractRange{T},N} <: AbstractRSBQuadGK{T}
    knots::R
    knotstup::NTuple{N,T}
    nodes::Vector{T}
    order::Int
    scale::T
    stretchmode::Symbol
    rtol::T

    function QuadGKPCHIP(nknots, order, scale::T; stretchmode=:power, rtol=64eps(T)) where {T}
        knots = range(zero(T), oneunit(T), nknots)
        knotstup = Tuple(knots)
        nodes = stretchmapplus.(knots[1:(end-1)], scale, stretchmode)
        N, R = length(knotstup), typeof(knots)
        return new{T,R,N}(knots, knotstup, nodes, order, scale, stretchmode, rtol)
    end
end

cutoff((; scale)::QuadGKPCHIP{T}) where {T} = copysign(scale, convert(T, Inf))

function transform!!(values::AbstractVector{T}, quad::QuadGKPCHIP{T}) where {T}
    return Interpolator(collect(quad.knots), [values; 0])
end

set_atolscale!(::QuadGKPCHIP{T}, ::T) where {T} = nothing
get_atol(::QuadGKPCHIP{T}) where {T} = zero(T)

# preallocating QuadGK segbuf seems not worth it
createbuffer(::QuadGKPCHIP) = nothing

function integrate(f::F, quad::QuadGKPCHIP, itp; kwargs...) where {F}
    (; scale, stretchmode) = quad
    return integrate(quad) do t
        ν = stretchmapplus(t, scale, stretchmode)
        Gν = itp(t)  # * (1 - t) / 2
        fν = f(ν, Gν)
        Jν = stretchjacplus(t, scale, stretchmode)
        return (t == oneunit(t)) ? (zero(fν) * zero(Jν)) : (fν * Jν)
    end
end

function integrate(integrand::F, quad::QuadGKPCHIP) where {F}
    (; knotstup, order, rtol) = quad
    norm(x) = LinearAlgebra.norm(x, Inf)
    return first(quadgk(integrand, knotstup...; order, rtol, norm))
end
