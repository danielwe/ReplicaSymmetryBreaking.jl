# this mixes two different concepts, the quadrature and the function approximation which
# should be kept more separate---you can for example use fixed-order Guass-Kronrod
# quadrature with any function approximation, even though you cannot use a vector of values
# sampled at fixed Gauss-Kronrod nodes as function approximation for any quadrature
struct QuadGKAAA{T,stretch} <: AbstractRSBQuadGK{T,stretch}
    order::Int
    rtol::T
    function QuadGKAAA{T}(order; rtol=64eps(), stretch=:power) where {T}
        @assert rtol > 0
        return new{T,stretch}(order, rtol)
    end
end

QuadGKAAA(args...; kwargs...) = QuadGKAAA{Float64}(args...; kwargs...)

getorder(quad::AbstractRSBQuadGK) = quad.order
getrtol(quad::AbstractRSBQuadGK) = quad.rtol
EnzymeRules.inactive(::typeof(getrtol), args...) = true

function gettransform(quad::AbstractRSBQuadrature{T,stretch}) where {T,stretch}
    return Halfstretch{stretch}(getlimits(quad)..., one(T))
end

getlimits(::QuadGKAAA{T}) where {T} = (zero(T), oneunit(T))
EnzymeRules.inactive(::typeof(getlimits), args...) = true

approximate(f::F, quad::QuadGKAAA) where {F} = aaa(f, getlimits(quad)...)

@inline function integrate(
    f::F, quad::AbstractRSBQuadGK{T}, breakpoints=(); scale::T=zero(T)
) where {F,T}
    order = getorder(quad)
    a, b = getlimits(quad)
    rtol = getrtol(quad)
    atol = rtol * scale
    qnorm(x) = norm(x, Inf)
    I, E = quadgk(f, a, breakpoints..., b; order, rtol, atol, norm=qnorm)
    # check convergence manually because quadgk can return without error or warning even if
    # it didn't converge, for example if maxevals is reached
    if !((E <= atol) || (E <= (rtol * qnorm(I))))
        throw(QuadGKConvergenceError())
    end
    return I
end

struct QuadGKConvergenceError <: AbstractRSBException end

function Base.showerror(io::IO, ::QuadGKConvergenceError)
    print(io, "QuadGK did not converge; reached max number of integrand evaluations")
    return nothing
end
