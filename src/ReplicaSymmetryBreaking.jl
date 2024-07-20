module ReplicaSymmetryBreaking

using DispatchDoctor: @stable
using FastGaussQuadrature: gausslegendre

@stable begin

abstract type AbstractRSBProblem{T<:AbstractFloat} end

struct Quadrature{T}
    nodes::Vector{T}
    weights::Vector{T}

    function Quadrature{T}(nodes, weights; scale, shift) where {T}
        return new{T}((scale .* nodes) .+ shift, scale .* weights)
    end
end

function Quadrature(::typeof(gausslegendre), n, a, b)
    nodes, weights = gausslegendre(n)
    scale, shift = (b - a) / 2, (a + b) / 2
    return Quadrature{promote_type(typeof(a), typeof(b))}(nodes, weights; scale, shift)
end

struct RSBCache{T<:AbstractFloat}
    a::Vector{T}
    q::Vector{T}
    dq::Vector{T}
    Gs::Tuple{Vector{T},Vector{T}}

    function RSBCache(problem::AbstractRSBProblem{T}, quad::Quadrature{T}) where {T}
        (; kRSB, β) = problem
        na = kRSB + 1
        nq = isinf(β) ? na : (na + 1)
        nG = length(quad.nodes)
        a = zeros(T, na)
        q = zeros(T, nq)
        dq = zeros(T, nq - 1)
        Gs = (zeros(T, nG), zeros(T, nG))
        return new{T}(a, q, dq, Gs)
    end
end

struct SKProblem{T} <: AbstractRSBProblem{T}
    kRSB::Int
    β::T
    h::T
    J0::T
    M::T
end

# Manual @kwdef-like constructors with more user-friendly conversion behavior
SKProblem{T}(; kRSB, β, h=0, J0=0, M=0) where {T} = SKProblem{T}(kRSB, β, h, J0, M)
SKProblem(; β, kwargs...) = SKProblem{float(typeof(β))}(; β, kwargs...)

function G0((; β)::SKProblem{T}, μ::T) where {T}
    isinf(β) && return zero(T)
    return log1p(exp(-2β * abs(μ))) / β
end

end  # @stable

end  # module
