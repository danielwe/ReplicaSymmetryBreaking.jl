module ReplicaSymmetryBreaking

using FastGaussQuadrature: gausslegendre
using LinearAlgebra: dot
using MappedArrays: mappedarray

export SKProblem

struct Quadrature{T}
    nodes::Vector{T}
    weights::Vector{T}

    function Quadrature{T}(nodes, weights; scale, shift) where {T}
        return new{T}((scale .* nodes) .+ shift, scale .* weights)
    end
end

function GaussLegendre(n, a, b)
    nodes, weights = gausslegendre(n)
    scale, shift = (b - a) / 2, (a + b) / 2
    return Quadrature{promote_type(typeof(a), typeof(b))}(nodes, weights; scale, shift)
end

quadrature(values, (; weights)::Quadrature) = dot(weights, values)

abstract type AbstractRSBProblem{T<:AbstractFloat} end

struct RSBWorkspace{T<:AbstractFloat}
    a::Vector{T}
    q::Vector{T}
    dq::Vector{T}
    Gvs::Tuple{Vector{T},Vector{T}}

    function RSBWorkspace(
        (; kRSB, β)::AbstractRSBProblem{T}, (; nodes)::Quadrature{T}
    ) where {T}
        nG = length(nodes)
        a = zeros(T, kRSB)
        q = zeros(T, kRSB + 1)
        dq = zeros(T, kRSB)
        Gvs = (zeros(T, nG), zeros(T, nG))
        return new{T}(a, q, dq, Gvs)
    end
end

gaussZ(var) = sqrt(2(π * var))

function gaussian(dx, var, expshift=nothing; normalized)
    gaussexp = -dx^2 / (2var)
    exponent = isnothing(expshift) ? gaussexp : expshift + gaussexp
    return normalized ? exp(exponent) : exp(exponent) / gaussZ(var)
end

function recursionkernel(μp, μabs, ai, dqi, expshift=nothing; normalized)
    dμp = μp - ai * dqi
    decay2 = -2ai * μabs
    expshift2 = isnothing(expshift) ? decay2 : expshift + decay2
    g1 = gaussian(μabs - dμp, dqi, expshift; normalized)
    g2 = gaussian(μabs + dμp, dqi, expshift2; normalized)
    return g1 + g2
end

function recursionmap_local(Gvalues, μabs, ai, dqi, quad)
    integrand(μp, Gμp) = recursionkernel(μp, μabs, ai, dqi, ai * Gμp; normalized=false)
    Z = gaussZ(dqi)
    lastval = integrand(quad.nodes[end], Gvalues[end])
    I = if lastval > eps(typeof(lastval))
        # integrand is leaking outside quad's domain, so return the known asymptotic value
        oneunit(eltype(quad.weights)) * oneunit(lastval) / oneunit(Z)
    else
        quadrature(mappedarray(integrand, quad.nodes, Gvalues), quad)
    end
    return log(I) / ai
end

function recursionmap!(Gvnext, Gv, ai, dqi, quad)
    map!(μ -> recursionmap_local(Gv, μ, ai, dqi, quad), Gvnext, quad.nodes)
    return nothing
end

function trace(Gv, habs, qlast, quad)
    za = zero(habs / oneunit(qlast))
    integrand(μ, Gμ) = (Gμ + μ) * recursionkernel(μ, habs, za, qlast; normalized=false)
    return -quadrature(mappedarray(integrand, quad.nodes, Gv), quad) / gaussZ(qlast)
end

function traceterm!((; a, q, dq, Gvs), rsbproblem, quad)
    GvA, GvB = Gvs
    map!(μ -> G0(rsbproblem, μ), GvA, quad.nodes)
    for i in eachindex(a, dq)
        recursionmap!(GvB, GvA, a[i], dq[i], quad)
        GvA, GvB = GvB, GvA
    end
    return trace(GvA, abs(heff(rsbproblem)), q[end], quad)
end

@views function rsbparams!((; a, q, dq), (; β), rsbweights)
    k = length(rsbweights) ÷ 2
    ξratios = mappedarray(sigmoid, rsbweights[begin:(begin + k - 1)])
    qratios = mappedarray(sigmoid, rsbweights[(begin + k):end])
    cumprodf!(invcompact, a, ξratios, compact(β))
    if isinf(β)  # q1 = 1 fixed => first qratio trivial and not included
        q[begin] = one(eltype(q))
        cumprodf!(identity, q[(begin + 1):end], qratios)
    else
        cumprodf!(identity, q, qratios)
    end
    dq .= q[begin:(end - 1)] .- q[(begin + 1):end]
    return nothing
end

function cumprodf!(f, y, x, init=one(eltype(x)))  # y .= f.(cumprod(x) .* init)
    current = init
    for i in eachindex(y, x)
        current *= x[i]
        y[i] = f(current)
    end
    return nothing
end

compact(a) = isinf(a) ? one(a) : a / (1 + a)
invcompact(ξ) = isone(ξ) ? oftype(one(ξ), Inf) : ξ / (1 - ξ)
sigmoid(x) = inv(1 + exp(-x))
invsigmoid(x) = -log(inv(x) - 1)

@views function initrsbweights((; kRSB, β)::AbstractRSBProblem)
    ξ0 = compact(β)
    nq = kRSB + (isinf(β) ? 2 : 3)  # q1 = 1 fixed => discard q0
    ξ = (range(ξ0, zero(ξ0), kRSB + 2)[begin:(end - 1)]).^inv(sqrt(2))
    q = sin.(range(π / 2, 0.0, nq)[begin:(end - 1)])
    ξratios = invsigmoid.(ξ[(begin + 1):end] ./ ξ[begin:(end - 1)])
    qratios = invsigmoid.(q[(begin + 1):end] ./ q[begin:(end - 1)])
    return [ξratios; qratios]
end

struct SKProblem{T} <: AbstractRSBProblem{T}
    kRSB::Int
    β::T
    h::T
    J0::T
    M::T
end

# Base.@kwdef-like constructors with more user-friendly conversion behavior
SKProblem{T}(; kRSB, β, h=0, J0=0, M=0) where {T} = SKProblem{T}(kRSB, β, h, J0, M)
SKProblem(; β::T, kwargs...) where {T} = SKProblem{T}(; β, kwargs...)

function G0((; β)::SKProblem{T}, μabs::T) where {T}
    isinf(β) && return zero(T)
    return log1p(exp(-2β * μabs)) / β
end

heff((; h, J0, M)::SKProblem) = abs(h + J0 * M)

@views function freeenergy!(rsbworkspace, rsbproblem::SKProblem, rsbweights, quad)
    rsbparams!(rsbworkspace, rsbproblem, rsbweights)
    (; a, q) = rsbworkspace
    (; β, J0, M) = rsbproblem
    qpotentials = mappedarray(
        (qi, qj) -> (1 - qi)^2 - (1 - qj)^2, q[begin:(end - 1)], q[(begin + 1):end]
    )
    fieldterm = dot(a, qpotentials) / 4
    fieldterm += J0 * (M^2) / 2
    if !isinf(β)
        fieldterm -= β * ((1 - q[begin])^2) / 4
    end
    return fieldterm + traceterm!(rsbworkspace, rsbproblem, quad)
end

end  # module
