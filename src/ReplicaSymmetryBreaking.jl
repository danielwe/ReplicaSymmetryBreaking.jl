module ReplicaSymmetryBreaking

using DispatchDoctor: @stable, @unstable
using FastGaussQuadrature: gausslegendre

export SKProblem

@stable default_mode="error" default_codegen_level="min" begin

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

@unstable function mappedquad(F, values, (; nodes, weights)::Quadrature)  # integrate F.(nodes, values)
    return sum(
        i -> weights[i] * F(nodes[i], values[i]), eachindex(weights, nodes, values);
        init=zero(eltype(weights)) * zero(F(oneunit(eltype(nodes)), zero(eltype(values)))),
    )
end

abstract type AbstractRSBProblem{T<:AbstractFloat} end

struct RSBWorkspace{T<:AbstractFloat}
    a::Vector{T}
    q::Vector{T}
    dq::Vector{T}
    Gvs::Tuple{Vector{T},Vector{T}}

    function RSBWorkspace((; kRSB, β)::AbstractRSBProblem{T}, (; nodes)::Quadrature{T}) where {T}
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

@unstable function recursionkernel(μp, μabs, ai, dqi, expshift=nothing; normalized)
    dμp = μp - ai * dqi
    decay2 = -2ai * μabs
    expshift2 = isnothing(expshift) ? decay2 : expshift + decay2
    g1 = gaussian(μabs - dμp, dqi, expshift; normalized)
    g2 = gaussian(μabs + dμp, dqi, expshift2; normalized)
    return g1 + g2
end

@unstable function recursionmap_local(Gvalues, μabs, ai, dqi, quad)
    integrand = let μabs=μabs, ai=ai, dqi=dqi
        integrand(μp, Gμp) = recursionkernel(μp, μabs, ai, dqi, ai * Gμp; normalized=false)
    end
    Z = gaussZ(dqi)
    lastval = integrand(quad.nodes[end], Gvalues[end])
    I = if lastval > eps(typeof(lastval))
        # integrand is leaking outside quad's domain, so we return known asymptotic value
        oneunit(eltype(quad.weights)) * oneunit(lastval) / oneunit(Z)
    else
        mappedquad(integrand, Gvalues, quad) / Z
    end
    return log(I) / ai
end

@unstable function recursionmap!(Gvnext, Gv, ai, dqi, quad)
    map!(μ -> recursionmap_local(Gv, μ, ai, dqi, quad), Gvnext, quad.nodes)
    return nothing
end

function traceterm!((; a, q, dq, Gvs), rsbproblem, quad)
    GvA, GvB = Gvs
    map!(μ -> G0(rsbproblem, μ), GvA, quad.nodes)
    for i in eachindex(a, dq)
        recursionmap!(GvB, GvA, a[i], dq[i], quad)
        GvA, GvB = GvB, GvA
    end
    integrand = let habs=abs(heff(rsbproblem)), za=zero(eltype(a)), qlast=q[end]
        integrand(μ, Gμ) = (Gμ + μ) * recursionkernel(μ, habs, za, qlast; normalized=false)
    end
    return -mappedquad(integrand, GvA, quad) / gaussZ(q[end])
end

@views function rsbparams!((; a, q, dq), (; β), rsbweights)
    k = length(rsbweights) ÷ 2
    ξratios = rsbweights[begin:(begin + k - 1)]
    qratios = rsbweights[(begin + k):end]
    mappedcumprod!(a, sigmoid, invcompact, ξratios, compact(β))
    if isinf(β)  # q1 = 1 fixed => first qratio trivial and not included
        q[begin] = one(eltype(q))
        mappedcumprod!(q[(begin + 1):end], sigmoid, identity, qratios)
    else
        mappedcumprod!(q, sigmoid, identity, qratios)
    end
    dq .= q[begin:(end - 1)] .- q[(begin + 1):end]
    return nothing
end

@unstable function mappedcumprod!(out, fpre, fpost, x, init=one(fpre(oneunit(eltype(x)))))
    # out .= fpost.(cumprod(fpre.(x)) .* init), but without allocations
    prev = init
    for i in eachindex(out, x)
        this = fpre(x[i]) * prev
        out[i] = fpost(this)
        prev = this
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
    fieldterm = sum(
        a[i] * ((q[i] - 1)^2 - (q[i + 1] - 1)^2) for i in eachindex(a, q[begin:(end - 1)]);
        init=zero(eltype(a)) * zero(eltype(q))^2,
    ) / 4
    (; β, J0, M) = rsbproblem
    fieldterm += J0 * (M^2) / 2
    if !isinf(β)
        fieldterm -= β * ((1 - q[begin])^2) / 4
    end
    return fieldterm + traceterm!(rsbworkspace, rsbproblem, quad)
end

end  # @stable

end  # module
