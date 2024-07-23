module ReplicaSymmetryBreaking

using Enzyme: Enzyme
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: dot
using MappedArrays: mappedarray
using Optimization:
    OptimizationFunction, OptimizationProblem, AutoEnzyme, MaxSense, SciMLBase, solve
using OptimizationNLopt: NLopt

export SKProblem

### RSB problem types

abstract type AbstractRSBProblem{T<:AbstractFloat} end

## RSB problems must implement methods for the following functions
function fieldterm end  # fieldterm(::AbstractRSBProblem, (; a, q))
function G0 end         # G0(::AbstractRSBProblem, μabs) : trace initial condition
function heff end       # heff(::AbstractRSBProblem)     : trace effective field

struct SKProblem{T} <: AbstractRSBProblem{T}
    k::Int
    β::T
    h::T
    J0::T
    M::T
end

# Base.@kwdef-like constructors with more user-friendly conversion behavior
SKProblem{T}(; k, β, h=0, J0=0, M=0) where {T} = SKProblem{T}(k, β, h, J0, M)
SKProblem(; β::T, kwargs...) where {T} = SKProblem{T}(; β, kwargs...)

@views function fieldterm((; β, J0, M)::SKProblem, (; a, q))
    overlappotentials = mappedarray(
        (qi, qj) -> (1 - qi)^2 - (1 - qj)^2, q[begin:(end - 1)], q[(begin + 1):end]
    )
    overlapterm = dot(a, overlappotentials) + 2(J0 * (M^2))
    if isinf(β)
        return overlapterm / 4
    end
    return (overlapterm - β * ((1 - q[begin])^2)) / 4
end

function G0((; β)::SKProblem{T}, μabs::T) where {T}
    isinf(β) && return zero(T)
    return log1p(exp(-2β * μabs)) / β
end

heff((; h, J0, M)::SKProblem) = abs(h + J0 * M)

### Quadrature for trace term

struct Quadrature{T<:AbstractFloat}
    nodes::Vector{T}
    weights::Vector{T}

    function Quadrature{T}(nodes, weights; scale, shift) where {T}
        return new{T}((scale .* nodes) .+ shift, scale .* weights)
    end
end

function GaussLegendre(T, n, a, b)
    nodes, weights = gausslegendre(n)
    scale, shift = (b - a) / 2, (a + b) / 2
    return Quadrature{T}(nodes, weights; scale, shift)
end

GaussLegendre(n, a, b) = GaussLegendre(promote_type(typeof(a), typeof(b)), n, a, b)

quadrature(values, (; weights)::Quadrature) = dot(weights, values)

## RSB workspace

struct RSBWorkspace{T<:AbstractFloat}
    a::Vector{T}
    q::Vector{T}
    dq::Vector{T}
    Gvs::Tuple{Vector{T},Vector{T}}
end

function RSBWorkspace((; k, β)::AbstractRSBProblem{T}, (; nodes)) where {T}
    (T == eltype(nodes)) || error("incompatible eltypes between RSB problem and quadrature")
    nG = length(nodes)
    a = zeros(T, k)
    q = zeros(T, k + 1)
    dq = zeros(T, k)
    Gvs = (zeros(T, nG), zeros(T, nG))
    return RSBWorkspace{T}(a, q, dq, Gvs)
end

## Define bidirectional map between RSB parameters and optimization state
function optimizationstate((; a, q)::RSBWorkspace, β)
    ξfull = [a2ξ(β); a2ξ.(a)]
    # q[begin] == 1 => first qratio trivial and not included
    qfull = (q[begin] == 1) ? q : [1; q]
    return ξq2optimizationstate(ξfull, qfull)
end

@views function ξq2optimizationstate(ξfull, qfull)
    ξratios = invactivation.(ξfull[(begin + 1):end] ./ ξfull[begin:(end - 1)])
    qratios = invactivation.(qfull[(begin + 1):end] ./ qfull[begin:(end - 1)])
    return [ξratios; qratios]
end

@views function update!((; a, q, dq)::RSBWorkspace, state, β)
    k = length(state) ÷ 2
    ξratios = mappedarray(activation, state[begin:(begin + k - 1)])
    qratios = mappedarray(activation, state[(begin + k):end])
    cumprodf!(ξ2a, a, ξratios, a2ξ(β))
    if length(q) == length(qratios)
        cumprodf!(identity, q, qratios)
    else  # β == Inf => q[begin] == 1 => first qratio trivial and not included
          # (no need to check that lengths match, cumprodf! will error otherwise)
        q[begin] = one(eltype(q))
        cumprodf!(identity, q[(begin + 1):end], qratios)
    end
    # Using map! instead of broadcasting here to elide an aliasing check and thus eliminate
    # an allocating codepath (the path would never be taken in practice, but can trigger
    # allocation detection in unit tests, so might as well avoid it)
    map!(-, dq, q[begin:(end - 1)], q[(begin + 1):end])
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

a2ξ(a) = compactify(a)
ξ2a(ξ) = invcompactify(ξ)
activation(weight) = sigmoid(weight)             # identity(ratio)
invactivation(ratio) = invsigmoid(ratio)         # identity(ratio)
upperbounds(state) = nothing  # oneunit.(state)  # should match activation
lowerbounds(state) = nothing  # zero.(state)     # should match activation

compactify(x) = isinf(x) ? one(x) : x / (1 + x)
invcompactify(x) = isone(x) ? oftype(one(x), Inf) : x / (1 - x)
sigmoid(x) = inv(1 + exp(-x))
invsigmoid(x) = -log(inv(x) - 1)

### Free energy

function freeenergy(problem, workspace, quad)
    return fieldterm(problem, workspace) + traceterm(problem, workspace, quad)
end

function traceterm(problem, (; a, q, dq, Gvs), quad)
    Gvnext, Gv = Gvs
    map!(μ -> G0(problem, μ), Gv, quad.nodes)
    for i in eachindex(a, dq)
        recursionmap!(Gvnext, Gv, a[i], dq[i], quad)
        Gvnext, Gv = Gv, Gvnext
    end
    return trace(Gv, abs(heff(problem)), q[end], quad)
end

function trace(Gv, habs, qlast, quad)
    za = zero(habs / oneunit(qlast))
    integrand(μ, Gμ) = (Gμ + μ) * recursionkernel(μ, habs, za, qlast; normalized=false)
    return -quadrature(mappedarray(integrand, quad.nodes, Gv), quad) / gaussnorm(qlast)
end

function recursionmap!(Gvnext, Gv, ai, dqi, quad)
    map!(μ -> recursionmap_local(Gv, μ, ai, dqi, quad), Gvnext, quad.nodes)
    return nothing
end

function recursionmap_local(Gvalues, μabs, ai, dqi, quad)
    integrand(μp, Gμp) = recursionkernel(μp, μabs, ai, dqi, ai * Gμp; normalized=false)
    Z = gaussnorm(dqi)
    lastval = integrand(quad.nodes[end], Gvalues[end])
    I = if lastval > eps(typeof(lastval))  # TODO: adapt threshold to tol?
        # integrand is leaking outside quad's domain, so return the known asymptotic value
        oneunit(eltype(quad.weights)) * oneunit(lastval) / oneunit(Z)
    else
        quadrature(mappedarray(integrand, quad.nodes, Gvalues), quad)
    end
    return log(I) / ai
end

function recursionkernel(μp, μabs, ai, dqi, expshift=nothing; normalized)
    dμp = μp - ai * dqi
    decay2 = -2ai * μabs
    expshift2 = isnothing(expshift) ? decay2 : expshift + decay2
    g1 = gaussian(μabs - dμp, dqi, expshift; normalized)
    g2 = gaussian(μabs + dμp, dqi, expshift2; normalized)
    return g1 + g2
end

function gaussian(dx, var, expshift=nothing; normalized)
    gaussexp = -dx^2 / (2var)
    exponent = isnothing(expshift) ? gaussexp : expshift + gaussexp
    return normalized ? exp(exponent) : exp(exponent) / gaussnorm(var)
end

gaussnorm(var) = sqrt(2(π * var))

### Optimization

function solve_rsbparams(
    problem::AbstractRSBProblem{T}; nquad=500, μmax=(20.0 / 3), kwargs...
) where {T}
    quad = GaussLegendre(T, nquad, 0.0, μmax)
    workspace = RSBWorkspace(problem, quad)
    state = solve_state(problem, workspace, quad; kwargs...)
    update!(workspace, state, problem.β)
    return (; workspace.a, workspace.q)
end

function solve_state(problem, workspace, quad; tol=16.0^(-11))
    f = OptimizationFunction{true}(rsbobjective, AutoEnzyme())
    u0 = initoptimizationstate(problem)
    p = (problem, workspace, quad)
    prob = OptimizationProblem(
        f, u0, p; sense=MaxSense, ub=upperbounds(u0), lb=lowerbounds(u0)
    )
    sol = solve(prob, NLopt.LD_LBFGS(); abstol=tol, reltol=tol)
    if !SciMLBase.successful_retcode(sol)
        error("Optimization failed with retcode $(sol.retcode)")
    end
    return sol.u
end

function rsbobjective(state, (problem, workspace, quad))
    update!(workspace, state, problem.β)
    return freeenergy(problem, workspace, quad)
end

@views function initoptimizationstate((; k, β)::AbstractRSBProblem)
    nq = k + (isinf(β) ? 2 : 3)  # q1 = 1 fixed => discard q0
    ξ0 = a2ξ(β)
    # These maps from linear ranges are simple heuristics that get us a little closer to the
    # general shape of the solution
    ξfull = (range(ξ0, zero(ξ0), k + 2)[begin:(end - 1)]).^inv(sqrt(2))
    qfull = sin.(range(π / 2, 0.0, nq)[begin:(end - 1)])
    return ξq2optimizationstate(ξfull, qfull)
end

end  # module
