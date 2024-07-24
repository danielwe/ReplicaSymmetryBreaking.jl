module ReplicaSymmetryBreaking

using Enzyme: Enzyme
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logistic, logit
using MappedArrays: mappedarray
using Optimization:
    OptimizationFunction, OptimizationProblem, AutoEnzyme, MaxSense, SciMLBase, solve
using OptimizationNLopt: NLopt

export SKProblem, solve_orderparameters

### RSB problem types

abstract type AbstractRSBProblem{T<:AbstractFloat} end

## RSB problems must implement methods for the following functions
function fieldterm end  # fieldterm(::AbstractRSBProblem, (; a, q))
function G0 end         # G0(::AbstractRSBProblem, μ >= 0): trace initial condition
function heff end       # heff(::AbstractRSBProblem) >= 0 : trace effective field

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

G0((; β)::SKProblem{T}, μ::T) where {T} = log1pexp(-2β * μ) / β

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

struct RSBSolver{T,RSBProblem<:AbstractRSBProblem{T}}
    problem::RSBProblem
    quad::Quadrature{T}
    a::Vector{T}
    q::Vector{T}
    dq::Vector{T}
    Gs::Tuple{Vector{T},Vector{T}}

    function RSBSolver(problem::AbstractRSBProblem{T}, quad::Quadrature{T}) where {T}
        (; k) = problem
        nG = length(quad.nodes)
        a = zeros(T, k)
        q = zeros(T, k + 1)
        dq = zeros(T, k)
        Gs = (zeros(T, nG), zeros(T, nG))
        return new{T,typeof(problem)}(problem, quad, a, q, dq, Gs)
    end
end

function RSBSolver(problem::AbstractRSBProblem{T}; nquad, μmax) where {T}
    return RSBSolver(problem, GaussLegendre(T, nquad, 0.0, μmax))
end

## Define bidirectional map between RSB parameters and optimization state
function getstate((; problem, a, q)::RSBSolver)
    ξfull = [a2ξ(problem.β); a2ξ.(a)]
    qfull = if isinf(problem.β)
        (q[begin] == 1) || error("Inconsistent order parameters: β == Inf but q[begin] != 1")
        q
    else
        [1; q]
    end
    return getstate(ξfull, qfull)
end

@views function getstate(ξfull, qfull)
    ξratios = invactivation.(ξfull[(begin + 1):end] ./ ξfull[begin:(end - 1)])
    qratios = invactivation.(qfull[(begin + 1):end] ./ qfull[begin:(end - 1)])
    return [ξratios; qratios]
end

@views function update!((; problem, a, q, dq)::RSBSolver, state)
    k = length(state) ÷ 2
    ξratios = mappedarray(activation, state[begin:(begin + k - 1)])
    qratios = mappedarray(activation, state[(begin + k):end])
    cumprodf!(ξ2a, a, ξratios, a2ξ(problem.β))
    if length(q) == length(qratios)
        cumprodf!(identity, q, qratios)
    else
        # β == Inf => q[1] == 1 => first qratio trivial and not included (no need to check
        # that lengths match, cumprodf! will error otherwise)
        q[begin] = 1
        cumprodf!(identity, q[(begin + 1):end], qratios)
    end
    # Use map! instead of broadcasting here to elide an aliasing check and thus eliminate an
    # allocating codepath (the path would never be taken, but could trigger allocation
    # detection in unit tests, so better to avoid it)
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

a2ξ(a) = isinf(a) ? one(a) : a / (1 + a)
ξ2a(ξ) = isone(ξ) ? oftype(one(ξ), Inf) : ξ / (1 - ξ)

# logistic transform, no bounds
activation(weight) = logistic(weight)
invactivation(ratio) = logit(ratio)
bounds(state) = (; lb=nothing, ub=nothing)

# no transform, hard bounds
# activation(weight) = identity(weight)
# invactivation(ratio) = identity(ratio)
# bounds(state) = (; lb=zero.(state), ub=oneunit.(state))

### Free energy

freeenergy(solver::RSBSolver) = fieldterm(solver.problem, solver) + traceterm(solver)

function traceterm((; problem, quad, a, q, dq, Gs))
    Gnext, G = Gs
    map!(μ -> G0(problem, μ), G, quad.nodes)
    for i in eachindex(a, dq)
        Gmap!(Gnext, G, (; ai=a[i], dqi=dq[i]), quad)
        Gnext, G = G, Gnext
    end
    return trace(G, heff(problem), q[end], quad)
end

function trace(G, h, qlast, quad)
    adqlast = (; ai=zero(h / oneunit(qlast)), dqi=qlast)
    integrand(ν, Gν) = (Gν + ν) * Gmapkernel(ν, h, adqlast; normalized=false)
    return -quadrature(mappedarray(integrand, quad.nodes, G), quad) / gaussnorm(qlast)
end

function Gmap!(Gnext, G, adqi, quad)
    map!(μ -> Gmap_point(μ, G, adqi, quad), Gnext, quad.nodes)
    return nothing
end

function Gmap_point(μ, G, adqi, quad)
    (; ai, dqi) = adqi
    Z = gaussnorm(dqi)
    integrand(ν, Gν) = Gmapkernel(ν, μ, adqi, ai * Gν; normalized=false)
    endpointvalue = abs(integrand(quad.nodes[end], G[end]))
    I = if endpointvalue > eps(typeof(endpointvalue))  # TODO: adapt threshold based on tol?
        # integrand is leaking outside quad's domain, so return the known asymptotic value
        oneunit(eltype(quad.weights)) * oneunit(endpointvalue) / oneunit(Z)
    else
        quadrature(mappedarray(integrand, quad.nodes, G), quad)
    end
    return log(I) / ai
end

function Gmapkernel(ν, μ, (; ai, dqi), expshift=nothing; normalized)
    dν = ν - ai * dqi
    decay2 = -2ai * μ
    expshift2 = isnothing(expshift) ? decay2 : expshift + decay2
    g1 = gaussian(μ - dν, dqi, expshift; normalized)
    g2 = gaussian(μ + dν, dqi, expshift2; normalized)
    return g1 + g2
end

function gaussian(dx, var, expshift=nothing; normalized)
    gaussexp = -dx^2 / (2var)
    exponent = isnothing(expshift) ? gaussexp : expshift + gaussexp
    return normalized ? exp(exponent) : exp(exponent) / gaussnorm(var)
end

gaussnorm(var) = sqrt(2(π * var))

### Optimization

function solve_orderparameters(problem::AbstractRSBProblem; nquad=26, μmax=5.0, kwargs...)
    solver = RSBSolver(problem; nquad, μmax)
    state = solve_state(solver; kwargs...)
    update!(solver, state)
    return (; solver.a, solver.q)
end

function solve_state(solver::RSBSolver; tol=16.0^(-11), showresult=(), io=stdout)
    f = OptimizationFunction(objective, AutoEnzyme())
    u0 = initstate(solver.problem)
    prob = OptimizationProblem(f, u0, solver; sense=MaxSense, bounds(u0)...)
    result = solve(prob, NLopt.LD_LBFGS(); abstol=tol, reltol=tol)
    if !SciMLBase.successful_retcode(result)
        error("Optimization failed with retcode $(result.retcode)")
    end
    for i in eachindex(showresult)
        if i != firstindex(showresult)
            println(io)
        end
        show(io, MIME"text/plain"(), getproperty(result, showresult[i]))
        println(io)
    end
    return result.u
end

function objective(state, solver)
    update!(solver, state)
    return freeenergy(solver)
end

@views function initstate((; k, β)::AbstractRSBProblem{T}) where {T}
    ξ0 = a2ξ(β)
    nq = k + (isinf(β) ? 2 : 3)  # β = Inf => q1 = 1 => discard q0 = 1
    ξrange = range(ξ0, zero(ξ0), k + 2)[begin:(end - 1)]
    qrange = range(oneunit(T), zero(T), nq)[begin:(end - 1)]
    # simple heuristics for a better initial guess (monotonic, 0 -> 0, 1 -> 1)
    ξfull = map(x -> x^inv(sqrt(2)), ξrange)
    qfull = map(x -> sinpi(x / 2), qrange)
    return getstate(ξfull, qfull)
end

end  # module
