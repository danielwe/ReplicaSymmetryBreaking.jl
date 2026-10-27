struct CIMProblem{T,inf} <: AbstractDisorderProblem{T,inf}
    k::Int
    β::T
    p::T

    function CIMProblem(k, β::T, p::T) where {T}
        if k < 0
            throw(ArgumentError("require k > 0, got k = $k"))
        end
        if !(β > 1)
            throw(ArgumentError("require β > 1, got k = $β"))
        end
        return new{T,β == Inf}(k, β, p)
    end
end

CIMProblem(; k, β, p) = CIMProblem(k, promote(β, p)...)

rsborder(problem::CIMProblem) = problem.k
inversetemp(problem::CIMProblem{T,inf}) where {T,inf} = inf ? convert(T, Inf) : problem.β
externalfield(::CIMProblem{T}) where {T} = zero(T)

# Differences to take into account:
# * q0 != 1, it's another optimization variable
#   * note that this messes with the non-equilibrium susceptibility term
#   * however, q1 → q0 as β → ∞ is likely still true!
# * inner integral probably not analytically solvable (@Schmidt2008Replica Appendix A.2)
#   * note that this potentially also messes with the non-equilibrium susceptibility term
# * asymptotic renormalization group has different asymptotics
#   * note that this messes with the field term, which in the Ising/quasi-isotropic case was
#     modified by pulling out field-only factors from the rewritten trace term
# * analytic integral parts I1 and I2 in Gnext may work out differently or not at all

# OK, here's the conclusion from hours of analytics and asymptotics. We need:
# * G0 ✔
# * asymptotic part ✔
# * field term ✔
# * free energy functions that are less specialized on SK problem, building on general
#   functions for G0 and asymptotic part
#   - note that this implies a rewrite of the SK field term
# * order parameters where q0 is variable
# * extra order parameter t
# * b_i = b_0 + \sum_{j = 0}^{i - 1} a_i dq_i
#   - possibly also part of the order parameter struct?

# b = p + t, where p is the gain parameter and t is the extra order parameter
_Emf(x, b, μ) = (((x^2 - 2b) * x / 4) - μ) * x

# Asymptotic behavior of -log(C(μ)) / β → min_x _Emf(x, b, μ)
# where C(μ) = sum_i exp(-β * _Emf(x_i, b, μ)) and the x_i are local minima of _Emf(x, b, μ)
ℰ(::CIMProblem, b, μ) = -((3(cbrt(μ)^2) + b)^2 + b^2) / 12

function dℰ(::CIMProblem, bnext, μnext, b, μ)
    μp1_3 = cbrt(μ)
    μnextp1_3 = cbrt(μnext)
    sb, db = b + bnext, b - bnext
    sμp2_3, dμp2_3 = μp1_3^2 + μnextp1_3^2, (μp1_3 + μnextp1_3) * (μp1_3 - μnextp1_3)
    return ((3sμp2_3 + sb) * (3dμp2_3 + db) + sb * db) / 12
    # or just
    # return ℰ(problem, bnext, μnext) - ℰ(problem, b, μ) TODO: or is it the other way around?
end

# Kernel correction initializer: G0(μ, t) = ℰ(p + t, μ) + log(C(μ)) / β
function G0(problem::CIMProblem{T,inf}, μ::T, t::T) where {T,inf}
    # need minima of _Emf, which correspond to stable (outer, upward-crossing) zeros of the
    # depressed cubic dEmf(x) = x^3 - b * x - μ
    # then evaluate log(sum(-β (ℰ - E_mf(xi))) / β at these values
    b = problem.p + t
    # special cases
    if iszero(b)      # ℰ = -(3/4) μ^(4/3), single minimum at cbrt(μ) where Emf = ℰ => ℰ - Emf = 0
        return cbrt(zero(μ))^4
    elseif iszero(μ)  # ℰ = -b^2 / 6
        if b < zero(b)  # single minimum at the origin where Emf = 0 => ℰ - Emf = -b^2 / 6
            return -b^2 / 6
        else            # symmetric minima at ±sqrt(b) where Emf = -b^2 / 4 => ℰ - Emf = b^2 / 12
            dEmf = b^2 / 12
            if inf
                return dEmf
            else # don't forget the entropy term
                β = inversetemp(problem)
                return dEmf + oftype(dEmf, log(2one(dEmf)) / β)
            end
        end
    end
    # general cases
    u = -b / 3
    v = -μ / 2
    Δ = u^3 + v^2
    if Δ > 0          # single minimum
        C = cbrt(v + flipsign(sqrt(Δ), v))
        return ℰ(problem, b, μ) - _Emf(C - (u / C), b, μ)
    elseif iszero(Δ)
        # single minimum at 3μ / b (the other critical point is a saddle at -3μ / 2b)
        # ℰ = -((3μ^(2/3) + b)^2 + b^2) / 12
        # Emf = -2b^2 / 3
        # ℰ - Emf = (7b^2 - (3μ^(2/3) + b)^2) / 12
        return (7b^2 - (3(cbrt(μ)^2) + b)^2) / 12
    else              # two minima
        s = sqrt(-u)
        s2 = 2s
        θ_3 = acos((-v) / (s^3)) / 3  # ∈ (0, π / 3)
        π2_3 = 2oftype(θ_3, π) / 3
        if inf  # only deepest minimum matters
            x0 = if μ > 0  # negative is deepest
                s2 * cos(θ_3 + π2_3)
            else          # positive is deepest
                s2 * cos(θ_3)
            end
            return ℰ(b, μ) - _Emf(x0, b, μ)
        else    # fully general case
            ϕs = (θ_3 + π2_3, θ_3)
            β = inversetemp(problem)
            Emf∞ = ℰ(problem, b, μ)
            exponents = β .* (Emf∞ .- _Emf.(s2 .* cos.(ϕs), b, μ))
            return logaddexp(exponents...) / β
        end
    end
end

# (1/2) q_0 t + (1/4) sum_i=0^κ a_i (q_i^2 - q_i+1^2)
# == (1/2) q_0 t + (1/4) sum_i=0^κ a_i (q_i + q_i+1)(q_i - q_i+1)
function fieldterm(problem::CIMProblem{T,inf}, op::Orderparameters{T}) where {T,inf}
    a, sqrtdq, q0, t = blockparams(op), overlapparams(op), maxoverlap(op), inducedgain(op)
    dq0 = first(sqrtdq)^2  # q_0 - q_1
    qi = if inf            # q_1
        @assert iszero(dq0)
        q0
    else
        q1 = q0 - dq0
        q1
    end
    cdqi = zero(dq0)       # q_1 - q_1
    sqrtdqbulk = @view sqrtdq[(begin+1):(end-1)]  # √(q_1 - q_2):√(q_κ - q_κ+1)
    overlapterm_m = zero(eltype(a)) * (zero(typeof(dq0))^2)
    for i in eachindex(a, sqrtdqbulk)
        # the first dqs _after dq0_ are the smallest, so minimize accumulated error by
        # iterating in order but keeping dq0 outside the accumulator
        dqi = sqrtdqbulk[i]^2   # q_i - q_i+1
        cdqi += dqi             # q_1 - q_i+1
        qinext = if inf         # q_i+1
            q0 - cdqi
        else
            # don't use q1 here because q0 - (dq0 + dqci) has less rounding error than
            # (q0 - dq0) - dqci when dq0 is small
            q0 - (dq0 + cdqi)
        end
        overlapterm_m = muladd(a[i], dqi * (qi + qinext), overlapterm_m)  # += a_i (q_i - q_i+1)(q_i + q_i+1)
        qi = qinext
    end
    if !inf
        β = inversetemp(problem)
        overlapterm_m = muladd(β, dq0 * (q0 + q1), overlapterm_m)         # += β * (q_0 - q_1)(q_0 + q_1)
    end
    return (overlapterm_m / 4) + (q0 * t / 2)
end

struct CIMReplicaStepper{T,P<:CIMProblem{T},O<:Orderparameters{T}}
    problem::P
    op::O
end

Base.length((; op)::CIMReplicaStepper) = length(op)
Base.eltype(::CIMReplicaStepper{T}) where {T} = Tuple{T,T,T}

@inline function Base.iterate(rstepper::CIMReplicaStepper)
    (; problem, op) = rstepper
    opi_s = iterate(op)
    b0 = problem.p + first(op.mfparams)  # TODO: make sure this is consistent, i.e., b0 = p + t
    return iterate(rstepper, (opi_s, b0))
end

@inline function Base.iterate((; op)::CIMReplicaStepper{T}, (opi_s, bi)) where {T}
    isnothing(opi_s) && return nothing
    ((ai, sqrtdqi), s) = opi_s
    return (ai::T, sqrtdqi::T, bi::T), (iterate(op, s), bi + ai * sqrtdqi)
end

function rsbstepper(problem::CIMProblem{T}, op::Orderparameters{T}) where {T}
    return CIMReplicaStepper(problem, op)
end

struct CIMOrderparameters{T}
    rsbop::Orderparameters{T}
    q0::T
    b0::T
end

Base.length((; rsbop)::CIMOrderparameters) = length(rsbop)
Base.eltype(::CIMOrderparameters{T}) where {T} = Tuple{T,T,T}

@inline function Base.iterate(op::CIMOrderparameters)
    (; rsbop, b0) = op
    rsb0_s = iterate(rsbop)
    return iterate(op, (rsb0_s, b0))
end

@inline function Base.iterate(op::CIMOrderparameters{T}, (rsbi_s, bi)) where {T}
    isnothing(rsbi_s) && return nothing
    ((ai, sqrtdqi), s) = rsbi_s
    return (ai::T, sqrtdqi::T, bi::T), (iterate(op, s), bi + ai * sqrtdqi)
end
