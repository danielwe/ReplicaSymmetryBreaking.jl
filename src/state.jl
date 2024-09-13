## Define bidirectional map between order parameters and optimization state
initstate(problem::AbstractRSBProblem) = state(problem, Orderparameters(problem))

function state((; β)::AbstractRSBProblem{T}, (; a, sqrtdq)::Orderparameters{T}) where {T}
    if isinf(β)
        amax = a[begin]
        sqrtscaledda = sqrt.(diff([0; reverse(a) / amax]))
        dqangles = cartesian2polar(sqrtdq[2:end])
        return T[sqrt((acot(amax) / 2) / π); cartesian2polar(sqrtscaledda); dqangles]
    else
        sqrtscaledda = sqrt.(diff([0; reverse(a) / β; 1]))
        dqangles = cartesian2polar(sqrtdq)
        return T[cartesian2polar(sqrtscaledda); dqangles]
    end
end

function Orderparameters(
    problem::AbstractRSBProblem{T}, u::AbstractVector{T}; kwargs...
) where {T}
    (; k) = problem
    if k != (length(u) ÷ 2)
        error("mismatched state dimension for problem: k = $k, length(u) = $(length(u))")
    end
    op = Orderparameters(zeros(T, k), zeros(T, k + 2))
    updateorderparameters!(op, problem, u; kwargs...)
    return op
end

function updateorderparameters!(
    op::Orderparameters{T}, problem::AbstractRSBProblem{T}, u::AbstractVector{T}; check=true
) where {T}
    (; a, sqrtdq) = op
    (; k, β) = problem
    Base.require_one_based_indexing(a, sqrtdq, u)
    @views if isinf(β)
        amax = inv(tanpi(2(u[1]^2)))
        polar2reversecumsum!(a[2:end], u[2:k], amax)
        if k > 0
            a[1] = amax
        end
        sqrtdq[1] = zero(eltype(sqrtdq))
        polar2cartesian!(sqrtdq[2:end], u[(k + 1):end])
    else
        polar2reversecumsum!(a, u[1:k], β)
        polar2cartesian!(sqrtdq, u[(k + 1):end])
    end
    check && checkorderparameters(op, problem)
    return nothing
end

# Map between unit N-sphere and misc reprs of a simplex (N + 1 numbers with fixed sum)
function cartesian2polar(cartesian)
    return [
        atan(cartesian[i - 1], norm(@view cartesian[i:end])) / π
        for i in eachindex(cartesian)[(begin + 1):end]
    ]
end

function polar2cartesian!(cartesian, angles)
    # angles in units of radians / π
    # elementwise square of simplex => cartesian unit vector in N + 1 dims
    Base.require_one_based_indexing(cartesian, angles)
    N = length(angles)
    Np1 = N + 1
    (length(cartesian) == Np1) || error("mismatched lengths of cartesian and angles")
    cosprod = cospi(zero(eltype(angles)))
    if N > 0
        sini, cosi = sincospi(angles[1])
        cartesian[1] = cosprod * sini
        for i in 2:N
            cosprod *= cosi
            sini, cosi = sincospi(angles[i])
            cartesian[i] = cosprod * sini
        end
        cartesian[Np1] = cosprod * cosi
    else
        cartesian[1] = cosprod
    end
    return nothing
end

function polar2reversecumsum!(reversecumsum, angles, radius)
    # angles in units of radians / π
    # reverse cumsum of simplex => N numbers in (0, r) in descending order
    Base.require_one_based_indexing(reversecumsum, angles)
    N = length(angles)
    Np1 = N + 1
    (length(reversecumsum) == N) || error("mismatched lengths of reversecumsum and angles")
    if N > 0
        cosprodsq = radius
        sini, cosi = sincospi(angles[1])
        reversecumsum[N] = current = cosprodsq * (sini^2)
        for i in 2:N
            cosprodsq *= cosi^2
            sini, cosi = sincospi(angles[i])
            reversecumsum[Np1 - i] = current = current + cosprodsq * (sini^2)
        end
    end
    return nothing
end
