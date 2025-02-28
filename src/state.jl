## Define bidirectional map between order parameters and optimization state
initstate(problem::AbstractRSBProblem) = state(problem, Orderparameters(problem))

function state(
    (; β)::AbstractRSBProblem{T,inf},
    (; a, sqrtdq)::Orderparameters{T},
) where {T,inf}
    ξ = applyinverse.((atransform,), a)
    # sqrtdξ = sqrt.(.-diff([1; ξ; 0]))
    # daangles = reverse(cartesian2polar(reverse(sqrtdξ)))
    dξ = reverse(diff([0; reverse(ξ)]))
    # da = reverse(diff([0; reverse(a)]))
    # dξ = applyinverse.((atransform,), da)
    dqangles = if inf
        cartesian2polar(@view sqrtdq[2:end])
    else
        cartesian2polar(sqrtdq)
    end
    # return [daangles; dqangles]
    return [dξ; dqangles]
    # sqrtdqvalues = if inf
    #     @view sqrtdq[2:end]
    # else
    #     sqrtdq
    # end
    # return [dξ; sqrtdqvalues]
    # return [dξ; sqrtdqvalues.^2]
    # if inf
    #     amax = a[begin]
    #     sqrtscaledda = sqrt.(.-diff([a / amax; 0]))
    #     daangles = reverse(cartesian2polar(reverse(sqrtscaledda)))
    #     dqangles = cartesian2polar(sqrtdq[(begin+1):end])
    #     return T[applyinverse(atransform, amax); daangles; dqangles]
    # else
    #     sqrtscaledda = sqrt.(.-diff([1; a / β; 0]))
    #     daangles = reverse(cartesian2polar(reverse(sqrtscaledda)))
    #     dqangles = cartesian2polar(sqrtdq)
    #     return T[daangles; dqangles]
    # end
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

function augment_state(u::AbstractVector{T}) where {T}
    k = fld(length(u), 2)
    # u_a = @view u[1:k]
    # a = similar(u_a)
    # @views begin
    #     cumsumfpost!(atransform, a[end:-1:1], u_a[end:-1:1])
    # end
    # new_a = [a; a[end] / 2]
    # new_ξ = applyinverse.((atransform,), new_a)
    # new_u_a = reverse(diff([0; reverse(new_ξ)]))
    new_u_a = [u[1:(k-1)]; u[k] / 2; u[k] / 2]
    new_u_sqrtdq = [u[(k+1):(end-1)]; u[(end-1):end]]
    return [new_u_a; new_u_sqrtdq]
end

# tanpi bypasses sincospi but doesn't have a dedicated rule => autodiff digs straight into
# its guts; screw that, what we really need is cotpi anyway, so let's define our own using
# sincospi, that way the sincospi rule will apply and everyone is happy
# function cotpi(x)
#     s, c = sincospi(x)
#     return c / s
# end

# const atransform = Halfstretch{:power}(0, 1, 1)
const atransform = Halfstretch{:power}(0, 1 // 2, 1 // 2)
# const atransform = Halfstretch{:power}(0, 1 // 2, 1)
# const atransform = Halfstretch{:power}(0, 1 / 2, π / 2)
# const atransform = Halfstretch{:tanpi}(0, 1 // 2, 1)
# const atransform = Halfstretch{:tanpi}(0, 1 / 2, 1 / π)
# const atransform = Expm1(1)
# const atransform = DoubleExpm1(1)
# const atransform = Sinh(1)
# const atransform = DoubleSinh(1)
# const atransform = Square(1 // 2)
# const atransform = Identity()

function updateorderparameters!(
    op::Orderparameters{T}, problem::AbstractRSBProblem{T,inf}, u::AbstractVector{T};
    check=true, kwargs...,
) where {T,inf}
    (; a, sqrtdq) = op
    Base.require_one_based_indexing(a, sqrtdq, u)
    (; k) = problem
    @views begin
        # polar2cartesiancumsumsq!(a[end:-1:1], u[k:-1:1])
        # map!(atransform, a, a)
        # cumsum!(a[end:-1:1], u[k:-1:1])
        # cumsumfpost!(atransform, a[end:-1:1], u[k:-1:1])
        # cumsumfpre!(atransform, a[end:-1:1], u[k:-1:1])
        for i in eachindex(a, u[1:k])
            # this would be a stupid way to do a reverse cumsum if it weren't for for the
            # fact that u[1:k] is ordered from smallest to largest, so this preserves more
            # accuracy at the expense of doing seemingly redundant work
            a[i] = atransform(foldl(+, u[i:k]))
        end
        # nrm = norm(u[(k+1):end])  # project onto manifold => not relying on the equality constraint
        # # nrmsq = sum(u[(k+1):end])  # project onto manifold => not relying on the equality constraint
        # if inf
        #     sqrtdq[1] = zero(eltype(sqrtdq))
        #     sqrtdq[2:end] .= u[(k+1):end] ./ nrm
        # else
        #     sqrtdq .= u[(k+1):end] ./ nrm
        # end
        if inf
            sqrtdq[1] = zero(eltype(sqrtdq))
            polar2cartesian!(sqrtdq[2:end], u[(k+1):end])
        else
            polar2cartesian!(sqrtdq, u[(k+1):end])
        end
    end
    # (; k, β) = problem
    # @views if inf
    #     amax = apply(atransform, u[1])
    #     polar2cartesiancumsumsq!(a[end:-1:2], u[k:-1:2], amax)
    #     if k > 0
    #         a[1] = amax
    #     end
    #     sqrtdq[1] = zero(eltype(sqrtdq))
    #     polar2cartesian!(sqrtdq[2:end], u[(k+1):end])
    # else
    #     polar2cartesiancumsumsq!(a[end:-1:1], u[k:-1:1], β)
    #     polar2cartesian!(sqrtdq, u[(k+1):end])
    # end
    check && checkorderparameters(op, problem; kwargs...)
    return nothing
end

# function cumsumfpre!(f::F, y, x) where {F}
#     c = zero(eltype(y))
#     for i in eachindex(y, x)
#         c += f(x[i])
#         y[i] = c
#     end
#     return nothing
# end

function cumsumfpost!(f::F, y, x) where {F}
    c = zero(eltype(x))
    for i in eachindex(y, x)
        c += x[i]
        y[i] = f(c)
    end
    return nothing
end

# Map between unit N-sphere and misc reprs of a simplex (N + 1 numbers with fixed sum)
function cartesian2polar(cartesian)
    return [
        atan(cartesian[i-1], norm(@view cartesian[i:end])) / π
        for i in eachindex(cartesian)[(begin+1):end]
    ]
end

function polar2cartesian!(cartesian, angles)
    # angles in units of radians / π
    # elementwise square of simplex => cartesian unit vector in N + 1 dims
    Base.require_one_based_indexing(cartesian, angles)
    N = length(angles)
    (length(cartesian) == (N + 1)) || error("mismatched lengths of cartesian and angles")
    cosprod = cospi(zero(eltype(angles)))
    if N > 0
        sini, cosi = sincospi(angles[1])
        cartesian[1] = cosprod * sini
        cosprod *= cosi
        for i in 2:N
            sini, cosi = sincospi(angles[i])
            cartesian[i] = cosprod * sini
            cosprod *= cosi
        end
    end
    cartesian[end] = cosprod
    nrm = norm(cartesian)
    if !isone(nrm)  # rescale to compensate for accumulated floating point error
        map!(c -> c / nrm, cartesian, cartesian)
    end
    return nothing
end

function polar2cartesiancumsumsq!(
    cartesiancumsumsq, angles, radiussq=oneunit(eltype(cartesiancumsumsq))
)
    # angles in units of radians / π
    # reverse cumsum of simplex => N numbers in (0, r) in descending order
    Base.require_one_based_indexing(cartesiancumsumsq, angles)
    N = length(angles)
    (length(cartesiancumsumsq) == N) || error("mismatched lengths of cartesiancumsum and angles")
    if N > 0
        cosprodsq = radiussq
        sini, cosi = sincospi(angles[1])
        cartesiancumsumsq[1] = current = cosprodsq * (sini^2)
        cosprodsq *= cosi^2
        for i in 2:N
            sini, cosi = sincospi(angles[i])
            cartesiancumsumsq[i] = current = current + cosprodsq * (sini^2)
            cosprodsq *= cosi^2
        end
        nrmsq = current + cosprodsq
        if nrmsq != radiussq  # rescale to compensate for accumulated floating point error
            factor = radiussq / nrmsq
            map!(csq -> csq * factor, cartesiancumsumsq, cartesiancumsumsq)
        end
    end
    return nothing
end
