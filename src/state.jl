### Bijective map between order parameters and optimization state
function state(::AbstractDisorderProblem{T,inf}, op::Orderparameters{T}) where {T,inf}
    # this direction does not matter for performance, so we don't worry about allocations
    a, sqrtdq = blockparams(op), overlapparams(op)
    ξ = applyinverse.((blockparamtransform(),), a)
    dξ = reverse(diff([0; reverse(ξ)]))
    dqangles = if inf
        cartesian2polar(@view sqrtdq[2:end])
    else
        cartesian2polar(sqrtdq)
    end
    return [dξ; dqangles]
end

# TODO: keep or scrap this, see bottom of optimization.jl
# function augment_state(u::AbstractVector{T}) where {T}
#     k = fld(length(u), 2)
#     # u_a = @view u[1:k]
#     # a = similar(u_a)
#     # @views begin
#     #     cumsumfpost!(blockparamtransform(), a[end:-1:1], u_a[end:-1:1])
#     # end
#     # new_a = [a; a[end] / 2]
#     # new_ξ = applyinverse.((blockparamtransform(),), new_a)
#     # new_u_a = reverse(diff([0; reverse(new_ξ)]))
#     new_u_a = [u[1:(k-1)]; u[k] / 2; u[k] / 2]
#     new_u_sqrtdq = [u[(k+1):(end-1)]; u[(end-1):end]]
#     return [new_u_a; new_u_sqrtdq]
# end

function orderparameters(problem::AbstractDisorderProblem{T}, u::AbstractVector{T}; kwargs...) where {T}
    k = rsborder(problem)
    if k != (length(u) ÷ 2)
        error("mismatched state dimension for problem: k = $k, length(u) = $(length(u))")
    end
    op = Orderparameters(Vector{T}(undef, k), Vector{T}(undef, k + 2))
    updateorderparameters!(op, problem, u; kwargs...)
    return op
end

function updateorderparameters!(
    op::Orderparameters{T}, problem::AbstractDisorderProblem{T,inf}, u::AbstractVector{T};
    check=true, kwargs...,
) where {T,inf}
    # this direction is performance sensitive, so we avoid allocations and redundant work
    a, sqrtdq = blockparams(op), overlapparams(op)
    Base.require_one_based_indexing(a, sqrtdq, u)
    k = rsborder(problem)
    tf = blockparamtransform()
    @views begin
        for i in eachindex(a, u[1:k])
            # we're doing seemingly redundant work here, as it would be possible to iterate
            # in the other direction and accumulate in a partial sum; however, near the
            # optimum, `a` is more or less ordered from smallest to largest, so we'd rather
            # minimize the accumulated error by always summing from left to right
            a[i] = apply(tf, foldl(+, u[i:k]))
        end
        if inf
            sqrtdq[1] = zero(eltype(sqrtdq))
            polar2cartesian!(sqrtdq[2:end], u[(k+1):end])
        else
            polar2cartesian!(sqrtdq, u[(k+1):end])
        end
    end
    check && checkorderparameters(op, problem; kwargs...)
    return nothing
end

# map between [0, 1 / 2) and [0, Inf)
blockparamtransform() = IntervalToRplus{:geometric}(0, 1 // 2, 2)

# map between cartesian and polar representation of a point on a unit N-sphere
function cartesian2polar(cartesian)
    return [
        atan(cartesian[i-1], norm(@view cartesian[i:end])) / π
        for i in eachindex(cartesian)[(begin+1):end]
    ]
end

function polar2cartesian!(cartesian, polar)
    # NOTE: angles in units of radians / π, i.e., arguments to sinpi and cospi
    Base.require_one_based_indexing(cartesian, polar)
    N = length(polar)
    (length(cartesian) == (N + 1)) || error("mismatched lengths of cartesian and polar")
    cosprod = cospi(zero(eltype(polar)))
    if N > 0
        sini, cosi = sincospi(polar[1])
        cartesian[1] = cosprod * sini
        cosprod *= cosi
        for i in 2:N
            sini, cosi = sincospi(polar[i])
            cartesian[i] = cosprod * sini
            cosprod *= cosi
        end
    end
    cartesian[end] = cosprod
    # rescale to compensate for any accumulated floating point error
    # the norm is always ≈ 1, so under-/overflow is not an issue; we therefore use a simple
    # and naive norm calculation rather than the robust but complex `norm`
    nrmsq = sum(abs2, cartesian)
    if !isone(nrmsq)
        nrm = sqrt(nrmsq)
        map!(c -> c / nrm, cartesian, cartesian)
    end
    return nothing
end

# bounds and constraints on the state
function statebounds(u, problem)
    u_L = fill!(similar(u), 0)
    u_U = similar(u)
    Base.require_one_based_indexing(u_U)
    k = rsborder(problem)
    u_U[1:k] .= Inf  # upper bound for u_U[1:k] not needed; handled by constraint
    u_U[(k+1):end] .= 1 // 2
    return u_L, u_U
end

function constraintf!(c, u, problem)
    Base.require_one_based_indexing(u)
    k = rsborder(problem)
    # sum left-to-right for minimal precision loss since u[1:k] goes small-to-large near the optimum
    c[begin] = foldl(+, @view u[1:k])
    return nothing
end

function dconstraintf!(dc, _, _)
    fill!(dc, 1)
    return nothing
end

dconstraintlength(problem) = rsborder(problem)
dconstraintfisconstant(_) = true

function constraintvalues(problem::AbstractDisorderProblem{T}) where {T}
    tf = blockparamtransform()
    β = inversetemp(problem)
    return T[-Inf], T[applyinverse(tf, β)]  # lower bound for constraint function not needed; handled by bounds
end
