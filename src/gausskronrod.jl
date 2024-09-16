## Quadrature on (0, b)
struct GaussKronrod{T} <: AbstractRSBQuadrature{T}
    cutoff::T
    nodes::Vector{T}
    weights::Vector{T}
    gaussweights::Vector{T}
    tol::T
    abstolscale::Base.RefValue{T}

    function GaussKronrod(
        cutoff::T, n, scale=4one(T); tol=16eps(T), abstolscale=zero(T), stretchmode=:exp
    ) where {T}
        nodes_, weights_, gaussweights_ = kronrod(T, n)
        Base.require_one_based_indexing(nodes_, weights_, gaussweights_)
        nodes = [nodes_; -nodes_[(end - 1):-1:1]]  # (-1, 0] ∪ (0, 1) = (-1, 1)
        weights = [weights_; weights_[(end - 1):-1:1]]
        gaussweights = [gaussweights_; gaussweights_[fld(n, 2):-1:1]]
        inf = isinf(cutoff)
        if inf
            (scale > 0) || throw(ArgumentError("scale must be positive"))
            b = copysign(T(scale), cutoff)
            jac = stretchjac.(nodes, b, stretchmode)
            nodes .= stretchmap.(nodes, b, stretchmode)
            weights .*= jac
            gaussweights .*= jac[2:2:end]
        else
            b = cutoff / 2
            nodes .= (nodes .+ 1) .* b  # (0, cutoff)
            weights .*= abs(b)
            gaussweights .*= abs(b)
        end
        return new{T}(cutoff, nodes, weights, gaussweights, tol, Ref(abstolscale))
    end
end

getnode(quad::GaussKronrod, i) = quad.nodes[i]
cutoff(quad::GaussKronrod) = quad.cutoff

## Stretch t ∈ (-1, 1) to ν ∈ (0, Inf)
# Cancellation for ν near 0, but map must be smooth, so can't avoid -1 -> 0 or 1 -> 0
# :exp:   ν = -b * log((1 - t) / 2); dν = b * dt / (t - 1)
# :power: ν = b * (1 + t) / (1 - t); dν = 2b / (1 - t)^2
function stretchmap(t, b, mode)
    (mode == :exp) && return -b * log((1 - t) / 2)
    (mode == :power) && return b * (1 + t) / (1 - t)
    throw(ArgumentError("invalid mode: must be :exp or :power"))
end

function stretchjac(t, b, mode)
    (mode == :exp) && return abs(b) / (1 - t)
    (mode == :power) && return 2abs(b) / ((1 - t)^2)
    throw(ArgumentError("invalid mode: must be :exp or :power"))
end

## Integrate f(ν, args...) over (0, cutoff(quad))
function integrate(
    f::F, quad::GaussKronrod{T}, args...;
    integrandbuffer::AbstractVector{T}=similar(quad.weights)
) where {F,T}
    map!(f, integrandbuffer, quad.nodes, args...)
    I = dot(integrandbuffer, quad.weights)
    checkgausskronrodprecision(I, integrandbuffer, quad)
    return I
end

function checkgausskronrodprecision(I, integrandbuffer, quad)
    Base.require_one_based_indexing(integrandbuffer)
    @views Igauss = dot(integrandbuffer[2:2:end], quad.gaussweights)
    err, Iabs = abs(I - Igauss), abs(I)
    tol, scale = quad.tol, oftype(Iabs, quad.abstolscale[])
    if err >= (tol * max(Iabs, scale))
        throw(GaussKronrodPrecisionError(err / Iabs, err / scale, tol))
    end
    return nothing
end

EnzymeRules.inactive(::typeof(checkgausskronrodprecision), args...) = true

struct GaussKronrodPrecisionError{T} <: AbstractRSBException
    relerr::T
    scalederr::T
    tol::T
end

function Base.showerror(io::IO, (; relerr, scalederr, tol)::GaussKronrodPrecisionError)
    print(io, "Gauss-Kronrod error outside tolerance: ")
    print(io, "relative error = $relerr, scaled error = $scalederr, tolerance = $tol")
    return nothing
end

# ## Use Gauss-Chebyshev instead of Gauss-Legendre quadrature
# # much poorer convergence, but facilitates direct Chebyshev interpolation of values
# # change 2:2:end to 1:2:end in checkintegral if using this
# function chebyshevkronrod(T, n)
#     N = (3n + 3) ÷ 2
#     half = oneunit(T) / 2
#     jacobisuperdiag = fill(half, N - 1)
#     jacobisuperdiag[begin] = sqrt(half)
#     jacobimatrix = QuadGK.HollowSymTridiagonal(jacobisuperdiag)
#     x, w, gw = kronrod(jacobimatrix, n, π)
#     # Absorb weight function into weights. The kronrod rule includes the cutoffs, and this
#     # sets the corresponding weights to zero. This means we're assuming that the integrand
#     # does not have an integrable singularity at one or both cutoffs, where f(±1) diverges
#     # but f(x) * sqrt(1 - x^2) has a finite limit. For the purposes of this package, that's
#     # a valid assumption, but this should be kept in mind for more general applications.
#     w .*= sqrt.(1 .- x.^2)
#     @views gw .*= sqrt.(1 .- (x[2:2:end].^2))
#     @assert iszero(w[begin]) "w[begin] = $(w[begin])"  # Assert first weight zero
#     return x[2:end], w[2:end], gw
# end
