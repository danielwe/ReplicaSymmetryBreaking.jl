## Quadrature on (0, b)
struct GaussKronrod{T} <: AbstractRSBQuadrature{T}
    cutoff::T
    nodes::Vector{T}
    weights::Vector{T}
    gaussweights::Vector{T}
    tol::T
    atolscale::Base.RefValue{T}

    function GaussKronrod(
        cutoff::T, order, scale=4one(T); tol=16eps(T), atolscale=zero(T), stretchmode=:exp
    ) where {T}
        nodes_, weights_, gaussweights_ = kronrod(T, order)
        Base.require_one_based_indexing(nodes_, weights_, gaussweights_)
        nodes = [nodes_; -nodes_[(end-1):-1:1]]  # (-1, 0] ∪ (0, 1) = (-1, 1)
        weights = [weights_; weights_[(end-1):-1:1]]
        gaussweights = [gaussweights_; gaussweights_[fld(order, 2):-1:1]]
        if isinf(cutoff)
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
        return new{T}(cutoff, nodes, weights, gaussweights, tol, Ref(atolscale))
    end
end

## Integrate f(ν, args...) over (0, cutoff(quad))
createbuffer(quad::GaussKronrod) = similar(quad.weights)

function integrate(
    f::F, quad::GaussKronrod{T}, args...; quadbuffer::AbstractVector{T}=createbuffer(quad)
) where {F,T}
    map!(f, quadbuffer, quad.nodes, args...)
    I = dot(quadbuffer, quad.weights)
    checkgausskronrodprecision(I, quadbuffer, quad)
    return I
end

function checkgausskronrodprecision(I, quadbuffer, quad)
    Base.require_one_based_indexing(quadbuffer)
    @views Igauss = dot(quadbuffer[2:2:end], quad.gaussweights)
    err, Iabs = abs(I - Igauss), abs(I)
    tol, scale = quad.tol, oftype(Iabs, quad.atolscale[])
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
