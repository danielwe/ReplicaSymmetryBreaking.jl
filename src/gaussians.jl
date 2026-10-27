### Gaussian utilities
function gaussian(dx, std; normalized)
    weight = exp(-(dx / std)^2 / 2)
    return normalized ? (weight * gaussnorm(std)) : weight
end

function wrappedgaussian(x, mean, std; normalized)
    weightl = gaussian(mean - x, std; normalized=false)
    weightr = gaussian(mean + x, std; normalized=false)
    weight = weightl + weightr
    return normalized ? (weight * gaussnorm(std)) : weight
end

# normalization constant
gaussnorm(std) = inv(sqrt(2oftype(one(std), π)) * std)

## apply a function to a tuple of inflection points pertaining to a Gaussian
@inline function withgaussianinflectionpoints(f::F, μ, σ) where {F}
    (eps(μ) < σ) || throw(GaussianWidthError(μ, σ))
    return f((μ - σ, μ + σ))
end

@inline function withsplitgaussianinflectionpoints(f::F, μ1, μ2, σ) where {F}
    # split Gaussian is two gaussians with the same width
    # if more than 4σ separation, return all inflection points
    # otherwise, if more than 2σ separation, return outer inflection points and midpoint
    # otherwise, return outer inflection points only
    # thus, returned points are never closer than 2σ
    (μ1 == μ2) && return withgaussianinflectionpoints(f, μ1, σ)
    μl, μr = (μ1 < μ2) ? (μ1, μ2) : (μ2, μ1)
    μmax = argmax(abs, (μl, μr))
    (eps(μmax) < σ) || throw(GaussianWidthError(μmax, σ))
    dμ = μr - μl
    return if dμ < 2σ
        f((μl - σ, μr + σ))
    elseif dμ < 4σ
        μm = (μl + μr) / 2
        f((μl - σ, μm, μr + σ))
    else
        f((μl - σ, μl + σ, μr - σ, μr + σ))
    end
end

struct GaussianWidthError{T} <: AbstractRSBException
    mean::T
    std::T
end

function Base.showerror(io::IO, (; mean, std)::GaussianWidthError)
    print(io, "Gaussian too narrow to resolve: μ = $mean, σ = $std")
end

## Validate that Gaussian kernel does not leak beyond cutoff
# TODO: is this still relevant?
# function checkgaussianleak(mean, std, cutoff)
#     (isfinite(mean) && isinf(cutoff)) && return nothing
#     abscutoff = abs(cutoff)
#     denom = sqrt(2one(std)) * std
#     zright, zleft = (abscutoff - mean) / denom, (abscutoff + mean) / denom
#     mass = (erfc(zright) + erfc(zleft)) / 2
#     (mass < (eps(typeof(mass)) / 2)) || throw(GaussianLeakError(cutoff, mean, std, mass))
#     return nothing
# end
#
# EnzymeRules.inactive(::typeof(checkgaussianleak), args...) = true
#
# struct GaussianLeakError{T} <: AbstractRSBException
#     cutoff::T
#     mean::T
#     std::T
#     mass::T
# end
#
# function Base.showerror(io::IO, (; cutoff, mean, std, mass)::GaussianLeakError)
#     print(io, "Gaussian kernel leaking beyond $cutoff; mean = $mean, std = $std, ")
#     print(io, "leaked mass = $mass")
#     return nothing
# end
