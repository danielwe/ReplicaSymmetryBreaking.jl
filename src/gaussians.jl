## Gaussians
# expshift is an optional term added to the exponent
function gaussian(dx, std, expshift=nothing; normalized)
    y = dx / std
    numerator = isnothing(expshift) ? exp(-y^2 / 2) : expmuladd(-y, y / 2, expshift)
    return normalized ? (numerator / gaussnorm(std)) : numerator
end

# Evaluate Gaussian at x and -x and add up
function wrappedgaussian(x, mean, std; normalized)
    return gaussian(mean - x, std; normalized) + gaussian(mean + x, std; normalized)
end

# Gaussian normalization constant
gaussnorm(std) = sqrt(2oftype(one(std), π)) * std

# Helper to calculate exp(x * y + z) with maximum precision
function expmuladd(x, y, z)
    p = x * y
    w = muladd(x, y, z)
    if (
        (abs(p) < log(nextfloat(zero(p)))) || (abs(p) > log(floatmax(p)))     # exp(p) over/underflow
        || (abs(z) < log(nextfloat(zero(z)))) || (abs(z) > log(floatmax(z)))  # exp(z) over/underflow
        || (abs(w) <= 1)  # error in + dominated by error in exp
                          # better to only call exp once and accept the error in +
    )
        return exp(w)
    else
        return exp(p) * exp(z)
    end
end

## Validate that Gaussian kernel does not leak beyond cutoff
function checkgaussianleak(mean, std, cutoff)
    (isfinite(mean) && isinf(cutoff)) && return nothing
    abscutoff = abs(cutoff)
    denom = sqrt(2one(std)) * std
    zright, zleft = (abscutoff - mean) / denom, (abscutoff + mean) / denom
    mass = (erfc(zright) + erfc(zleft)) / 2
    (mass < (eps(typeof(mass)) / 2)) || throw(GaussianLeakError(cutoff, mean, std, mass))
    return nothing
end

EnzymeRules.inactive(::typeof(checkgaussianleak), args...) = true

struct GaussianLeakError{T} <: AbstractRSBException
    cutoff::T
    mean::T
    std::T
    mass::T
end

function Base.showerror(io::IO, (; cutoff, mean, std, mass)::GaussianLeakError)
    print(io, "Gaussian kernel leaking beyond $cutoff; mean = $mean, std = $std, ")
    print(io, "leaked mass = $mass")
    return nothing
end
