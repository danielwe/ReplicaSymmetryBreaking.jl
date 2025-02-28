## Gaussians
# expshift is an optional term added to the exponent
function gaussian(dx, std, expshift=nothing; normalized)
    y = dx / std
    weight = isnothing(expshift) ? exp(-y^2 / 2) : expmuladd(-y, y / 2, expshift)
    return normalized ? (weight * gaussnorm(std)) : weight
end

# Evaluate Gaussian at x and -x and add up
function wrappedgaussian(x, mean, std; normalized)
    return gaussian(mean - x, std; normalized) + gaussian(mean + x, std; normalized)
end

# Gaussian normalization constant
gaussnorm(std) = inv(sqrt(2oftype(one(std), π)) * std)

# Helper to calculate exp(x * y + z) with maximum precision
function expmuladd(x, y, z)
    return exp(x * y) * exp(z)
    return exp(muladd(x, y, z))
    # TODO:
    # XXX: make a decision here: use the custom variant or not?
    #
    # p = x * y
    # w = muladd(x, y, z)
    # if (
    #     (abs(p) < log(nextfloat(zero(p)))) || (abs(p) > log(floatmax(p)))     # exp(p) over/underflow
    #     || (abs(z) < log(nextfloat(zero(z)))) || (abs(z) > log(floatmax(z)))  # exp(z) over/underflow
    #     || (abs(w) <= 1)  # error in + dominated by error in exp
    #     # better to only call exp once and accept the error in +
    # )
    #     return exp(w)
    # else
    #     return exp(p) * exp(z)
    # end
end

# invoke quadrature with breakpoints that ensure arbitrarily narrow gaussians are resolved
@inline function integralwithgaussian(f::F, μ, σ, quad, transform=Identity(); kwargs...,) where {F}
    @assert μ >= 0
    (σ > eps(μ)) || throw(GaussianWidthError(μ, σ))
    return _integralwithbreakpoints(f, 2σ, (μ - σ, μ + σ), (), quad, transform; kwargs...)
end


@inline function integralwithsplitgaussian(  # split gaussian is two gaussians of the same width
    f::F, μ1, μ2, σ, quad, transform=Identity(); kwargs...,
) where {F}
    (μ1 == μ2) && return integralwithgaussian(f, μ1, σ, quad, transform; kwargs...)
    @assert μ1 >= 0
    @assert μ2 >= 0
    μl, μr = (μ1 < μ2) ? (μ1, μ2) : (μ2, μ1)
    (σ > eps(μr)) || throw(GaussianWidthError(μr, σ))
    dμ = μr - μl
    I = if dμ >= 4σ  # μr >= 4σ
        _integralwithbreakpoints(
            f, 2σ, (μl - σ, μl + σ), (μr - σ, μr + σ), quad, transform; kwargs...
        )
    elseif dμ >= 2σ  # μr >= 2σ, μm >= σ
        μm = (μl + μr) / 2
        _integralwithbreakpoints(f, 2σ, (μl - σ, μm), (μr + σ,), quad, transform; kwargs...)
    else
        _integralwithbreakpoints(f, 2σ, (μl - σ, μr + σ), (), quad, transform; kwargs...)
    end
    return I
end

@inline function _integralwithbreakpoints(
    f::F, bpmin::T, bpl::NTuple{2,T}, bpr::NTuple{N,T}, quad, transform; kwargs...
) where {F,T,N}
    @assert bpl[1] < bpl[2]
    if N > 0
        @assert bpmin < bpr[1]
        @assert bpl[2] < bpr[1]
        ntuple(Val(N - 1)) do i
            @inline
            @assert bpr[i] < bpr[i+1]
            nothing
        end
    end
    I = if bpl[2] < bpmin
        let breakpoints = applyinverse.((transform,), bpr)
            integrate(f, quad, breakpoints; kwargs...)
        end
    elseif bpl[1] < bpmin
        let breakpoints = applyinverse.((transform,), (bpl[2], bpr...))
            integrate(f, quad, breakpoints; kwargs...)
        end
    else
        let breakpoints = applyinverse.((transform,), (bpl..., bpr...))
            integrate(f, quad, breakpoints; kwargs...)
        end
    end
    return I
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

struct GaussianWidthError{T} <: AbstractRSBException
    mean::T
    std::T
end

function Base.showerror(io::IO, (; mean, std)::GaussianWidthError)
    print(io, "Gaussian too narrow to resolve: μ = $mean, σ = $std")
end

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
