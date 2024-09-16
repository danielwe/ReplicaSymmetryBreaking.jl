## Preallocated buffers for calculating the trace term
struct Tracebuffers{T,ntasks}
    kernelbuffers::Tuple{Vector{T},Vector{T}}
    integrandbuffers::NTuple{ntasks,Vector{T}}

    function Tracebuffers(T, N; ntasks)
        return new{T,ntasks}((zeros(T, N), zeros(T, N)), ntuple(_ -> zeros(T, N), ntasks))
    end
end

function _make_zero!(op::Tracebuffers)
    foreach(_make_zero!, op.kernelbuffers)
    foreach(_make_zero!, op.integrandbuffers)
    return nothing
end

## Free energy calculation
function freeenergy(
    problem::AbstractRSBProblem{T},
    quad::GaussKronrod{T},
    op::Orderparameters{T},
    buffers::Tracebuffers{T},
) where {T}
    return fieldterm(problem, op) + traceterm(problem, quad, op, buffers)
end

function traceterm(problem, quad, (; a, sqrtdq), (; kernelbuffers, integrandbuffers))
    Gnext, G = kernelbuffers
    map!(i -> G0(problem, getnode(quad, i)), G, eachindex(G))
    sqrtdqbulk = @view sqrtdq[(begin + 1):(end - 1)] 
    for i in eachindex(a, sqrtdqbulk)
        ai, sqrtdqi = a[i], sqrtdqbulk[i]
        Gmap!(Gnext, G, ai, sqrtdqi, quad, integrandbuffers)
        Gnext, G = G, Gnext
    end
    return -trace(heff(problem), G, last(sqrtdq), quad, first(integrandbuffers))
end

function trace(h, G, sqrtqlast, quad, integrandbuffer)
    Z = gaussnorm(sqrtqlast)
    integrand(ν, Gν) = (Gν + abs(ν)) * wrappedgaussian(ν, h, sqrtqlast; normalized=false) / Z
    set_abstolscale!(quad, zero(eltype(G)))
    I = integrate(integrand, quad, G; integrandbuffer)
    checkgaussianleak(h, sqrtqlast, cutoff(quad))
    return I
end

function Gmap!(Gnext, G, ai, sqrtdqi, quad, integrandbuffers)
    Base.require_one_based_indexing(Gnext)
    N, ntasks = length(Gnext), length(integrandbuffers)
    (N > 0) || return nothing
    integrandbuffer, Z = first(integrandbuffers), gaussnorm(sqrtdqi)
    set_abstolscale!(quad, zero(eltype(Gnext)))
    Gnext[1] = Gnext_1 = Gnext_index(1, G, ai, sqrtdqi, quad; integrandbuffer, Z)
    set_abstolscale!(quad, abs(Gnext_1))
    Gnext_indices_rest = 2:N
    Gnext_index_beyond = N + 1
    Gnext_index_vanish = if ntasks == 1
        Gmap_inner!(Gnext_indices_rest, Gnext, G, ai, sqrtdqi, quad, integrandbuffer, Z)
    else
        Base.require_one_based_indexing(integrandbuffers)
        Gnext_index_vanish_current = Threads.Atomic{Int}(Gnext_index_beyond)
        done(j) = (j > Gnext_index_vanish_current[])
        Threads.@threads for i in 1:ntasks
            # We recompute Z, ntasks, and N; it's cheap and saves 3 captured variables,
            # which actually kind of matters in the reverse AD pass
            # Interleaving single indices introduces false sharing, but the cache miss cost
            # is negligible compared to Gnext_index(), and interleaving cache line-sized
            # blocks results in a slower gradient due to the more complex code
            Gnext_index_vanish_task = Gmap_inner!(
                (i + 1):length(integrandbuffers):length(Gnext),
                Gnext,
                G,
                ai,
                sqrtdqi,
                quad,
                integrandbuffers[i],
                gaussnorm(sqrtdqi),
                done
            )
            Threads.atomic_min!(Gnext_index_vanish_current, Gnext_index_vanish_task)
        end
        Gnext_index_vanish_current[]
    end
    cutoff_ = cutoff(quad)
    lastμ = if Gnext_index_vanish < Gnext_index_beyond
        Gnext[Gnext_index_vanish:N] .= zero(eltype(Gnext))
        getnode(quad, Gnext_index_vanish)
    else
        cutoff_
    end
    checkgaussianleak(lastμ, sqrtdqi, cutoff_)
    return nothing
end

function Gmap_inner!(indices, Gnext, G, ai, sqrtdqi, quad, integrandbuffer, Z, done=nothing)
    for j in indices
        if !isnothing(done)
            done(j) && break
        end
        Gnext_j = Gnext_index(j, G, ai, sqrtdqi, quad; integrandbuffer, Z)
        (Gnext_j <= zero(Gnext_j)) && return j  # zero => it's over
        Gnext[j] = Gnext_j
    end
    return lastindex(Gnext) + 1
end

function Gnext_index(j, G, ai, sqrtdqi, quad; Z=gaussnorm(sqrtdqi), kwargs...)
    μabs = abs(getnode(quad, j))
    dqi = sqrtdqi^2
    mai = -ai
    μabspadq, μabsmadq = muladd(ai, dqi, μabs), muladd(mai, dqi, μabs)
    shift = 2mai * μabs
    denom = sqrt(2one(sqrtdqi)) * sqrtdqi
    # Is it possible to compute I1 without subtraction => cancellation?
    I1 = ((exp(shift) * erfc(μabsmadq / denom)) - erfc(μabspadq / denom)) / 2
    function integrand(ν, Gν)
        g1 = gaussian(abs(ν) - μabspadq, sqrtdqi; normalized=false)
        g2 = gaussian(abs(ν) + μabsmadq, sqrtdqi, shift; normalized=false)
        return ((g1 + g2) * expm1(ai * Gν) / Z) + (I1 * exp(-abs(ν)))
    end
    I = integrate(integrand, quad, G; kwargs...)
    return log1p(I) / ai
end
