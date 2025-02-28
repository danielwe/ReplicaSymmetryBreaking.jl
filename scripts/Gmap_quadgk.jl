# Scoped integrand call counting for QuadGK
let quadgkcounter=ScopedValue{Base.RefValue{Int}}()
    global function quad(args...; atol=0.0, rtol=16^(-11), order=18, kwargs...)
        I, E, count = quadgk_count(args...; atol, rtol, order, kwargs...)
        tol = max(atol, rtol * norm(I))
        if E > 2tol  # QuadGK is a mystery sometimes, hence the factor of 2
            error(
                "integral did not converge\n"
                * "final value = $I\n"
                * "final error estimate = $E\n"
                * "atol = $atol\n"
                * "rtol = $rtol)\n"
                * "max(atol, rtol * norm(I)) = $tol\n"
                * "order = $order\n"
                * "integrand call count = $count"
            )
        end
        counter = ScopedValues.get(quadgkcounter)
        isnothing(counter) || (something(counter)[] += count)
        return I, E
    end

    global function countintegrandcalls(f)
    ncalls = Ref(0)
    returns = @with quadgkcounter => ncalls f()
    return ncalls[], returns
    end
end

Gmapscale(μ, a, dq) = μ + a * dq + 8sqrt(dq)

function Gmap_quadgk!(gcoeffsnext, gcoeffs, ginterval, adq, nodes, plan!)
    (; dq) = adq
    if dq < zero(dq)
        fill!(gcoeffsnext, eltype(gcoeffsnext)(NaN))
        return nothing
    elseif iszero(dq)
        copyto!(gcoeffsnext, gcoeffs)
        return nothing
    end
    g = Fun(Chebyshev(ginterval), gcoeffs)
    Z = gaussnorm(adq.dq)
    map!(μ -> Gmap_quadgk_point(μ, g, adq, Z), gcoeffsnext, nodes)
    plan! * gcoeffsnext
    return nothing
end

function Gmap_quadgk_point(μ, g, adq, Z=gaussnorm(adq.dq))
    (; ri) = adq
    scale = Gmapscale(μ, adq.a, adq.dq)
    function integrand(x)
        ν = x * scale
        return (g(ν)^ri) * Gmapkernel(ν, μ, adq; normalized=false)
    end
    return first(quad(integrand, 0.0, Inf)) * scale / Z
end

# function μtrunc(G)
#     # Do a binary search for a good location to truncate an interpolation of g
#     local μ
#     μl, μr = 0.0, 10.0
#     gmax = g(μl)
#     while true
#         μ = (μl + μr) / 2
#         r = G(μ) / Gmax
#         if r >= 2.0^(-26)
#             μl = μ
#         elseif r <= 2.0^(-27)
#             μr = μ
#         else
#             break
#         end
#     end
#     return 2μ
# end

function Gmap_quadgk(G, adqi)
    _Gnext = Gmap_quadgk_lazy(G, adqi)
    # Return sanitized interpolant
    return let μm=μtrunc(_Gnext), itp=Fun(_Gnext, Chebyshev(0.0..μm))
    function Gnext(μ)
        μabs = abs(μ)
        (μabs > μm) && return zero(itp(zero(μabs)))
        ret = itp(μabs)
        return max(ret, zero(ret))
    end
    end
end

@views function Gs_quadgk((; problem, quad), (; a, q))
    Gs = Function[μ -> G0(problem, μ)]
    q1, q2 = q[begin:(end - 1)], q[(begin + 1):end]
    for i in eachindex(a, q1, q2)
        push!(Gs, Gmap_quadgk(last(Gs), (; ai=a[i], dqi=(q1[i] - q2[i]))))
    end
    return Gs
end

# Ginspect, Ginspect_quadgk, qparams = let β=inv(0.0), nquad=1260
#     (; a, q) = RSBparams(β, RSBweights)
#     dq = q[begin:(end - 1)] .- q[(begin + 1):end]
#     μmax = 8.25  #6.6 + 17sqrt(maximum(diff(-q))) / 2  # Heuristic that works(?)
#     @show μmax
#     quadparams = Quadrature(gausslegendre, nquad, zero(μmax), μmax)
#     Gvs = [G02.(quadparams.nodes, β)]
#     for i in eachindex(a, dq)
#         push!(Gvs, Gmap2(last(Gvs), a[i], dq[i], quadparams))
#     end
#     Gs_quadgk = let
#         G0 = μ -> G02(μ, β)
#         Gs = [G0, Gmap_quadgk(G0, a[1], dq[1])]
#         for i in eachindex(a, dq)[(begin + 1):end]
#             push!(Gs, Gmap_quadgk(last(Gs), a[i], dq[i]))
#         end
#         Gs
#     end
#     Gvs, Gs_quadgk, quadparams
# end
