
# using Enzyme

# Enzyme.API.runtimeActivity!(true)

# using ApproxFun: Fun, Chebyshev
# using Optimization
# using OptimizationNLopt
# using QuadGK

#
#
# function solveRSBweightsSK(β, h, J0, M, RSBweights0, quadparams; tol=16.0^(-11))
#     f = OptimizationFunction{true}(RSBobjectiveSK, AutoEnzyme())
#     p = (β, h, J0, M, quadparams)
#     prob = OptimizationProblem(f, RSBweights0, p; sense=MaxSense)
#     sol = solve(
#         prob,
#         ## Optim.jl
#         # BFGS(linesearch=Optim.LineSearches.BackTracking()); reltol=tol, g_tol=tol,
#         # LBFGS(linesearch=Optim.LineSearches.BackTracking()); reltol=tol, g_tol=tol,
#         # IPNewton(); reltol=tol, g_tol=tol,  # bounds only
#         # NewtonTrustRegion(); reltol=tol, g_tol=tol,  # unbounded only, brittle
#         ## Ipopt.jl
#         # Ipopt.Optimizer(); tol=tol, acceptable_tol=tol,  # brittle
#         ## NLopt.jl
#         NLopt.LD_LBFGS(); abstol=tol, reltol=tol,  # Winner so far
#         # NLopt.LD_VAR1(); abstol=tol, reltol=tol,
#         # NLopt.LD_VAR2(); abstol=tol, reltol=tol,
#         # NLopt.LD_TNEWTON(); abstol=tol, reltol=tol,  # bounds only
#         # NLopt.LD_TNEWTON_RESTART(); abstol=tol, reltol=tol, # bounds only
#         # NLopt.LD_TNEWTON_PRECOND(); abstol=tol, reltol=tol,
#         # NLopt.LD_TNEWTON_PRECOND_RESTART(); abstol=tol, reltol=tol,
#         # NLopt.LD_SLSQP(); abstol=tol, reltol=tol,
#         ## OptimizationPolyalgorithms.jl
#         # PolyOpt(); reltol=tol, maxiters=80,
#     )
#     if !SciMLBase.successful_retcode(sol)
#         error("Optimization failed with retcode $(sol.retcode)")
#     end
#     return sol.u
# end
#
# RSBweights = let β=inv(0.0), h=0.0, J0=0.0, M=0.0, kRSB=50
#     RSBweights = initRSBweightsSK(β, kRSB)
#     μmax = 8.25 #6.5 + 17sqrt(maximum(diff(-q))) / 2  # Heuristic that works(?)
#     for nquad in (900, 1260)
#         quadparams = Quadrature(gausslegendre, nquad, zero(μmax), μmax)
#     # @profview begin
#         RSBweights = solveRSBweightsSK(β, h, J0, M, RSBweights, quadparams)
#     # end
#     end
#     RSBweights
# end
#
# let β=inv(0.0)
#     (; a, q) = RSBparams(β, RSBweights)
#     afull = isinf(β) ? [a; 0.0; 0.0] : [β; a; 0.0; 0.0]
#     qfull = isinf(β) ? [q; 0.0] : [1.0; q; 0.0]
#     fig = Figure()
#     ax = Axis(fig[1, 1]; xlabel=L"a", ylabel=L"q(a)")
#     stairs!(ax, afull, qfull; step=:pre)
#     # stairs!(ax, a2ξ.(afull).^sqrt(2), asin.(qfull) / (π / 2); step=:pre)
#     # xlims!(ax, 0.0, 10.0)
#     fig
# end
#
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
#
# let
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     for Gv in Ginspect
#         lines!(ax, qparams.nodes, Gv)
#     end
#     fig
# end
#
# let
#     (; a) = RSBparams(β, RSBweights)
#     (; nodes) = qparams
#     i = lastindex(a)
#     diffplot(
#         Ginspect_quadgk[i + 1].(nodes), Ginspect[i + 1], nodes
#         # exp.(a[i] .* Ginspect_quadgk[i + 1].(nodes)),
#         # exp.(a[i] .* Ginspect[i + 1]),
#         # nodes
#         flabel=L"\mathcal{G}_{%$i}",
#         xlabel=L"\mu",
#         approxlabel="Legendre",
#         relativeto=first,
#         relativetolabel=L"%$(first(nodes))",
#     )
# end
