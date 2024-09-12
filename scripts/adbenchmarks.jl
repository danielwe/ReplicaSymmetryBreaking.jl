using BenchmarkTools
using LinearAlgebra: BLAS, norm, opnorm
BLAS.set_num_threads(1)

using ReplicaSymmetryBreaking

k = 12
β = Inf
problem = SKProblem(; k, β)

quadendpoint = Inf
quadtol = 4.0^(-22)
quadn = 256
quadscale = 8.75
quad = GaussKronrod(quadn, quadendpoint; scale=quadscale, tol=quadtol)

u0 = initstate(problem)
ws = Workspace(u0, problem, quad; ntasks=1)
ws_threads = Workspace(u0, problem, quad; ntasks=Threads.threadpoolsize())

let
    obj = @btime objective($u0, $problem, $quad, $ws)
    @show obj
    obj_threads = @btime objective($u0, $problem, $quad, $ws_threads)
    @show obj_threads
    @show abs(obj_threads - obj)
end

let
    # XXX: requires maxtypedepth!(7)
    du=similar(u0)
    dobjective! = ReplicaSymmetryBreaking.make_dobjective(ws)
    @btime $dobjective!($du, $u0, $problem, $quad, $ws)
    @show du
    @show norm(du, Inf)
    # XXX: requires runtimeActivity!(true)
    du_threads=similar(u0)
    dobjective_threads! = ReplicaSymmetryBreaking.make_dobjective(ws_threads)
    @btime $dobjective_threads!($du_threads, $u0, $problem, $quad, $ws_threads)
    @show du_threads
    @show norm(du_threads, Inf)
    @show norm(du_threads - du, Inf)
end

let
    # function hessian!(hvp!::F, u0, problem, quad, ws) where {F}
    #     vs = ReplicaSymmetryBreaking.Enzyme.onehot(u0)
    #     Hvs = similar.(vs)
    #     for i in eachindex(Hvs, vs)
    #         hvp!(Hvs[i], u0, vs[i], problem, quad, ws)
    #     end
    #     return reduce(hcat, Hvs)
    # end
    # # XXX: requires looseTypeAnalysis!(true), sometimes segfaults due to GC interactions
    # hvpobjective! = ReplicaSymmetryBreaking.make_hvpobjective(u0, ws)
    # hess = @btime $hessian!($hvpobjective!, $u0, $problem, $quad, $ws) gcsample=true
    # println("hess =")
    # show(stdout, "text/plain", hess)
    # println()
    # # XXX: can't make this work
    # hvpobjective_threads! = ReplicaSymmetryBreaking.make_hvpobjective(u0, ws_threads)
    # hess_threads = @btime $hessian!(
    #     $hvpobjective_threads!, $u0, $problem, $quad, $ws_threads
    # ) gcsample=true
    # println("hess_threads =")
    # show(stdout, "text/plain", hess_threads)
    # println()
    # @show opnorm(hess_threads - hess, Inf)
end
