### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 2398a4a2-492c-11ef-024d-0bd270e9187a
begin
    if occursin("intel", lowercase(first(Sys.cpu_info()).model))
        using MKL
        # elseif Sys.isapple()
        #     using AppleAccelerate
    end

    using BenchmarkTools, Revise  # From global env

    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
    using Pkg: Pkg
    Pkg.activate(temp=true)
    Pkg.develop(path="..")
    Pkg.develop(path="../../Enzyme.jl")
    Pkg.develop(path="../../Enzyme.jl/lib/EnzymeTestUtils")
    Pkg.add([
        "CairoMakie", "LinearAlgebra", "Logging", "PlutoHooks", "ProgressLogging"
    ])
    Pkg.precompile()

    using Logging: Logging, @logmsg, LogLevel
    using PlutoHooks: is_running_in_pluto_process, @only_as_script
    if !is_running_in_pluto_process()
        # ProgressLogging support outside Pluto
        Pkg.add(["TerminalLoggers"])
        Pkg.precompile()
        using TerminalLoggers
        Logging.global_logger(TerminalLogger())
    end

    using LinearAlgebra: BLAS, norm
    BLAS.set_num_threads(1)

    using ReplicaSymmetryBreaking, CairoMakie, ProgressLogging
    import Enzyme, EnzymeTestUtils
end

# ╔═╡ 6c2156a8-66d4-439a-ae96-9e9307b0dc3c
ntasks = 1  # Threads.threadpoolsize()

# ╔═╡ 907b7363-234d-46ae-aacd-6379017238dd
begin
    problem = SKProblem(;
        k=12,
        β=Inf,
        # β=inv(0.25),
    )
    # u0 = initstate(problem)
	# op0 = orderparameters(problem, u0)
	op0 = initorderparameters(problem)
    # @only_as_script @show u0
	@only_as_script @show op0
    # (; u0, op0)
	op0
end

# ╔═╡ 305d710c-82c9-4db0-a516-429c17f976f6
cheb = ChebyshevInterpolator{typeof(inversetemp(problem))}();

# ╔═╡ 78830d31-fdf0-4a46-9856-08ecfa22d9a0
begin
    quadorder = 13
    quadoptions = (rtol=eps()^(7 // 8), transformtype=:geometric)
end;

# ╔═╡ f1a7902d-a98e-4572-8cf3-5df3893404b6
quad = QuadGK(quadorder, cheb; quadoptions...);

# ╔═╡ cab48286-823e-475c-ac76-874f17b3a80b
# Quadrature sanity checks
let
	(; integrate) = ReplicaSymmetryBreaking
	transform = ReplicaSymmetryBreaking.gettransform(quad)
    function f1(t)
        x = ReplicaSymmetryBreaking.apply(transform, t)
        Jx = ReplicaSymmetryBreaking.applyjacobian(transform, t)
        return Jx * 3exp(-3abs(x))
    end
    function f2(t)
        x = ReplicaSymmetryBreaking.apply(transform, t)
        Jx = ReplicaSymmetryBreaking.applyjacobian(transform, t)
        return Jx * exp(-π * x^2 / 4)
    end
    @show integrate(f1, quad)
    @show integrate(f2, quad)
end;

# ╔═╡ b46036e0-2a7f-491e-abef-c5c956493e3b
### for GaussKronrod only I think
# let nodes = getnodes(quad)
#     @show nodes[end]
#     fig = Figure()
#     ax = Axis(fig[1, 1]; xlabel=L"$x$", ylabel=L"fraction of nodes below $|x|$")
#     stairs!(ax, [0; nodes], [0; cumsum(one.(nodes))] / length(nodes); step=:post)
#     vlines!(ax, nodes; color=(:gray, 0.25))
#     xlims!(ax, sort(flipsign.([-0.1, 10.1], cutoff(quad))))
# 	@only_as_script display(fig)
#     fig
# end

# ╔═╡ ba8b018b-ef61-4477-9406-a01e43d205fa
# op = deepcopy(op0);
u0 = ReplicaSymmetryBreaking.state(problem, op0);

# ╔═╡ bf889797-59cf-4a0d-b2b0-8198c50a2d88
let
    obj = begin
        # countintegrandcalls() do
        # objective(u0, problem, quad, ws)
        # objective(u0, problem, quad, op)
		objective(u0, problem, quad)
        # end
    end
    @show obj
    # b = @benchmark objective($u0, $problem, $quad, $ws)
    # b = @benchmark objective($u0, $problem, $quad, $op)
	b = @benchmark objective($u0, $problem, $quad)
    @only_as_script display(b)
    b
end

# ╔═╡ edb403f7-1b49-4312-b0e5-81eed91bf24b
let
    du = zero(u0)
    # dobjective! = ReplicaSymmetryBreaking.make_dobjective(ws)
    # dobjective! = ReplicaSymmetryBreaking.make_dobjective(op)
	dobjective! = ReplicaSymmetryBreaking.dobjective!
    # dobjective!(du, u0, problem, quad, ws)
    # dobjective!(du, u0, problem, quad, op)
	dobjective!(du, u0, problem, quad)
    @show du
    @show norm(du, Inf)
    b = @benchmark begin
        # $dobjective!($du, $u0, $problem, $quad, $ws)
        # $dobjective!($du, $u0, $problem, $quad, $op)
		$dobjective!($du, $u0, $problem, $quad)
    end setup = (fill!($du, 0)) evals = 1
    @only_as_script display(b)
    b
end

# ╔═╡ b77c3cbd-8544-4eaa-97ae-fc492455ab06
# ╠═╡ disabled = true
#=╠═╡
begin
	Enzyme.make_zero!(op)
	EnzymeTestUtils.test_reverse(
		objective,
		Enzyme.Active,
		(u0, Enzyme.Duplicated),
		(problem, Enzyme.Const),
		(quad, Enzyme.Const),
		(op, Enzyme.Duplicated);
		atol=16.0^(-10),
		rtol=16.0^(-5),
	)
end
  ╠═╡ =#

# ╔═╡ cdc7474e-af4a-4120-ba4e-6c095a00ef68
# ╠═╡ disabled = true
#=╠═╡
let p=rand(4), atol=0.0, rtol=16.0^(-5), qtol=16.0^(-11), quad=quad
	a, b = getlimits(quad)
	function testcheb(p)
		f(x) = cos(evalpoly(x, p))
		fapprox = ReplicaSymmetryBreaking.approximate(f, quad)
		return fapprox.(a:0.1:b)
	end
	EnzymeTestUtils.test_reverse(
		testcheb, Enzyme.Duplicated, (p, Enzyme.Duplicated); atol, rtol
	)
	function testintegral(p)
		f(x) = cos(evalpoly(x, p))
		return first(ReplicaSymmetryBreaking.quadgk(f, a, b; rtol=qtol))
	end
	EnzymeTestUtils.test_reverse(
		testintegral, Enzyme.Active, (p, Enzyme.Duplicated); atol, rtol
	)
	function testintegralcheb(p)
		f(x) = cos(evalpoly(x, p))
		fapprox = ReplicaSymmetryBreaking.approximate(f, quad)
		return first(ReplicaSymmetryBreaking.quadgk(fapprox, a, b; rtol=qtol))
	end
	EnzymeTestUtils.test_reverse(
		testintegralcheb, Enzyme.Active, (p, Enzyme.Duplicated); atol, rtol
	)
	function testchebintegral(p)
		f(x, y) = cos(evalpoly(x, y .* p))
		g(y) = first(ReplicaSymmetryBreaking.quadgk(x -> f(x, y), a, b; rtol))
		gapprox = ReplicaSymmetryBreaking.approximate(g, quad)
		return gapprox.(a:0.1:b)
	end
	EnzymeTestUtils.test_reverse(
		testchebintegral, Enzyme.Duplicated, (p, Enzyme.Duplicated); atol, rtol
	)
end
  ╠═╡ =#

# ╔═╡ a1bac9d8-ccd8-4c5c-9c23-6a80b53311c9
# ╠═╡ disabled = true
#=╠═╡
(funknots, quadorder, quadscale)
  ╠═╡ =#

# ╔═╡ 444d9618-037a-45ee-83d8-5c084bb0563f
# mapping the s parameter in the block parameter transform to the number of
# iterations for convergence at k = 37, β = Inf
Dict("1/π" => 1572, "1/2" => 1390, "1" => 1302, "2" => 1179, "π" => 1363)

# ╔═╡ bdba40e8-d461-44c7-b077-a58549331b3f
function plot_orderparameters(problem, op)
    (; β) = problem
    (; a, sqrtdq) = op
    afull = isinf(β) ? [a; 0; 0] : [β; a; 0; 0]
    q = 1 .- cumsum(sqrtdq .^ 2)
    qfull = isinf(β) ? q : [1; q]
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"a", ylabel=L"q(a)")
    stairs!(ax, afull, qfull; step=:pre)
    return fig
end;

# ╔═╡ bb3f40b4-38be-440c-b04f-e24f52a561b1
let
    p = plot_orderparameters(problem, op0)
    @only_as_script display(p)
    p
end

# ╔═╡ ea51f021-afc1-4c4c-b235-4c42d5979525
function plot_tracefunctions(problem, quad, op; interval=(0.0 .. 3.0))
    (; a, sqrtdq) = op

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"\mu", ylabel=L"\mathcal{G}(\mu)")

    (; apply, applyinverse, applyjacobian, gettransform, G0, Gmap) = ReplicaSymmetryBreaking

    transform = gettransform(quad)
    GJfirst(t) = G0(problem, apply(transform, t)) * applyjacobian(transform, t)
    # t0 = applyinverse(transform, 0.0)
    # J0 = jacobian(transform, t0)
    # @show GJfirst(t0) / J0
    lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJfirst(t) / applyjacobian(transform, t)))

    GJ = Gmap(GJfirst, a[begin], sqrtdq[begin+1], quad)
    # @show GJ(t0) / J0
    lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJ(t) / applyjacobian(transform, t)))

    arest, sqrtdqrest = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
    for i in eachindex(arest, sqrtdqrest)
        GJ = ReplicaSymmetryBreaking.Gmap(GJ, arest[i], sqrtdqrest[i], quad)
        # @show GJ(t0) / J0
        lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJ(t) / applyjacobian(transform, t)))
    end
    return fig
end;

# ╔═╡ 3a572f35-472e-481f-8c38-5d1453186e8c
let interval = (7.245..10.0)
    fig = plot_tracefunctions(problem, quad, op0; interval)
    ylims!(-1e-15, 1e-15)
    @only_as_script display(fig)
    fig
end

# ╔═╡ 69eede71-0051-4590-9f66-d0483dea0dd5
function plot_tracefunctions_transformed(
    problem, quad, op; interval=(..(limits(quad)...))
)
    (; a, sqrtdq) = op

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"\mu", ylabel=L"\mathcal{G}(\mu)")

    (; apply, applyinverse, applyjacobian, gettransform, G0, Gmap) = ReplicaSymmetryBreaking

    transform = gettransform(quad)
    GJfirst(t) = G0(problem, apply(transform, t)) * applyjacobian(transform, t)
    # t0 = applyinverse(transform, 0.0)
    # @show GJfirst(t0)
    lines!(ax, interval, t -> GJfirst(t))

    GJ = Gmap(GJfirst, a[begin], sqrtdq[begin+1], quad)
    # @show GJ(t0)
    lines!(ax, interval, t -> GJ(t))

    arest, sqrtdqrest = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
    for i in eachindex(arest, sqrtdqrest)
        GJ = ReplicaSymmetryBreaking.Gmap(GJ, arest[i], sqrtdqrest[i], quad)
        # @show GJ(t0)
        lines!(ax, interval, t -> GJ(t))
    end
    return fig
end;

# ╔═╡ bf8ed7ad-9424-4900-878c-99347ee4828e
let interval = (0.76..1.0)
    fig = plot_tracefunctions_transformed(problem, quad, op0; interval)
    ylims!(-1e-15, 1e-15)
    @only_as_script display(fig)
    fig
end

# ╔═╡ 09cce4ed-0807-450f-b1c8-34c9015db9b7
begin
    struct ProgressCallback{P}
        problem::P
        i::Base.RefValue{Int}
        unans::Vector{Vector{Float64}}

        function ProgressCallback(problem)
            return new{typeof(problem)}(problem, Ref(0), Vector{Float64}[])
        end
    end

    function (c::ProgressCallback)(state, value)
        if isnan(value)
            # This branch is never encountered when using this callback with Ipopt,
            # as Ipopt handles nans/infs in the line search code and never lets them
            # through to the iteration postamble where the callback is called
            push!(c.unans, copy(state.u))
            op = Orderparameters(c.problem, state.u)
            @debug "Got $value" op = op
            nnans = length(c.unans)
            if nnans >= 64
                @error "$nnans iterations returned $value, terminating"
                return true
            end
        else
            i = c.i[]
            logprogress(i)
            c.i[] = i + 1  # post-increment; first callback happens before first step
        end
        return false
    end

    function reset!(c::ProgressCallback)
        empty!(c.unans)
        logprogress(0)
        c.i[] = 0
    end

    function logprogress(progress)
        p, q = fldmod(progress, 100) ./ 100
        @logmsg LogLevel(-1) _id = "steps % 100" progress = q
        @logmsg LogLevel(-1) _id = "steps ÷ 100" progress = p
    end
end;

# ╔═╡ c17b023d-f5d3-497c-8bf3-913e9d75f6fa
begin
    # Find the lowest-order quadrature that reproducibly achieves the required
    # tolerance throughout the optimization
    us, unans, times = Dict(), Dict(), Dict()
    # lm = 2.0^4
    let callback = ProgressCallback(problem), ipoptions = (
            tol=eps()^(11 // 12),
            # acceptable_tol=eps()^(5 // 6),
            # acceptable_obj_change_tol=eps()^(11 // 12),
            # acceptable_iter=2^4,
            # limited_memory_init_val_min=lm * sqrt(eps()),
            # limited_memory_init_val=lm,
            # limited_memory_init_val_max=lm / sqrt(eps()),
            limited_memory_max_history=2^14,
            max_wall_time=8 * 3600.0,
            max_iter=2^14,
            print_level=4,
            output_file="ipopt.log",
            file_print_level=5,
            # derivative_test="first-order",
        )
        @info quadoptions
        # for scale in sqrt(2.0), quadorder in 13, funknots in 128
        quadconfig = (; quadorder)
        @info quadconfig
        quad = QuadGK(quadconfig..., cheb; quadoptions...)
        try
            times[quadconfig] = @elapsed for _ in 1:1  # 3x for robustness
                reset!(callback)
                # u = optimize_sequence(
                #     problem, quad;
                u = optimize(
                    # problem, quad, u0;
					problem, quad, op0;
                    # ntasks,
                    warn_on_handled_exceptions=true, callback, ipoptions...,
                )
                @assert keys(us) == keys(unans)
                if quadconfig in keys(us)
                    push!(us[quadconfig], u)
                    push!(unans[quadconfig], callback.unans)
                else
                    us[quadconfig] = [u]
                    unans[quadconfig] = [callback.unans]
                end
            end
        catch e
            @error(
                ReplicaSymmetryBreaking.errormessage(e),
                # exception=e,
                exception = (e, catch_backtrace()),
            )
        end
        # end
    end
    sort(times; byvalue=true)
end

# ╔═╡ 753bba73-ccbf-426f-9e54-64f8f3bbe656
let p = nothing
    for (quadconfignan, unans_) in unans
        if !isempty(first(unans_))
            unan = first(first(unans_))
            opnan = Orderparameters(problem, unan)
            @show quadconfignan
            @show unan
            @show opnan
            p = plot_orderparameters(problem, opnan)
            break
        end
    end
    p
end

# ╔═╡ 01b947d1-287f-4079-a23d-56e38c540100
begin
    quadconfig_ = first(keys(us))  # first(keys(times))
    quad_ = QuadGK(quadconfig_..., cheb; quadoptions...)
    # ws_ = Workspace(problem, quad_; ntasks)
    u_ = first(us[quadconfig_])
    op_ = orderparameters(problem, u_)
    @only_as_script @show quadconfig_
    @only_as_script @show u_
    @only_as_script @show op_
    (; quadconfig_, u_, op_)
end

# ╔═╡ b58543ee-106b-4414-9c1b-e6e626f2cd82
let a = op_.a
    a_valid = all(>(0), a) && all(<(0), diff(a))
    @only_as_script @show a_valid
    a_valid
end

# ╔═╡ f71f7f3f-354e-46df-afe6-4cd09a51c7d4
let sqrtdq = op_.sqrtdq
    @show sum(abs2, sqrtdq)
    q_valid = (sqrtdq[1] >= 0) && all(>(0), sqrtdq[2:end]) && (sum(abs2, sqrtdq) ≈ 1)
    @only_as_script @show q_valid
    q_valid
end

# ╔═╡ 7aa22855-90fb-400c-80ee-e15ce653b672
let
    p = plot_orderparameters(problem, op_)
    @only_as_script display(p)
    p
end

# ╔═╡ 438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
let
    obj_ = objective(u_, problem, quad_)
    @only_as_script @show obj_
    obj_
end

# ╔═╡ f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
let du_ = zero(u_)
    # dobjective! = ReplicaSymmetryBreaking.make_dobjective(op_)
	dobjective! = ReplicaSymmetryBreaking.dobjective!
    # dobjective!(du_, u_, problem, quad_, op_)
	dobjective!(du_, u_, problem, quad_)
    @only_as_script @show du_
    @only_as_script @show norm(du_, Inf)
    (; du_, norm=norm(du_, Inf))
end

# ╔═╡ 24e46e23-37d7-4bae-93bb-b6e30a5ead82
let
    fig = plot_tracefunctions(problem, quad_, op_)
    @only_as_script display(fig)
    fig

end

# ╔═╡ 4e9b3ab2-b77b-4513-882f-8521cfe51d2e
let
    fig = plot_tracefunctions_transformed(problem, quad_, op_)
    @only_as_script display(fig)
    fig
end

# ╔═╡ aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
let interval = (6.94..10.0)
    fig = plot_tracefunctions(problem, quad_, op_; interval)
    ylims!(-1e-15, 1e-15)
    @only_as_script display(fig)
    fig

end

# ╔═╡ 26252945-58a9-4b7d-a8ee-403aa54dc8a3
let interval = (0.759..1.0)
    fig = plot_tracefunctions_transformed(problem, quad_, op_; interval)
    ylims!(-1e-15, 1e-15)
    @only_as_script display(fig)
    fig
end

# ╔═╡ 791c6fa7-69fb-44ef-b53b-8be6f937a7bb
# ╠═╡ disabled = true
#=╠═╡
EnzymeTestUtils.test_reverse(
	objective, Enzyme.Active,
	(u_, Enzyme.Duplicated),
	(problem, Enzyme.Const),
	(quad_, Enzyme.Const),
	(op_, Enzyme.Duplicated);
	atol=16.0^(-10),
	rtol=16.0^(-5),
)
  ╠═╡ =#

# ╔═╡ 4a14560b-2a4a-477d-b882-906c06a03504
objectives_ = vcat((
    let quad__ = QuadGK(quadconfig__..., cheb; quadoptions...)
        [
			objective(u__, problem, quad__)  #, orderparameters(problem, u__))
			for u__ in us__
		]
    end
    for (quadconfig__, us__) in us
)...)

# ╔═╡ e741586e-fc4f-4406-8152-aa20cef4e37e
objectivemax_ = maximum(objectives_)

# ╔═╡ 76a6bb36-768b-41fd-8b13-fc7450729e3b
objectivemax_ .- objectives_

# ╔═╡ 2f946e07-9726-4517-b094-0facbf73766a
objectivemax_ - minimum(objectives_)

# ╔═╡ Cell order:
# ╠═6c2156a8-66d4-439a-ae96-9e9307b0dc3c
# ╠═2398a4a2-492c-11ef-024d-0bd270e9187a
# ╠═907b7363-234d-46ae-aacd-6379017238dd
# ╠═bb3f40b4-38be-440c-b04f-e24f52a561b1
# ╠═305d710c-82c9-4db0-a516-429c17f976f6
# ╠═78830d31-fdf0-4a46-9856-08ecfa22d9a0
# ╠═f1a7902d-a98e-4572-8cf3-5df3893404b6
# ╠═cab48286-823e-475c-ac76-874f17b3a80b
# ╠═b46036e0-2a7f-491e-abef-c5c956493e3b
# ╠═ba8b018b-ef61-4477-9406-a01e43d205fa
# ╠═bf889797-59cf-4a0d-b2b0-8198c50a2d88
# ╠═edb403f7-1b49-4312-b0e5-81eed91bf24b
# ╠═b77c3cbd-8544-4eaa-97ae-fc492455ab06
# ╠═cdc7474e-af4a-4120-ba4e-6c095a00ef68
# ╠═3a572f35-472e-481f-8c38-5d1453186e8c
# ╠═bf8ed7ad-9424-4900-878c-99347ee4828e
# ╠═a1bac9d8-ccd8-4c5c-9c23-6a80b53311c9
# ╠═c17b023d-f5d3-497c-8bf3-913e9d75f6fa
# ╠═444d9618-037a-45ee-83d8-5c084bb0563f
# ╠═753bba73-ccbf-426f-9e54-64f8f3bbe656
# ╠═01b947d1-287f-4079-a23d-56e38c540100
# ╠═b58543ee-106b-4414-9c1b-e6e626f2cd82
# ╠═f71f7f3f-354e-46df-afe6-4cd09a51c7d4
# ╠═7aa22855-90fb-400c-80ee-e15ce653b672
# ╠═438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
# ╠═f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
# ╠═24e46e23-37d7-4bae-93bb-b6e30a5ead82
# ╠═4e9b3ab2-b77b-4513-882f-8521cfe51d2e
# ╠═aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
# ╠═26252945-58a9-4b7d-a8ee-403aa54dc8a3
# ╠═791c6fa7-69fb-44ef-b53b-8be6f937a7bb
# ╠═4a14560b-2a4a-477d-b882-906c06a03504
# ╠═e741586e-fc4f-4406-8152-aa20cef4e37e
# ╠═76a6bb36-768b-41fd-8b13-fc7450729e3b
# ╠═2f946e07-9726-4517-b094-0facbf73766a
# ╠═bdba40e8-d461-44c7-b077-a58549331b3f
# ╠═ea51f021-afc1-4c4c-b235-4c42d5979525
# ╠═69eede71-0051-4590-9f66-d0483dea0dd5
# ╠═09cce4ed-0807-450f-b1c8-34c9015db9b7
