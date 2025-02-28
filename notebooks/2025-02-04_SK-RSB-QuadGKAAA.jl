### A Pluto.jl notebook ###
# v0.20.4

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
	# Pkg.develop(path="../../Enzyme.jl")
	# Pkg.develop(path="../../Enzyme.jl/lib/EnzymeTestUtils")
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
	# import Enzyme, EnzymeTestUtils
end

# ╔═╡ 6c2156a8-66d4-439a-ae96-9e9307b0dc3c
ntasks = 1  # Threads.threadpoolsize()

# ╔═╡ 907b7363-234d-46ae-aacd-6379017238dd
begin
    problem = SKProblem(; k=12, β=Inf)
    op0 = Orderparameters(problem)
    u0 = state(problem, op0)
    ReplicaSymmetryBreaking.updateorderparameters!(op0, problem, u0)
    (; op0, u0)
end

# ╔═╡ 78830d31-fdf0-4a46-9856-08ecfa22d9a0
begin
    quadorder = 12
	quadoptions = (; rtol=eps()^(11 // 13), stretch=:power)
end;

# ╔═╡ f1a7902d-a98e-4572-8cf3-5df3893404b6
quad = QuadGKAAA(quadorder; quadoptions...);

# ╔═╡ cab48286-823e-475c-ac76-874f17b3a80b
# Quadrature sanity checks
let integrate=ReplicaSymmetryBreaking.integrate, transform=ReplicaSymmetryBreaking.gettransform(quad)
	function f1(t)
		x = ReplicaSymmetryBreaking.apply(transform, t)
		Jx = ReplicaSymmetryBreaking.jacobian(transform, t)
		return Jx * 3exp(-3abs(x))
	end
	function f2(t)
		x = ReplicaSymmetryBreaking.apply(transform, t)
		Jx = ReplicaSymmetryBreaking.jacobian(transform, t)
		return Jx * exp(-π * x^2 / 4)
	end
	@show integrate(f1, quad)
	@show integrate(f2, quad)
end;

# ╔═╡ b46036e0-2a7f-491e-abef-c5c956493e3b
# ╠═╡ disabled = true
#=╠═╡
let nodes = getnodes(quad)
    @show nodes[end]
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"$x$", ylabel=L"fraction of nodes below $|x|$")
    stairs!(ax, [0; nodes], [0; cumsum(one.(nodes))] / length(nodes); step=:post)
    vlines!(ax, nodes; color=(:gray, 0.25))
    xlims!(ax, sort(flipsign.([-0.1, 10.1], cutoff(quad))))
	@only_as_script display(fig)
    fig
end
  ╠═╡ =#

# ╔═╡ ba8b018b-ef61-4477-9406-a01e43d205fa
# ws = Workspace(problem, quad; ntasks);
op = Orderparameters(problem);

# ╔═╡ bf889797-59cf-4a0d-b2b0-8198c50a2d88
let
    obj = begin
        # countintegrandcalls() do
        # objective(u0, problem, quad, ws)
		objective(u0, problem, quad, op)
        # end
    end
    @show obj
    # @benchmark objective($u0, $problem, $quad, $ws)
	@benchmark objective($u0, $problem, $quad, $op)
end

# ╔═╡ edb403f7-1b49-4312-b0e5-81eed91bf24b
let
	du = zero(u0)
	# dobjective! = ReplicaSymmetryBreaking.make_dobjective(ws)
	dobjective! = ReplicaSymmetryBreaking.make_dobjective(op)
 	# dobjective!(du, u0, problem, quad, ws)
	dobjective!(du, u0, problem, quad, op)
	@show du
	@show norm(du, Inf)
	@benchmark begin
	    # $dobjective!($du, $u0, $problem, $quad, $ws)
		$dobjective!($du, $u0, $problem, $quad, $op)
	end setup = (fill!($du, 0)) evals = 1
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
let p=rand(4), atol=0.0, rtol=16.0^(-5), qtol=16.0^(-11)
	function testaaa(p)
		f(x) = cos(evalpoly(x, p))
		fapprox = ReplicaSymmetryBreaking.aaa(f, 0.0, 1.0)
		return fapprox.(0.0:0.05:1.0)
	end
	EnzymeTestUtils.test_reverse(
		testaaa, Enzyme.Duplicated, (p, Enzyme.Duplicated); atol, rtol
	)
	function testintegral(p)
		f(x) = cos(evalpoly(x, p))
		return first(ReplicaSymmetryBreaking.quadgk(f, 0.0, 1.0; rtol=qtol))
	end
	EnzymeTestUtils.test_reverse(
		testintegral, Enzyme.Active, (p, Enzyme.Duplicated); atol, rtol
	)
	function testintegralaaa(p)
		f(x) = cos(evalpoly(x, p))
		fapprox = ReplicaSymmetryBreaking.aaa(f, 0.0, 1.0)
		return first(ReplicaSymmetryBreaking.quadgk(fapprox, 0.0, 1.0; rtol=qtol))
	end
	EnzymeTestUtils.test_reverse(
		testintegralaaa, Enzyme.Active, (p, Enzyme.Duplicated); atol, rtol
	)
	function testaaaintegral(p)
		f(x, y) = cos(evalpoly(x, y .* p))
		g(y) = first(ReplicaSymmetryBreaking.quadgk(x -> f(x, y), 0.0, 1.0; rtol))
		gapprox = ReplicaSymmetryBreaking.aaa(g, 0.0, 1.0)
		return gapprox.(0.0:0.05:1.0)
	end
	EnzymeTestUtils.test_reverse(
		testaaaintegral, Enzyme.Duplicated, (p, Enzyme.Duplicated); atol, rtol
	)
end
  ╠═╡ =#

# ╔═╡ 9c342e4e-4b50-4b9c-b576-607eb2016417
# ╠═╡ disabled = true
#=╠═╡
let v=ReplicaSymmetryBreaking.Enzyme.onehot(u0)[1], Hv=similar(v)
	hvpobjective! = ReplicaSymmetryBreaking.make_hvpobjective(u0, ws)
	@time hvpobjective!(Hv, v, u0, problem, quad, ws)
	Hv
end
  ╠═╡ =#

# ╔═╡ a1bac9d8-ccd8-4c5c-9c23-6a80b53311c9
# ╠═╡ disabled = true
#=╠═╡
(funknots, quadorder, quadscale)
  ╠═╡ =#

# ╔═╡ 67de3269-f6b2-4a90-9cc1-fc3dce98ac60
ustart = begin
	u0
	# [0.020815172385939107, 0.22724622236431458, 0.41740860908345745, 0.16799301507260012, 0.2827447017830682, 0.0008834561958721565, 0.09379475849998009, 0.49521703996883204, 0.3752485666147596, 0.18313681907636692, 0.17560454889544097, 0.28667255222784505]
	# [0.4329803052734796, 0.16433846147268147, 0.15539147870007025, 0.14266739658660896, 0.15979129993076951, 0.1935912759067546, 0.11784864798446296, 0.17802711182172726, 0.22624754477250583, 0.27259654103795117, 0.20235641629359236, 0.24960311893019066]
end

# ╔═╡ c17b023d-f5d3-497c-8bf3-913e9d75f6fa
# ╠═╡ disabled = true
#=╠═╡
begin
    # Find the lowest-order quadrature that reproducibly achieves the required
    # tolerance throughout the optimization
    us, unans, times = Dict(), Dict(), Dict()
	lm = 2.0^5
    let callback = ProgressCallback(problem), ipoptions = (
            tol=16.0^(-5),
            max_wall_time=300.0,
            max_iter=10000,
            print_level=5,
            output_file="ipopt.log",
            file_print_level=5,
			limited_memory_init_val_min=lm * sqrt(eps()),
			limited_memory_init_val=lm,
			limited_memory_init_val_max=lm / sqrt(eps()),
			# limited_memory_max_history=6,
			# accept_every_trial_step="yes",
			# accept_after_max_steps=10,
			# derivative_test="first-order",
        )
        @info quadoptions
        # for scale in sqrt(2.0), quadorder in 13, funknots in 128
        quadconfig = (; quadorder)
        @info quadconfig
		quad = QuadGKAAA(quadconfig...; quadoptions...)
        try
            times[quadconfig] = @elapsed for _ in 1:1  # 3x for robustness
                reset!(callback)
                u = optimize(
                    problem, quad, ustart;
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
  ╠═╡ =#

# ╔═╡ 753bba73-ccbf-426f-9e54-64f8f3bbe656
#=╠═╡
let p=nothing
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
  ╠═╡ =#

# ╔═╡ 01b947d1-287f-4079-a23d-56e38c540100
#=╠═╡
begin
	quadconfig_ = first(keys(us))  # first(keys(times))
	quad_ = QuadGKAAA(quadconfig_...; quadoptions...)
	# ws_ = Workspace(problem, quad_; ntasks)
	u_ = first(us[quadconfig_])
	op_ = Orderparameters(problem, u_)
	(; quadconfig_, u_, op_)
end
  ╠═╡ =#

# ╔═╡ b58543ee-106b-4414-9c1b-e6e626f2cd82
#=╠═╡
all(>(0), op_.a) && all(<(0), diff(op_.a))
  ╠═╡ =#

# ╔═╡ f71f7f3f-354e-46df-afe6-4cd09a51c7d4
#=╠═╡
let sqrtdq = op_.sqrtdq
	@show sum(abs2, sqrtdq)
	(sqrtdq[1] >= 0) && all(>(0), sqrtdq[2:end]) && (sum(abs2, sqrtdq) ≈ 1)
end
  ╠═╡ =#

# ╔═╡ 7aa22855-90fb-400c-80ee-e15ce653b672
#=╠═╡
let
	p = plot_orderparameters(problem, op_)
	@only_as_script display(p)
	p
end
  ╠═╡ =#

# ╔═╡ aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
#=╠═╡
let interval = (7.0..40.0)
    fig = plot_tracefunctions(problem, quad_, op_; interval)
	ylims!(-1e-15, 1e-15)
	@only_as_script display(fig)
    fig
	
end
  ╠═╡ =#

# ╔═╡ 26252945-58a9-4b7d-a8ee-403aa54dc8a3
#=╠═╡
let interval=(0.88..1.0)
    fig = plot_tracefunctions_transformed(problem, quad_, op_; interval)
	ylims!(-1e-14, 1e-14)
	@only_as_script display(fig)
    fig
end
  ╠═╡ =#

# ╔═╡ 438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
#=╠═╡
objective(u_, problem, quad_, op_)
  ╠═╡ =#

# ╔═╡ f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
#=╠═╡
let du_=zero(u_)
	dobjective! = ReplicaSymmetryBreaking.make_dobjective(op_)
	dobjective!(du_, u_, problem, quad_, op_)
	(; du_, norm=norm(du_, Inf))
end
  ╠═╡ =#

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
#=╠═╡
objectives_ = vcat((
	let quad__=QuadGKAAA(quadconfig__...; quadoptions...)
		op__ = Orderparameters(problem)
		[objective(u__, problem, quad__, op__) for u__ in us__]
	end
	for (quadconfig__, us__) in us
)...)
  ╠═╡ =#

# ╔═╡ e741586e-fc4f-4406-8152-aa20cef4e37e
#=╠═╡
objectivemax_ = maximum(objectives_)
  ╠═╡ =#

# ╔═╡ 76a6bb36-768b-41fd-8b13-fc7450729e3b
#=╠═╡
objectivemax_ .- objectives_
  ╠═╡ =#

# ╔═╡ 2f946e07-9726-4517-b094-0facbf73766a
#=╠═╡
objectivemax_ - minimum(objectives_)
  ╠═╡ =#

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

	(; apply, applyinverse, jacobian, gettransform, G0, Gmap) = ReplicaSymmetryBreaking
	
	transform = gettransform(quad)
	GJfirst(t) = G0(problem, apply(transform, t)) * jacobian(transform, t)
	t0 = applyinverse(transform, 0.0)
	J0 = jacobian(transform, t0)
 	@show GJfirst(t0) / J0
    lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJfirst(t) / jacobian(transform, t)))

    GJ = Gmap(GJfirst, a[begin], sqrtdq[begin+1], quad)
	@show GJ(t0) / J0
	lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJ(t) / jacobian(transform, t)))
	
    arest, sqrtdqrest = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
    for i in eachindex(arest, sqrtdqrest)
        GJ = ReplicaSymmetryBreaking.Gmap(GJ, arest[i], sqrtdqrest[i], quad)
		@show GJ(t0) / J0
    	lines!(ax, interval, μ -> (t = applyinverse(transform, μ); GJ(t) / jacobian(transform, t)))
    end
    return fig
end;

# ╔═╡ 3a572f35-472e-481f-8c38-5d1453186e8c
let interval=(0.0..3.0)  # (5.5..10.0)
    fig = plot_tracefunctions(problem, quad, op0; interval)
	# ylims!(-1e-15, 1e-15)
	@only_as_script display(fig)
    fig
end

# ╔═╡ 69eede71-0051-4590-9f66-d0483dea0dd5
function plot_tracefunctions_transformed(
	problem, quad, op; interval=..(getlimits(quad)...)
)
	(; a, sqrtdq) = op
	
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"\mu", ylabel=L"\mathcal{G}(\mu)")

	(; apply, applyinverse, jacobian, gettransform, G0, Gmap) = ReplicaSymmetryBreaking
	
	transform = gettransform(quad)
	GJfirst(t) = G0(problem, apply(transform, t)) * jacobian(transform, t)
	t0 = applyinverse(transform, 0.0)
 	@show GJfirst(t0)
    lines!(ax, interval, GJfirst)

    GJ = Gmap(GJfirst, a[begin], sqrtdq[begin+1], quad)
	@show GJ(t0)
	lines!(ax, interval, t -> GJ(t))
	
    arest, sqrtdqrest = @views (a[(begin+1):end], sqrtdq[(begin+2):(end-1)])
    for i in eachindex(arest, sqrtdqrest)
        GJ = ReplicaSymmetryBreaking.Gmap(GJ, arest[i], sqrtdqrest[i], quad)
		@show GJ(t0)
    	lines!(ax, interval, t -> GJ(t))
    end
    return fig
end;

# ╔═╡ bf8ed7ad-9424-4900-878c-99347ee4828e
let interval=(0.0..1.0)  #(0.85..1.0)
    fig = plot_tracefunctions_transformed(problem, quad, op0; interval)
	# ylims!(-1e-14, 1e-14)
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
            c.i[] = i = c.i[] + 1
            logprogress(i)
        end
        return false
    end

    function reset!(c::ProgressCallback)
        c.i[] = 0
        empty!(c.unans)
        logprogress(0)
    end
	
    function logprogress(progress)
		@logmsg LogLevel(-1) _id = "steps % 100" progress = (progress % 100) / 100
		@logmsg LogLevel(-1) _id = "steps ÷ 100" progress = (progress ÷ 100) / 100
	end
end;

# ╔═╡ Cell order:
# ╠═6c2156a8-66d4-439a-ae96-9e9307b0dc3c
# ╠═2398a4a2-492c-11ef-024d-0bd270e9187a
# ╠═907b7363-234d-46ae-aacd-6379017238dd
# ╠═bb3f40b4-38be-440c-b04f-e24f52a561b1
# ╠═78830d31-fdf0-4a46-9856-08ecfa22d9a0
# ╠═f1a7902d-a98e-4572-8cf3-5df3893404b6
# ╠═cab48286-823e-475c-ac76-874f17b3a80b
# ╠═b46036e0-2a7f-491e-abef-c5c956493e3b
# ╠═ba8b018b-ef61-4477-9406-a01e43d205fa
# ╠═bf889797-59cf-4a0d-b2b0-8198c50a2d88
# ╠═edb403f7-1b49-4312-b0e5-81eed91bf24b
# ╠═b77c3cbd-8544-4eaa-97ae-fc492455ab06
# ╠═cdc7474e-af4a-4120-ba4e-6c095a00ef68
# ╠═9c342e4e-4b50-4b9c-b576-607eb2016417
# ╠═3a572f35-472e-481f-8c38-5d1453186e8c
# ╠═bf8ed7ad-9424-4900-878c-99347ee4828e
# ╠═a1bac9d8-ccd8-4c5c-9c23-6a80b53311c9
# ╠═67de3269-f6b2-4a90-9cc1-fc3dce98ac60
# ╠═c17b023d-f5d3-497c-8bf3-913e9d75f6fa
# ╠═753bba73-ccbf-426f-9e54-64f8f3bbe656
# ╠═01b947d1-287f-4079-a23d-56e38c540100
# ╠═b58543ee-106b-4414-9c1b-e6e626f2cd82
# ╠═f71f7f3f-354e-46df-afe6-4cd09a51c7d4
# ╠═7aa22855-90fb-400c-80ee-e15ce653b672
# ╠═aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
# ╠═26252945-58a9-4b7d-a8ee-403aa54dc8a3
# ╠═438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
# ╠═f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
# ╠═791c6fa7-69fb-44ef-b53b-8be6f937a7bb
# ╠═4a14560b-2a4a-477d-b882-906c06a03504
# ╠═e741586e-fc4f-4406-8152-aa20cef4e37e
# ╠═76a6bb36-768b-41fd-8b13-fc7450729e3b
# ╠═2f946e07-9726-4517-b094-0facbf73766a
# ╠═bdba40e8-d461-44c7-b077-a58549331b3f
# ╠═ea51f021-afc1-4c4c-b235-4c42d5979525
# ╠═69eede71-0051-4590-9f66-d0483dea0dd5
# ╠═09cce4ed-0807-450f-b1c8-34c9015db9b7
