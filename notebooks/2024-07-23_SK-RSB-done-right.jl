### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 2398a4a2-492c-11ef-024d-0bd270e9187a
begin
	using MKL, Revise  # From global env

	ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
	using Pkg: Pkg
	Pkg.activate(temp=true)
	Pkg.develop(path="..")
	Pkg.add(["CairoMakie", "LinearAlgebra", "Logging", "ProgressLogging"])
	Pkg.precompile()

	using LinearAlgebra: BLAS, norm
	BLAS.set_num_threads(1)

	using ReplicaSymmetryBreaking, CairoMakie, ProgressLogging
	using Logging: @logmsg, LogLevel
end

# ╔═╡ e33ec073-447d-40b2-905b-ed0db0c5b999
html"""
<script>
	const button = document.createElement("button")

	button.addEventListener("click", () => {
		editor_state_set(old_state => ({
			notebook: {
				...old_state.notebook,
				process_status: "no_process",
			},
		})).then(() => {
			window.requestAnimationFrame(() => {
				document.querySelector("#process_status a").click()
			})
		})
	})
	button.innerText = "Restart process"

	return button
</script>
"""

# ╔═╡ 6c2156a8-66d4-439a-ae96-9e9307b0dc3c
ntasks = Threads.threadpoolsize()

# ╔═╡ 907b7363-234d-46ae-aacd-6379017238dd
begin
	k = 6
	β = Inf
	problem = SKProblem(; k, β)
	u0 = initstate(problem)
	op0 = Orderparameters(problem, u0)
	(; u0, op0)
end

# ╔═╡ 78830d31-fdf0-4a46-9856-08ecfa22d9a0
begin
	quadcutoff = Inf
	quadtol = 2.0^(-48)
end;

# ╔═╡ f41354f8-a0a2-445a-95c9-352fa7290107
begin
	quadn = 60
	quadscale = 4.0
	quadstretchmode = :exp
	quad = GaussKronrod(
		quadn, quadcutoff; scale=quadscale, tol=quadtol, stretchmode=quadstretchmode
	)
end;

# ╔═╡ cab48286-823e-475c-ac76-874f17b3a80b
# Quadrature sanity checks
let integrate=ReplicaSymmetryBreaking.integrate
	f1(x) = exp(-x)
	f2(x) = exp(-π * x^2 / 4)
	@show integrate(f1, quad)
	@show integrate(f2, quad)
end;

# ╔═╡ b46036e0-2a7f-491e-abef-c5c956493e3b
let nodes=quad.nodes
	@show nodes[end]
	fig = Figure()
	ax = Axis(fig[1, 1])
	vlines!(ax, nodes; color=(:gray, 0.25))
	xlims!(ax, 0, 16)
	fig
end

# ╔═╡ ba8b018b-ef61-4477-9406-a01e43d205fa
ws = Workspace(u0, problem, quad; ntasks);

# ╔═╡ bf889797-59cf-4a0d-b2b0-8198c50a2d88
@time objective(u0, problem, quad, ws)

# ╔═╡ a0b2825e-53d7-4ddf-be69-fe7b069648b6
let du=similar(u0)
	dobjective! = ReplicaSymmetryBreaking.make_dobjective(ws)
	@time dobjective!(du, u0, problem, quad, ws)
	(; du, norm=norm(du))
end

# ╔═╡ 9c342e4e-4b50-4b9c-b576-607eb2016417
# ╠═╡ disabled = true
#=╠═╡
let v=ReplicaSymmetryBreaking.Enzyme.onehot(u0)[1], Hv=similar(v)
	hvpobjective! = ReplicaSymmetryBreaking.make_hvpobjective(u0, ws)
	@time hvpobjective!(Hv, v, u0, problem, quad, ws)
	Hv
end
  ╠═╡ =#

# ╔═╡ bdba40e8-d461-44c7-b077-a58549331b3f
function plot_orderparameters((; k, β), (; a, sqrtdq))
	afull = isinf(β) ? [a; 0; 0] : [β; a; 0; 0]
	q = 1 .- cumsum(sqrtdq.^2)
	qfull = isinf(β) ? q : [1; q]
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"a", ylabel=L"q(a)")
	stairs!(ax, afull, qfull; step=:pre)
	return fig
end;

# ╔═╡ bb3f40b4-38be-440c-b04f-e24f52a561b1
plot_orderparameters(problem, op0)

# ╔═╡ ea51f021-afc1-4c4c-b235-4c42d5979525
@views function plot_tracefunctions(
	problem, (; a, sqrtdq);
	interval=(0.0, Inf),
	quadn=quadn,
	quadcutoff=quadcutoff,
	quadscale=quadscale,
	quadtol=quadtol,
	quadstretchmode=quadstretchmode,
	ntasks=ntasks,
)
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"\mu", ylabel=L"\mathcal{G}(\mu)")
	
	quad = GaussKronrod(
		quadn, quadcutoff; scale=quadscale, tol=quadtol, stretchmode=quadstretchmode
	)
	(; nodes) = quad
	T = eltype(nodes)
	buffers = ReplicaSymmetryBreaking.Tracebuffers(T, length(nodes); ntasks)
	(; kernelbuffers, integrandbuffers) = buffers
	G, Gnext = kernelbuffers
	map!(μ -> ReplicaSymmetryBreaking.G0(problem, μ), G, nodes)
	
	I = (interval[1] .< nodes) .& (nodes .< interval[2])
	lines!(ax, nodes[I], G[I])
	
	sqrtdqbulk = sqrtdq[(begin + 1):(end - 1)]
    for i in eachindex(a, sqrtdqbulk)
		ai, sqrtdqi = a[i], sqrtdqbulk[i]
        ReplicaSymmetryBreaking.Gmap!(Gnext, G, ai, sqrtdqi, quad, integrandbuffers)
        Gnext, G = G, Gnext
		
		lines!(ax, nodes[I], G[I])
    end
	return fig
end;

# ╔═╡ 3a572f35-472e-481f-8c38-5d1453186e8c
plot_tracefunctions(problem, op0; interval=(0.0, 3.0))

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
			@debug "Got $value" op=op
			nnans = length(c.unans)
			if nnans >= 64
				@error "$nnans iterations returned $value, terminating"
				return true
			end
		else
			c.i[] = i = c.i[] + 1
			logprogress(i / (i + 128))
		end
		return false
	end

	function reset!(c::ProgressCallback)
		c.i[] = 0
		empty!(c.unans)
		logprogress(0)
	end
	
	done!(c::ProgressCallback) = logprogress(1)

	logprogress(progress) = @logmsg LogLevel(-1) _id="" progress=progress
end;

# ╔═╡ c17b023d-f5d3-497c-8bf3-913e9d75f6fa
begin
	# Find the lowest-order quadrature that reproducibly achieves the required
	# tolerance throughout the optimization
	us, unans, times = Dict(), Dict(), Dict()
	let callback=ProgressCallback(problem), ipoptions=(
		tol=2.0^(-49), print_level=3, max_wall_time=120.0,
	)
		quadoptions = (; quadcutoff, quadtol, quadstretchmode)
		@info quadoptions
		for quadn in 60, quadscale in 4.0
			reset!(callback)
			quadconfig = (; quadn, quadscale)
			@info quadconfig
			try
				times[quadconfig] = @elapsed for _ in 1:3  # robustness threshold
					reset!(callback)
					u = optimize(
						problem;
						warn_on_handled_exceptions=false,
						quadn, quadscale, quadoptions...,
						ntasks, callback, ipoptions...,
					)
					done!(callback)
					@assert keys(us) == keys(unans)
					if quadconfig in keys(us)
						push!(us[quadconfig], u)
						push!(unans[quadconfig], callback.unans)
					else
						us[quadconfig] = [u]
						unans[quadconfig] = [callback.unans]
					end
				end
				@info "Success!"
			catch e
				@error ReplicaSymmetryBreaking.errormessage(e) exception=e
			end
		end
	end
	sort(times; byvalue=true)
end

# ╔═╡ 753bba73-ccbf-426f-9e54-64f8f3bbe656
let p=nothing
	for (quadconfig, qcunans) in unans
		if !isempty(first(qcunans))
			unan = first(first(qcunans))
			opnan = Orderparameters(problem, unan)
			@show quadconfig
			@show unan
			@show opnan.a
			@show opnan.sqrtdq
			p = plot_orderparameters(problem, opnan)
			break
		end
	end
	p
end

# ╔═╡ 01b947d1-287f-4079-a23d-56e38c540100
begin
	quadconfig = first(keys(us))  # first(keys(times))
	u = first(us[quadconfig])
	qcquad=GaussKronrod(
		quadconfig.quadn, quadcutoff;
		scale=quadconfig.quadscale, tol=quadtol, stretchmode=quadstretchmode,
	)
	qcws = Workspace(u0, problem, qcquad; ntasks)
	(; quadconfig, u)
end

# ╔═╡ df2af3df-3a84-431a-99e7-f98c14660753
op = Orderparameters(problem, u)

# ╔═╡ 7aa22855-90fb-400c-80ee-e15ce653b672
plot_orderparameters(problem, op)

# ╔═╡ aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
plot_tracefunctions(problem, op; interval=(0.0, 3.0), quadconfig...)

# ╔═╡ 438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
objective(u, problem, qcquad, qcws)

# ╔═╡ f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
let du=zero(u)
	dobjective! = ReplicaSymmetryBreaking.make_dobjective(qcws)
	dobjective!(du, u, problem, qcquad, qcws)
	(; du, norm=norm(du, Inf))
end

# ╔═╡ 4a14560b-2a4a-477d-b882-906c06a03504
objectives = vcat(
	(
		let qcquad=GaussKronrod(
				quadconfig.quadn, quadcutoff;
				scale=quadconfig.quadscale, tol=quadtol, stretchmode=quadstretchmode,
			)
			qcws = Workspace(u0, problem, qcquad; ntasks)
			[objective(u, problem, qcquad, qcws) for u in qcus]
		end
		for (quadconfig, qcus) in us
	)...,
)

# ╔═╡ e741586e-fc4f-4406-8152-aa20cef4e37e
objectivemax = maximum(objectives)

# ╔═╡ 76a6bb36-768b-41fd-8b13-fc7450729e3b
objectivemax .- objectives

# ╔═╡ 2f946e07-9726-4517-b094-0facbf73766a
objectivemax - minimum(objectives)

# ╔═╡ Cell order:
# ╟─e33ec073-447d-40b2-905b-ed0db0c5b999
# ╠═6c2156a8-66d4-439a-ae96-9e9307b0dc3c
# ╠═2398a4a2-492c-11ef-024d-0bd270e9187a
# ╠═907b7363-234d-46ae-aacd-6379017238dd
# ╠═bb3f40b4-38be-440c-b04f-e24f52a561b1
# ╠═78830d31-fdf0-4a46-9856-08ecfa22d9a0
# ╠═f41354f8-a0a2-445a-95c9-352fa7290107
# ╠═cab48286-823e-475c-ac76-874f17b3a80b
# ╠═b46036e0-2a7f-491e-abef-c5c956493e3b
# ╠═ba8b018b-ef61-4477-9406-a01e43d205fa
# ╠═bf889797-59cf-4a0d-b2b0-8198c50a2d88
# ╠═a0b2825e-53d7-4ddf-be69-fe7b069648b6
# ╠═9c342e4e-4b50-4b9c-b576-607eb2016417
# ╠═3a572f35-472e-481f-8c38-5d1453186e8c
# ╠═c17b023d-f5d3-497c-8bf3-913e9d75f6fa
# ╠═753bba73-ccbf-426f-9e54-64f8f3bbe656
# ╠═01b947d1-287f-4079-a23d-56e38c540100
# ╠═df2af3df-3a84-431a-99e7-f98c14660753
# ╠═7aa22855-90fb-400c-80ee-e15ce653b672
# ╠═aedf3f12-a426-4f08-b3ae-9a04bbc91e5e
# ╠═438b7e14-1ccc-4bd1-bc38-84faaf2ca1ec
# ╠═f2d0316c-35c2-4aaf-9ba9-9de5e08d3c23
# ╠═4a14560b-2a4a-477d-b882-906c06a03504
# ╠═e741586e-fc4f-4406-8152-aa20cef4e37e
# ╠═76a6bb36-768b-41fd-8b13-fc7450729e3b
# ╠═2f946e07-9726-4517-b094-0facbf73766a
# ╠═bdba40e8-d461-44c7-b077-a58549331b3f
# ╠═ea51f021-afc1-4c4c-b235-4c42d5979525
# ╠═09cce4ed-0807-450f-b1c8-34c9015db9b7
