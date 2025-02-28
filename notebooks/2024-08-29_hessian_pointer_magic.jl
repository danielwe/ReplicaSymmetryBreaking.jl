### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 093ee5e8-c9dd-4bf7-9198-719178260a68
begin
	using Enzyme, ForwardDiff, LinearAlgebra
	# Enzyme.API.runtimeActivity!(true)
end

# ╔═╡ c1aebda6-f0b9-4cd0-ac2b-84f04c584733
# ╠═╡ disabled = true
#=╠═╡
f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
  ╠═╡ =#

# ╔═╡ 9773c395-c5ea-4f58-bbea-89cdc36d72c7
N = 4

# ╔═╡ 32456f95-bd72-4251-849f-f3121ee7fe88
x = rand(N)

# ╔═╡ 296f8136-bf14-4949-86c0-413c3db04397
#=╠═╡
f(x)
  ╠═╡ =#

# ╔═╡ d1c44bf1-fe72-43f0-a711-3c1ca9f7321d
#=╠═╡
function gradf_deferred!(dx, x)
	autodiff_deferred(Reverse, f, Active, Duplicated(x, dx))
	return nothing
end
  ╠═╡ =#

# ╔═╡ b0c1d9e4-fcd1-4d46-8bde-4feaf21cc4bd
#=╠═╡
let	dx=make_zero(x)
	gradf_deferred!(dx, x)
	dx
end
  ╠═╡ =#

# ╔═╡ 7afa235e-507a-4e1a-95f7-e52a41fac52c
#=╠═╡
function hvpf!(hv, v, x, dx=make_zero(x))
	autodiff(Forward, gradf_deferred!, DuplicatedNoNeed(dx, hv), Duplicated(x, v))
	return nothing
end
  ╠═╡ =#

# ╔═╡ fda14951-41dd-4e2d-a4ca-ddbf9bb05655
v = rand(size(x)...)

# ╔═╡ 67539f01-44af-44c8-9114-4bf512531470
#=╠═╡
let dx=make_zero(x), hv=make_zero(dx)
	hvpf!(hv, v, x, dx)
	hv
end
  ╠═╡ =#

# ╔═╡ 1b615c9d-47ad-471c-8f64-66000dc9fcf3
#=╠═╡
function hessianf!(H, x, dx=make_zero(x), V=similar(x, (size(x)..., size(x)...)))
	make_zero!(H)
	copyto!(V, I)
	Hcolsize, Hcolstride = size(H, 1), stride(H, 2)
	Vcolsize, Vcolstride = size(V, 1), stride(V, 2)
	GC.@preserve H V begin
		Hvs = Tuple(
			unsafe_wrap(Array, pointer(H, 1 + Hcolstride * (i - 1)), Hcolsize)
			for i in axes(H, 2)
		)
		Vvs = Tuple(
			unsafe_wrap(Array, pointer(V, 1 + Vcolstride * (i - 1)), Vcolsize)
			for i in axes(V, 2)
		)
		autodiff(
			Forward,
			gradf_deferred!,
			BatchDuplicatedNoNeed(dx, Hvs),
			BatchDuplicated(x, Vvs),
		)
	end
	return nothing
end
  ╠═╡ =#

# ╔═╡ 356a716d-8d5b-4893-9d60-15ed78584b31
#=╠═╡
let H=similar(x, (size(x)..., size(x)...))
	hessianf!(H, x)
	H
end
  ╠═╡ =#

# ╔═╡ 7e467738-04a1-4226-89cd-143a16e1e3a4
function g(x, tmp)
	tmp .= x ./ 2
	# map!(xi -> xi / 2, tmp, x)
	return dot(tmp, x)
end

# ╔═╡ c9a6ae31-9282-402b-b416-6288bde22bce
function g_gradient!(dx, x, tmp)
	dtmp = make_zero(tmp)
	autodiff(Reverse, g, Active, Duplicated(x, dx), Duplicated(tmp, dtmp))
	return nothing
end

# ╔═╡ 2b970356-2530-4131-a80a-f7fc320508a7
function g_gradient_deferred!(dx, x, tmp)
	dtmp = make_zero(tmp)
	autodiff_deferred(Reverse, g, Active, Duplicated(x, dx), Duplicated(tmp, dtmp))
	return nothing
end

# ╔═╡ 17ed4bb8-471b-4dce-ba8b-94f060c1984c
function g_hvp!(hv, x, v, tmp)
	dx = make_zero(x)
	btmp = make_zero(tmp)
	autodiff(
		Forward,
		g_gradient_deferred!,
		Duplicated(dx, hv),
		Duplicated(x, v),
		Duplicated(tmp, btmp),
	)
	return nothing
end

# ╔═╡ 4dc96190-67c2-44d3-b822-e9b6776cefb1
function jvp_forwarddiff!(dy, f!, x, v, tmp)
	T = ForwardDiff.Tag{f!,typeof(x)}
	dualy = ForwardDiff.Dual{T}.(dy, ForwardDiff.Partials.(tuple.(dy)))
    dualx = ForwardDiff.Dual{T}.(x, ForwardDiff.Partials.(tuple.(v)))
	dualtmp = ForwardDiff.Dual{T}.(tmp, ForwardDiff.Partials.(tuple.(zero(tmp))))
    f!(dualy, dualx, dualtmp)
    dy .= ForwardDiff.partials.(cachey)
	return nothing
end

# ╔═╡ fb42de6f-2b7e-49b7-bb28-57e16280cebc
g_hvp_forwarddiff!(hv, x, v, tmp) = jvp_forwarddiff!(hv, g_gradient!, x, v, tmp)

# ╔═╡ ed6e2485-0882-4924-8576-14b240ed880d
let dx=make_zero(x), tmp=similar(x)
	g_gradient!(dx, x, tmp)
	dx
end

# ╔═╡ e5588e14-c4e1-4cfd-8b34-0a99bc25ccba
let hv=make_zero(v), tmp=similar(x)
	g_hvp!(hv, x, v, tmp)
	hv
end

# ╔═╡ a77b4a47-17e9-490d-934c-557e2ebf54c1
let hv=make_zero(v), tmp=similar(x)
	g_hvp_forwarddiff!(hv, x, v, tmp)
	hv
end

# ╔═╡ ea0a120b-c46a-464a-9892-8c524a75383d
# ╠═╡ disabled = true
#=╠═╡
let
	function F(x::Array{Float64}, y::Array{Float64})
	    y[1] = x[1] * x[1] + x[2] * x[1]
	    return nothing
	end
	
	y = [0.0]
	x = [2.0, 2.0]
	
	dy = [0.0]
	dx = [1.0, 0.0]
	
	bx = [0.0, 0.0]
	by = [1.0]
	dbx = [0.0, 0.0]
	dby = [0.0]
	
	autodiff(
	    Forward,
	    (x, y) -> autodiff_deferred(Reverse, F, x, y),
	    Duplicated(Duplicated(x, bx), Duplicated(dx, dbx)),
	    Duplicated(Duplicated(y, by), Duplicated(dy, dby)),
	)
	dbx
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
Enzyme = "~0.12.33"
ForwardDiff = "~0.10.36"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "0f677b4f853a0b2304a82915d8696a7fd0b291d4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random"]
git-tree-sha1 = "056ba87ce8b1739e18ed86789e5ffd43739fc4bc"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.12.33"

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

    [deps.Enzyme.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.EnzymeCore]]
git-tree-sha1 = "8f205a601760f4798a10f138c3940f0451d95188"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.7.8"

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

    [deps.EnzymeCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "9c766fcc7e2ab3fbb56812278d9539b9816c3ef1"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.145+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "583bc7c16682e4292295419432b2ec8a6c3e9c22"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.27.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "2470e69781ddd70b8878491233cd09bc1bd7fc96"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "8.1.0"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "597d1c758c9ae5d985ba4202386a607c675ee700"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.31+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "7249afa1c4dfd86bfbcc9b28939ab6ef844f4e11"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.2"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═093ee5e8-c9dd-4bf7-9198-719178260a68
# ╠═c1aebda6-f0b9-4cd0-ac2b-84f04c584733
# ╠═9773c395-c5ea-4f58-bbea-89cdc36d72c7
# ╠═32456f95-bd72-4251-849f-f3121ee7fe88
# ╠═296f8136-bf14-4949-86c0-413c3db04397
# ╠═d1c44bf1-fe72-43f0-a711-3c1ca9f7321d
# ╠═b0c1d9e4-fcd1-4d46-8bde-4feaf21cc4bd
# ╠═7afa235e-507a-4e1a-95f7-e52a41fac52c
# ╠═fda14951-41dd-4e2d-a4ca-ddbf9bb05655
# ╠═67539f01-44af-44c8-9114-4bf512531470
# ╠═1b615c9d-47ad-471c-8f64-66000dc9fcf3
# ╠═356a716d-8d5b-4893-9d60-15ed78584b31
# ╠═7e467738-04a1-4226-89cd-143a16e1e3a4
# ╠═c9a6ae31-9282-402b-b416-6288bde22bce
# ╠═2b970356-2530-4131-a80a-f7fc320508a7
# ╠═17ed4bb8-471b-4dce-ba8b-94f060c1984c
# ╠═4dc96190-67c2-44d3-b822-e9b6776cefb1
# ╠═fb42de6f-2b7e-49b7-bb28-57e16280cebc
# ╠═ed6e2485-0882-4924-8576-14b240ed880d
# ╠═e5588e14-c4e1-4cfd-8b34-0a99bc25ccba
# ╠═a77b4a47-17e9-490d-934c-557e2ebf54c1
# ╠═ea0a120b-c46a-464a-9892-8c524a75383d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
