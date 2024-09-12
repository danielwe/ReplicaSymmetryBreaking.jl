### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 2b6685d9-8b9f-4f58-9f22-cfc2058cabe8
begin
	using ApproxFun, CairoMakie, DispatchDoctor, FastGaussQuadrature, FFTW, LinearAlgebra, Optimization, OptimizationMOI, OptimizationNLopt, Printf, ProfileView, QuadGK, ScopedValues, SpecialFunctions, StaticArrays
	using Enzyme: Enzyme
	Enzyme.API.runtimeActivity!(true)
end

# ╔═╡ 6ead3e9a-227f-11ef-1940-cbc927e46201
md"""
# Finite-order RSB for the CIM-SK

Trying to reproduce Atsushi here, using the approach from Manuel Schmidt's Ph.D. dissertation (@Schmidt2008Replica). There are a lot of equations, but my approach today is to take the fast track to the recursion relations that I actually need to implement code for and optimize over, and then backfill the derivation and understanding as needed.
"""

# ╔═╡ a2bf328f-393e-4c87-a6c5-ff2e3920ffe0
md"""
## Step 1: Finite-order RSB for the SK model

We start here. That way we can follow @Schmidt2008Replica step by step. Translating to the CIM-SK landscape critical points model should be easier once we have a fully working implementation for the simplest and most studied case.
"""

# ╔═╡ 006edbbb-f585-4a2a-a351-2d0ce9f5e8e2
md"""
The SK model is stated as follows:
```math
\mathbf{H}({S_r}) = -\sum_{r < r'} J_{rr'} S_r S_r' - \sum_r h S_r \, ,
```
where ``S_r \in \{1, -1\}``, and the couplings ``J_{rr'}`` are sampled independently from the probability distribution with the following density:
```math
P(J_{rr'}) = \sqrt{\frac{N}{2 \pi J^2}} \exp\left(-\frac{N}{2 J^2}(J_{rr'} - J_0 / N)^2\right) \, .
```
Here, ``N`` is the total number of spins ``S_r``, and the notation ``\sum_{r < r'}`` implies a sum running once over all unordered and unequal index pairs ``(r, r')`` (i.e., no double counting and no diagonal ``(r, r)`` terms).
"""

# ╔═╡ 463c134f-700b-4fa1-8f1e-bb4efd9a3cb8
md"""
We want to compute the partition function ``Z`` such that we can obtain the free energy per spin ``f = - \langle \log Z \rangle_J / (\beta N)``. Here, we introduced the average over ``P(J_{rr'})``, denoted as ``\langle \cdot \rangle_d`` with ``d`` for disorder. Importantly, we apply the disorder average to observables like ``f``, i.e., to ``\log Z``, rather than to ``Z``. This is the key characteristic of _quenched_ disorder, where the couplings ``J_{rr'}`` are taken to be constant on the timescales of dynamics and observation that we care about, even though they are specified by a probability distribution. In other words, their randomness does not represent thermal fluctuations. Still, for sufficiently large sample sizes ``N`` we expect the distribution of observables to concentrate tightly around their means, so we don't want to define them in a way that depends on the disorder realization. Hence, we incorporate the mean into their definition, such that the realization is integrated out and only the distribution parameters remain as parameters of the observables.
"""

# ╔═╡ 17a08b3a-32b6-43eb-b367-063fde038034
md"""
We compute ``\langle \log Z \rangle_d`` using the replica method: first, we compute the replicated partition function ``Z_{(l)} = Z^l``, where ``Z`` is the partition function for a single replica (i.e., the standard partition function for an SK model instance). Then, we use the relation ``\log Z = \lim_{l \to 0} (Z^l - 1) / l``, which lets us calculate the average of ``\log Z`` via an average over ``Z_{(l)}``:
```math
\langle \log Z \rangle_d = \lim_{l \to 0} \frac{\langle Z_{(l)} \rangle_d - 1}{l} \, .
```
"""

# ╔═╡ 57d22519-9cfc-4610-a67f-1054b3b0aacd
md"""
The ``l`` independent instances comprising ``Z_{(l)}`` are labeled by a replica index ``a \in \{1, ..., l\}``, such that the spin variables are now labeled ``S_r^a``.

> Note that all replicas experience the same disorder realization ``J_{rr'}``. The disorder average is taken after computing the ``l`` times replicated partition for ``l`` replicas with identical couplings. In fact, that's kind of the point: it let's us circumvent any disorder-induced ergodicity breaking by letting an arbitrary number of independent systems equilibrate independently under the same disorder realization.

We also introduce replica-indexed auxilliary fields ``M_a`` and ``q_{ab}``, which enter through Hubbard-Stratonovich transformations to decouple spin-spin interactions in the expression for ``Z_{(l)}`` such that we can carry out the average over the couplings ``J_{rr'}``. We won't enter into the details of these transformations here, but only state the final result for the free energy per spin, which is (see (2.19-2.20) in @Schmidt2008Replica):
```math
\begin{aligned}
&f(M_a, q_{ab}) \\
&= -\frac{\beta J^2}{4} + \lim_{l \to 0} \frac{1}{l} \left(\frac{J_0}{2} \sum_a M_a^2 + \frac{\beta J^2}{2} \sum_{a < b} q_{ab}^2 - \beta^{-1} \log \operatorname{tr} \exp(L(M_a, q_{ab}, S^a))\right) \, ,
\end{aligned}
```
where,
```math
L(M_a,q_{ab},S^a) = \beta \sum_{a} (h + J_0 M_a) S^a + \beta^2 J^2 \sum_{a < b} q_{ab} S^a S^b \, .
```
Here, ``\operatorname{tr}`` denotes the thermodynamic trace, i.e., the sum over all realizations of the ``\{S_r^a\}``; however, the decoupling of sites due to the auxilliary fields means we no longer have to carry a site index ``r``, and consequently ``L`` only contains a single spin variable per replica for us to sum over, ``S^a \in \{1, -1\}``.
"""

# ╔═╡ 62a0d0f4-d5e5-4ecd-aaae-8380ce2fe534
md"""
The values of ``M_a`` and ``q_{ab}`` should be chosen to extremize ``f``, and the resulting self-consistency conditions reveal that ``M_a`` has the represents the mean magnetization, ``M_a = \langle S^a \rangle``, while ``q_{ab}`` is an order parameter representing correlations between replicas, ``q_{ab} = \langle S^a S^b \rangle``. We refer to this quantity as the overlap matrix. Note that because ``(S^a)^2 = 1`` for Ising spins, the diagonal is constant, ``q_{aa} = 1``, which let us pull out the constant term ``-\beta J / 4``. (Observe that the above are thermodynamic averages, not disorder averages.) 
> In contrast, an SK-like model with multicomponent spins (e.g., XY spins or Heisenberg spins) would have continuously variable spin components ``S_\nu^a`` subject to an overall length constraint ``(\mathbf{S}^a)^2 = \sum_\nu (S_\nu^a)^2 = 1``, and would thus obtain non-trivial values of ``q^\nu_{aa} = \langle (S_\nu^a)^2 \rangle``.
"""

# ╔═╡ 95ab5021-113d-4ad7-9b12-ced70b0f9820
md"""
### Replica symmetric solution

The simplest ansatz for making progress is that of full permutation symmetry among the replicas, such that ``q_{ab} = q`` for all ``a \ne b``, and ``M_a = M`` for all ``a``. (Also, ``q_{aa} = d`` for all ``a``, but that goes without saying for Ising spins since we already concluded that ``q_{aa} = 1``). This is called _replica symmetry_ (RS). This ansatz is incorrect: it restricts us to a subspace in order parameter space that does not contain the global optimum. However, let's take a brief detour to derive the resulting self-consistency equations for the SK model, as a first step towards the more elaborate but correct solution.

Substituting ``q_{ab} = q`` and ``q_{aa} = 1``, we can carry out the sums in the free energy and obtain the following expression for the free energy:
```math
f(M, q) = -\frac{\beta J^2}{4} + \frac{J_0}{2} M^2 + \lim_{l \to 0} \left(\frac{(l - 1) \beta J^2}{4} q^2 - (l \beta)^{-1} \log \operatorname{tr} \exp(L(M, q, S^a))\right) \, ,
```
where,
```math
L(M,q,S^a) = \beta\sum_{a} (h + J_0 M) S^a + \frac{\beta^2 J^2 q}{2} \left(\sum_a S^a\right)^2 - \frac{l \beta^2 J^2 q}{2} \, .
```
"""

# ╔═╡ e5089794-3554-446e-b334-2ed9a4516032
md"""
We can pull the final term in ``L``, which does not depend on the ``S^a``, outside the trace and take its logarithm. Together with applying the ``l \to 0`` limit to the ``q^2``-term, we obtain
```math
f(M, q) = \frac{J_0}{2} M^2 - \frac{\beta J^2}{4} (1 - q)^2 - \lim_{l \to 0} (l \beta)^{-1} \log \operatorname{tr} \exp(\tilde{L}(M, q, S^a)) \, ,
```
where,
```math
\tilde{L}(M,q,S^a) = \beta \sum_{a} (h + J_0 M) S^a + \frac{\beta^2 J^2 q}{2} \left(\sum_a S^a\right)^2 \, .
```
"""

# ╔═╡ e3b8887e-226e-4627-ab82-cab722c7a627
md"""
The final step is to decouple the squared sum using yet another Hubbard-Stratonovich transformation by introducing a field ``z`` such that
```math
\exp \frac{1}{2}\left(\beta J \sqrt{q} \sum_a S^a\right)^2 = \int \frac{dz}{\sqrt{2 \pi}} \exp\left(\beta J \sqrt{q} z \sum_a S^a - \frac{z^2}{2}\right) \, ,
```
and thus,
```math
\begin{aligned}
\operatorname{tr} \exp \tilde{L}(M, q, S^a) &= \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} \operatorname{tr} \prod_a \exp \beta (h + J_0 M + J \sqrt{q} z) S^a \\
&= \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} (2 \cosh \beta (h + J_0 M + J \sqrt{q} z))^l \\
&\approx \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} (1 + l \log 2 \cosh \beta (h + J_0 M + J \sqrt{q} z)) \\
& = 1 + l \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} \log 2 \cosh \beta (h + J_0 M + J \sqrt{q} z) \, ,
\end{aligned}
```
where in the third line we expanded to linear order around ``l = 0`` in preparation for taking the ``l \rightarrow 0`` limit. Applying the logarithm and invoking the small-``l`` limit again leaves us with only the second term; as a result,
```math
f(M, q) = \frac{J_0}{2} M^2 - \frac{\beta J^2}{4} (1 - q)^2 - \beta^{-1} \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} \log 2 \cosh \beta (h + J_0 M + J \sqrt{q} z) \, .
```
"""

# ╔═╡ 52fe5149-e50c-48e6-8a00-d78e05369906
md"""
Taking derivatives to locate a critical point, we obtain the self-consistency equations for the replica symmetric saddle point of the SK model:
```math
\begin{aligned}
q &= \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} \tanh^2 \beta (h + J_0 M + J \sqrt{q} z) \, , \\
M &= \int \frac{dz}{\sqrt{2 \pi}} e^{-z^2 / 2} \tanh \beta (h + J_0 M + J \sqrt{q} z) \, .
\end{aligned}
```
Note that a partial integration step is required to obtain the equation for ``q`` as stated from the way it appears when first differentiating ``f``.

Note also that as ``J \to 0``, we recover the Ising ferromagnet, for which we observe that ``q = M^2``. This simply expresses the vanishing covariance between replicas: ``\langle S^a S^b \rangle = \langle S^a \rangle \langle S^b \rangle``.
"""

# ╔═╡ fec9c08f-e827-42c2-97fe-60a23c791436
md"""
Here's some code to study this solution. Since we're not interested in non-disordered systems, we fix ``J = 1`` while still allowing ``\beta``, ``h``, and ``J_0`` to vary. Since this is just quick and dirty code for a case we know is incorrect, I'm keeping this simple and just running fixed point iterations with default tolerances and minimal checks here.
"""

# ╔═╡ 840d726e-708f-4e01-9632-c605d05cfac2
function qMrhs(qM, β, h, J0)
	integrand = let constpart=β * (h + J0 * qM[2]), βsqrtq=(β * sqrt(qM[1]))
		function integrand(z)
			kernel = tanh(constpart + βsqrtq * z)
			Mintegrand = kernel * exp(-z^2 / 2) / sqrt(2π)
			qintegrand = Mintegrand * kernel
			return SVector(qintegrand, Mintegrand)
		end
	end
	return first(quadgk(integrand, -Inf, Inf))
end;

# ╔═╡ ccfe01dd-9879-43ca-bd10-db3fe99a035c
function qM(β, h, J0)
	qMold = SVector(0.0, 0.0)
	qM = qMrhs(qMold, β, h, J0)
	while !(qMold ≈ qM)
		qMold = qM
		qM = qMrhs(qMold, β, h, J0)
	end
	return qM
end;

# ╔═╡ 92a23b3b-5ad1-4dda-919f-ea020051242a
let h=1e-5, J0=0.0
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"T")
	lines!(ax, 0.0..2.0, T -> first(qM(inv(T), h, J0)); label=L"q")
	lines!(ax, 0.0..2.0, T -> last(qM(inv(T), h, J0)); label=L"M")
	axislegend(ax)
	fig
end

# ╔═╡ d48c3912-b7f7-4599-bd3a-d9d52b856cee
md"""
Notice the transition from a paramagnetic phase for ``T > 1`` to something else for ``T < 1``. Notably, magnetization stays small, vanishing as ``h \to 0``, but the overlap ``q`` turns on; that is, states are still unmagnetized on the whole, but their spin polarization patterns are correlated. This indicates that the system is pinned to specific subsets of spin states by the coupling disorder, and even though these states don't have any overall magnetization, they are not arbitrary, and, keeping the disorder realization fixed, different copies of the system will look similar in equilibrium. (Though keep in mind this is what's predicted under the assumption of replica symmetry, which we know is incorrect.)
"""

# ╔═╡ c3da10d3-d997-4047-8c80-d3d7266c071c
md"""
One thing that's a bit unclear to me is the justification for the ``q \ge 0`` assumption that was done implicitly when introducing ``\sqrt{q}``. But let's move on, maybe I'll understand better after working through replica symmetry breaking. (An intuitive a posteriori justification, based on the knowledge that ``q = \langle S^a S^b \rangle`` for every pair of replicas, is that each replica is polarized by the same ``h \ne 0``. However, I'd like something less post-hoc.)
"""

# ╔═╡ c356ae36-56b1-40b9-8e46-d68b03a4ef50
md"""
### Replica symmetry breaking

The RS ansatz is wrong. In the full space of possible overlaps, the RS saddle point is unstable. In the Boltzmann distribution for the SK model, some subsets of high-probability spin glass states are more similar than others. As a result, replicas form clusters where within-cluster overlaps are higher than between-cluster overlaps. This is called _replica symmetry breaking_ (RSB). But it need not stop there! Zooming into a cluster, there may be sub-clusters, within which there are sub-sub-clusters, and so on---in principle ad infinitum. The number of iterations needed to find a saddle point that is not unstable to adding another iteration varies between models. For the SK model it turns out that the full infinite hierarchy is needed (full replica symmetry breaking, or fRSB), while some models only need one iteration (``1``-RSB). A scheme with ``k`` iterations is usually referred to as ``k``-step RSB or ``k``-RSB.

The replicas can be organized as a tree where each node represents a lump in the Boltzmann distribution, and each time you follow a branch to a lower node in the tree you're zooming in on a smaller "lump within the lump", and so on. Each node can be labeled with the overlap ``q_{ab}`` between states belonging to different branches of that node. This is sometimes referred to as the _ultrametric_ structure of the overlap matrix because distances between leaf nodes on a tree satisfy a strengthened triangle inequality known as the ultrametric inequality: ``D(x, z) \le \max\{D(x, y), D(x, z)\}``. The overlap between two replicas is given by their lowest common node, so a metric derived from the overlap satisfies the ultrametric inequality---in fact, due to the tree structure, it holds as an equality. (Since the overlap is larger if the states are more similar, the overlap itself is not a good metric, but we can easily turn it into one, for example ``D_{ab} = d - q_{ab}``.)

> Note that each of the lumps in the description above is thermodynamically isolated, which is to say that thermal fluctuations are confined within one bottom-level lump, known as a _pure state_. This breaking of ergodicity is similar to how the thermal fluctuations of an Ising ferromagnet are confined to either the spin-up or the spin-down state, and provides some intuition for why it's so helpful to introduce replicas when studying glassy systems: they let us sample the entire Boltzmann distribution and probe the hierarchy of correlations within it, something a single system frozen in a pure state can't do. 
"""

# ╔═╡ 3a1afb31-ca60-4655-972b-7e42a1a5eae5
md"""
Ideally, we'd like to solve the fRSB limit of the SK model. This can be done at finite temperature, yielding a variational optimization problem where one must solve a partial differential equation to calculate the objective. That's hairy enough, but the limit breaks down at zero temperature, which is our main interest. Hence, we proceed with studying and writing code for ``k``-RSB for general ``k``, and then push ``k`` as far as we can reach computationally.

> I've done quite a bit of work trying to heal the ``T \to 0`` limit of fRSB through various substitutions and rescalings, but I have yet to write a corresponding working code (though see attempts in `2024-04-10_XYIsing-SKglobalmintransition.jl`). Further, @Schmidt2008Method strongly suggests that my approach won't work---see Fig. 7 and the apparent discontinuity at ``\zeta = 1``, which severs initial condition from the smooth, PDE-obeying rest of the surface. I wonder whether a careful treatment of the limit will yield a mixed procedure where the evolution from ``\zeta = 1`` to ``\zeta = 0`` proceeds by a one finite step of some recursion or difference equation, followed by PDE evolution from ``\zeta = 1 - \epsilon`` to ``\zeta = 0``. Either way, the first step is to implement ``k``-RSB so I can investigate the limiting behavior for myself.
"""

# ╔═╡ c803242d-503d-4af6-b39a-62341ea255f8
md"""
With some additional assumptions about permutation symmetry within and between clusters the picture described above leads to the Parisi solution for Replica symmetry breaking, which imbues the overlap matrix with a particular nested structure of diagonal blocks subdivided by identical smaller diagonal blocks. We won't spill ink on trying to describe this structure in more detail here; see for example @Schmidt2008Replica for details (in particular Section 2.3 and Figure 2.1, although the figure can be slightly misleading because it only illustrates the special case where the RSN hierarchy forms a binary tree). Let's just try to state the resulting free energy optimization problem in a compact but readable way.

With ``k`` steps of RSB and setting ``J = 1`` without loss of generality (we've simply absorb ``J`` into ``\beta`` and rescaled the other parameters accordingly), the free energy of the SK model is
```math
f[\mathbf{m},\mathbf{q},M] = \frac{J_0}{2} M^2 - \frac{\beta}{4} (1 - q_1)^2 + \frac{\beta}{4} \sum_{i = 1}^k m_i (q_i^2 - q_{i + 1}^2) - \mathcal{T}[\mathbf{m},\mathbf{q},M] \, ,
```
where,
```math
\mathcal{T}[\mathbf{m},\mathbf{q},M] = \frac{1}{\beta m_k} \mathcal{I}_{k + 1} \left[\log \mathcal{J}_k\left[\cdots \mathcal{J}_1\left[2 \cosh \beta \left(h + J_0 M + \sum_{i = 1}^{k + 1} z_i \sqrt{\Delta q_i} \right) \right]\right]\right] \, .
```
Here we've introduced the order parameter vectors ``\mathbf{m} = (m_1, \ldots, m_k)`` and ``\mathbf{q} = (q_1, \ldots, q_{k + 1})``, and the auxiliary field vector ``\mathbf{z} = (z_1, \ldots, z_{k + 1})``. We also defined ``\Delta q_i = q_i - q_{i + 1}``, where ``q_{k + 2} = 0`` is implied, and made use of two integral operators ``\mathcal{I}_i`` and ``\mathcal{J}_i``, defined as follows:
```math
\begin{aligned}
\mathcal{I}_i[g(\mathbf{z})] &= \int_{-\infty}^\infty \frac{dz_i}{\sqrt{2 \pi}} e^{-z_i^2 / 2} g(\mathbf{z}) \, , \\
\mathcal{J}_i[g(\mathbf{z})] &= \mathcal{I}_i[g(\mathbf{z})^{m_i / m_{i - 1}}]
\end{aligned}
```
In words, ``\mathcal{I}_i`` denotes a Gaussian integral over the ``i``th field ``z_i``, and ``\mathcal{J}_i`` additionally raises the integrand to the power ``m_i / m_{i - 1}``. It's implied that ``m_0 = 1`` such that ``m_1 / m_0 = m_1``.
"""

# ╔═╡ c6050ea9-836a-4dac-8a39-7be23fbd2dd5
md"""
We thus have ``2k + 2`` paramteres to optimize over: ``k`` from ``\mathbf{m}``, ``k + 1`` from ``\mathbf{q}``, and ``M``. Just a quick note on interpretation: ``q_1`` is the overlap of replicas in the lowest-level clusters, and ``m_1`` represents the size of these clusters; ``q_2`` is the size of clusters at the next level of the hierarchy, with size represented by ``m_2``, and so on. The interpretation of the ``m_i`` is a bit tricky, however: as long as we actually keep a large integer number ``l`` of replicas around, they're monotonically ordered integers greater than ``1``, with ``m_2`` an integer multiple of ``m_1`` and so on, up to ``l`` being an integer multiple of ``m_k``. That is, they actually count the number of replicas within each cluster at a given level of the hierarchy, with an implied ``m_0 = 1`` for the leaf nodes of the tree (which trivially have self-overlap ``q_0 = 1``). However, when taking the limit ``l \to 0``, everything gets turned on its head: now ``1 > m_1 > m_2 \cdots > m_k > 0``. Perhaps they can now be seen as the reciprocals of said integer sizes? I don't know. Let's move on.
"""

# ╔═╡ 73f96e10-d0ca-40f4-a7c9-8a36962ff8ef
md"""
In the above formulation, the nested integrals within the ``\mathcal{J}_i`` are successively integrating out one variable each from a multivariate integrand, which is then raised to some power before integrating out the next variable. For both analytical and computational purposes it would be much better to recursively integrate and transform a univariate function. Fortunately, such a rewriting can be achieved follows:

- First, we introduce the effective fields ``h_i`` such that ``h_1`` is the effective field in the innermost integrand, while the ``h_i`` for ``i > 1`` drop successive ``z_i``-terms. That is,
  ```math
  h_i = h + J_0 M + \sum_{j = i}^{k + 1} z_j \sqrt{\Delta q_j} \, .
  ```
- Since ``h_{i + 1}`` is independent of ``z_i``, we can use the relation ``h_i = h_{i + 1} + z_i \sqrt{\Delta q_i}`` to perform an integral substitution where
  ```math
  dh_i = \sqrt{\Delta q_i} dz_i \, .
  ```
- Now, if we rewrite the integral operators from above as
  ```math
  \mathcal{I}[g(\mu'); \mu, \Delta q] = \int_{-\infty}^\infty \frac{d\mu'}{\sqrt{2 \pi \Delta q}} e^{-(\mu - \mu')^2 / 2 \Delta q} g(\mu') \, ,
  ```
  we obtain a recursion where each step consists of univariate integration followed by exponentiation and transforms a function of ``h_i`` into a function of ``h_{i + 1}`` (although since these fields are now only integration variables we drop the subscripts and use generic variable names ``\mu`` and ``\mu'`` when stating the recursion relation; this also avoids confusion with the external field ``h`` from the Hamiltonian). We can write the recursion this as follows:
  ```math
  g_i(\mu) = \mathcal{I}[g_{i - 1}(\mu')^{m_i / m_{i - 1}}; \mu, \Delta q_i] \, ,
  ```
  with initial condition ``g_0(\mu) = 2 \cosh \beta \mu``.
- This lets us express the trace term in the free energy as follows:
  ```math
  \mathcal{T}[\mathbf{m},\mathbf{q},M] = \frac{1}{\beta m_k} \mathcal{I} [\log g_k(\mu); h + J_0 M, q_{k + 1}] \, ,
  ```
  Note how this transformation moved the dependence on ``h`` and ``M`` from the innermost integrand to the outermost convolution kernel.
"""

# ╔═╡ d10ad4af-d383-429d-a3b0-b679229f6b65
md"""
#### Low temperature

The expression for ``f[\mathbf{m},\mathbf{q},M]`` looks like it will diverge in the zero-temperature limit, i.e., as ``\beta \to \infty``. At the same time, ``\mathcal{T}``-term seems to vanish in this limit. However, we need to keep in mind that the values of ``\mathbf{m}``, ``\mathbf{q}``, and ``M`` should be chosen to extremize ``f``, and it's not immediately clear what the ``\beta \to \infty`` limit will look like along the resulting curve of solutions.

The form of both ``f`` and ``\mathcal{T}`` are highly suggestive of a parameter rescaling ``a_i = \beta m_i``. Note that since ``m_i \in [0, 1]``, we have ``a_i \in [0, \beta]``, becoming entire positive half-line as ``\beta \to \infty``. Nevertheless, it turns out that these parameters are well behaved in the limit. Let us rewrite the free energy expressions:
```math
f[\mathbf{a},\mathbf{q},M] = \frac{J_0}{2} M^2 - \frac{\beta}{4} (1 - q_1)^2 + \frac{1}{4} \sum_{i = 1}^k a_i (q_i^2 - q_{i + 1}^2) - \mathcal{T}[\mathbf{a},\mathbf{q},M] \, ,
```
where,
```math
\mathcal{T}[\mathbf{a},\mathbf{q},M] = \frac{1}{a_k} \mathcal{I} [\log g_k(\mu); h + J_0 M, q_{k + 1}] \, ,
```
and ``g_k(\mu)`` is defined through the recursion
```math
g_i(\mu) = \mathcal{I}[g_{i - 1}(\mu')^{a_i / a_{i - 1}}; \mu, \Delta q_i] \, ,
```
and ``g_0(\mu) = 2 \cosh \beta \mu``. Note that ``a_0 = \beta`` such that the innermost integrand is
```math
\begin{aligned}
g_0(\mu)^{a_1 / \beta} &= \left(2 \cosh \beta \mu\right)^{a_1 / \beta} \\
&= \exp\left(a_1 \log(2 \cosh \beta \mu) / \beta\right) \\
&\to \exp(a_1 |\mu|) \, .
\end{aligned}
```
Let's verify the convergence to this limit with some plots for ``a_1 = 1``:
"""

# ╔═╡ a0fc11ad-99e9-4abb-96f1-c1c7a92e4d70
let
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"\mu", ylabel=L"(2 \cosh \beta \mu)^{1 / \beta}")
	for β in [1.0, sqrt(10.0), 10.0, 10sqrt(10.0)]
		βstr = @sprintf("%.2f", β)
		lines!(ax, -1..1, h -> (2cosh(β * h))^inv(β); label=L"\beta = %$(βstr)")
	end
	lines!(ax, -1..1, h -> exp(abs(h)); label=L"\beta = \infty")
	axislegend(ax)
	fig
end

# ╔═╡ 9df607cb-4d4b-4d25-9a06-bcf12dc342a6
md"""
There's still the term ``-\beta (1 - q_1)^2 / 4``, which must be handled analytically.

> This is a good time to take a step back and think about what we mean when we say "extremize" or finding the "optimum" in everything written above. Recall from the RS solution that the sum over ``q_{ab}^2`` resulted in a term ``\propto (l - 1) q^2`` when assuming replica symmetry. Then spooky things happen. The number of replicas ``l`` is initially some finite integer greater than ``1``, but then we take the limit ``l \to 0``, and in some sense the number of pairs of replicas goes negative (since the number ``l (l - 1) / 2`` of above-diagonal elements in the overlap matrix changes sign), and as a result, the ``q^2``-term in the free energy changes sign.
>
> For positive integer ``l``, when the free energy was a physical quantity, we wanted to minimize with respect to ``q``, and the ``q^2``-term was indeed bounded from below as it should be. However, if we want to stay on the same saddle point while taking ``l \to 0``, we need to change from minimizing to maximizing. Indeed, in the final expression, the ``q^2``-term is bounded from above, not below, so minimization won't work. As @Schmidt2008Replica concludes: "Loosely speaking, a minimization with respect to a _negative number of equal parameters_ results in a maximization with respect to the value of the parameters."
>
> This argument generalizes to replica symmetry breaking: there are now multiple values of ``q_i``, but they all enter ``f[\mathbf{m},\mathbf{q},M]`` through a "negative" number of terms, ``m_i - m_{i - 1} < 0``. Thus, when extremizing ``f``, we should _maximize_ with respect to ``\mathbf{q}`` and _minimize_ with respect to ``M``. (It's less clear which way to extremize in ``m_i``-space, but it doesn't matter much in practice, we don't need to know for analytical purposes, and numerically we're just searching for a root of the gradient of ``f``.)
"""

# ╔═╡ 08f3a66e-a6c3-4d76-b817-ad2b27a0ce00
md"""
The maximum of the term ``-\beta (1 - q_1)^2`` is found at ``q_1 = 1``, and since this term would otherwise diverge to ``-\infty`` and dominate all other terms as ``\beta \to \infty``, it's clear that maximizing ``f`` with respect to ``\mathbf{q}`` must imply ``\lim_{\beta \to \infty} q_1 = 1``. Further, if we expand ``q_1 = 1 + b_1 / \beta + \mathcal{O}(1 / \beta^2)``, we get ``-\beta (1 - q_1)^2 \to -b_1^2 / \beta \to 0``; in other words, the term as a whole must vanish as ``\beta \to 0``, it cannot leave a finite residual from a cancellation between the diverging and the vanishing factor (this argument assumes that ``q_1`` depends analytically on ``1 / \beta`` in a neighborhood of ``1 / \beta = 0``).
"""

# ╔═╡ ccd108bf-28e9-4385-b9fb-be687be17f09
md"""
#### Asymptotic renormalization group

We already observed that ``g_0(\mu) \to \exp(a_1 |\mu|)`` as ``\beta \to \infty``. We can build on this approach to further rewrite the RSB recursion in terms of the deviation from an asymptotic form of the ``g_i``, rather than the ``g_i`` itself. This is both analytically informative and leads to more efficient numerics.

It can be shown that, asymptotically for ``|\mu| \gg 1``,
```math
g_i(\mu) \sim \exp(a_i |\mu|) \, .
```
Note that, here as in several cases above, the expression we state is only valid for the Ising spin glass and is considerably simpler than for models with more spin components. We'll keep working with Ising spins only until we've worked out that case fully before adding back in any additional complexity needed to treat the CIM.
"""

# ╔═╡ af9c0340-17f2-4fc8-9b0b-aa4816b07d59
md"""
Now we introduce the kernel correction function ``\mathcal{G}_i``, such that
```math
g_i(\mu) = \exp\left[a_i\left(\mathcal{G}_i(\mu) + |\mu| + \frac{1}{2} \sum_{j = 1}^i a_j \Delta q_j\right)\right] \, .
```
Let's work out how this transforms under the recursion. Plugging in, we get
```math
\begin{aligned}
&\exp\left[a_i\left(\mathcal{G}_i(\mu) + |\mu| + \frac{1}{2} \sum_{j = 1}^i a_j \Delta q_j\right)\right] \\
&= \mathcal{I}\left[\exp\left[a_i\left(\mathcal{G}_{i - 1}(\mu') + |\mu'| + \frac{1}{2} \sum_{j = 1}^{i - 1} a_j \Delta q_j\right)\right]; \mu, \Delta q_i \right] \, ,
\end{aligned}
```
where we notice that the effect of the exponent ``a_i / a_{i - 1}`` in the recursion map is that the exponent prefactor is ``a_i`` on both sides, i.e., ``a_{i - 1}`` simply drops out of this recursive step. Isolating ``\mathcal{G}_i(\mu)`` on the LHS, we obtain
```math
\mathcal{G}_i(\mu) = \frac{1}{a_i} \log \mathcal{I}\left[\exp\left[a_i\left(\mathcal{G}_{i - 1}(\mu') + |\mu'| - |\mu| - a_i \Delta q_i / 2\right)\right]; \mu, \Delta q_i \right] \, .
```
Notice how we could cancel all but one of the ``a_j \Delta q_j``-terms against the terms from the LHS. Also, we keep the ``\mu'``-independent term in the integrand---we could pull them out, but they make the integral more well-behaved, keeping the exponential from blowing up within the effective support of the the Gaussian envelope.

The initial condition is
```math
\begin{aligned}
\mathcal{G}_0(\mu) &= \frac{1}{\beta} \log(2 \cosh \beta \mu) - |\mu| \\
&= \frac{1}{\beta} \log\left(1 + e^{-2 \beta |\mu|}\right) \, ,
\end{aligned}
```
which vanishes as ``\beta \rightarrow \infty``. (Note that ``a_0 = \beta`` since ``m_0 = 1``.)

Since ``\mathcal{G}_0(\mu)`` is even, so is ``\mathcal{G}_i(\mu)`` for all ``i``. (To calculate ``\mathcal{G}_i(-\mu)``, substitute ``\mu' \to -\mu'`` and observe that the resulting integral is identical to the one that determines ``\mathcal{G}_i(\mu)``, assuming ``\mathcal{G}_{i - 1}(-\mu') = \mathcal{G}_{i - 1}(\mu')``, so by induction all the ``\mathcal{G}_i`` are symmetric.) Thus, we only have to compute these functions at nonnegative values, simplifying the recursive convolutions slightly.
"""

# ╔═╡ 8d3d6bad-4eab-4061-9a67-3be4e89b72a0
md"""
Plugging this back into the expression for the free energy, we find,
```math
\begin{aligned}
f[\mathbf{m},\mathbf{q},M] &= \frac{J_0}{2} M^2 - \frac{\beta}{4} (1 - q_1)^2 + \frac{1}{4} \sum_{i = 1}^k a_i (q_i^2 - q_{i + 1}^2) \\
&- \mathcal{I}\left[\mathcal{G}_k(\mu) + |\mu| + \frac{1}{2} \sum_{i = 1}^k a_k \Delta q_k; h + J_0 M, q_{k + 1}\right] \, ,
\end{aligned}
```
which we can simplify by noting that the ``\mu``-independent term can be pulled out from the convolution integral, since the convolution kernel has measure one; hence,
```math
\begin{aligned}
f[\mathbf{m},\mathbf{q},M] &= \frac{J_0}{2} M^2 - \frac{\beta}{4} (1 - q_1)^2 + \frac{1}{4} \sum_{i = 1}^k a_i ((q_i - 1)^2 - (q_{i + 1} - 1)^2) \\
&- \mathcal{I}\left[\mathcal{G}_k(\mu) + |\mu|; h + J_0 M, q_{k + 1}\right] \, ,
\end{aligned}
```
where we used that ``(q_i - 1)^2 - (q_{i + 1} - 1)^2 = q_i^2 - q_{i + 1}^2 - 2q_i + 2q_{i + 1} = q_i^2 - q_{i + 1}^2 - 2\Delta q_i``.
"""

# ╔═╡ 0c7a4d65-1c78-4c80-a86a-7357514f775b
md"""
### Implementation of finite-order RSB for the SK model

> On closer inspection, using ``\mathcal{G}_i`` instead of ``g_i`` doesn't buy us any efficiency directly; we still need to exponentiate and get an integrand ``\propto g_i`` for the convolution. According to @Schmidt2008Replica, the speed-up comes from making use of the boundary condition ``\mathcal{G}_i(\mu) \rightarrow 0`` as ``|\mu| \rightarrow \infty`` (Section 2.4.2, page 29). Unfortunately, no details about this implementation are provided.

It's time for code. Let's start by getting the recursion up and running for arbitrary ``a_i, q_i``. Let's take a closer look at the form of the integrals, putting as much into the integrand on the RHS as possible, like @Schmidt2008Replica. First, due to symmetry we only need to consider positive ``|\mu|``, so let's make this explicit. Second, the integrand has a singularity at ``|\mu'| = 0``, so numerical quadrature will want us to split the integral there, compelling us to exploit that ``\mathcal{G}'(-\mu') = \mathcal{G}(\mu')``. Thus,
```math
\begin{aligned}
&e^{a \mathcal{G}(\mu)} \\
&= \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \exp\left(-\frac{(\mu' - |\mu|)^2}{2 \Delta q} + a\left(\mathcal{G}'(\mu') + |\mu'| - |\mu| - \frac{1}{2} a \Delta q\right)\right) \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{a(\mathcal{G}'(\mu') - a \Delta q / 2)} \\
&\times \left[\exp\left(-\frac{(\mu' - |\mu|)^2}{2 \Delta q} + a\left(\mu' - |\mu|\right)\right) + \exp\left(-\frac{(\mu' + |\mu|)^2}{2 \Delta q} + a\left(\mu' + |\mu|\right) - 2a|\mu|\right)\right] \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{a\mathcal{G}'(\mu')} \\
&\times \left[\exp\left(-\frac{(\mu' - |\mu| - a \Delta q)^2}{2 \Delta q}\right) + \exp\left(-\frac{(\mu' + |\mu| - a \Delta q)^2}{2 \Delta q} - 2a|\mu|\right)\right] \\ 
&= \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \Theta(\mu') e^{a\mathcal{G}'(\mu')} \exp\left(-\frac{(|\mu| - \mu' + a \Delta q)^2}{2 \Delta q}\right) \\
&+ e^{-2a|\mu|} \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \Theta(-\mu') e^{a\mathcal{G}'(\mu')} \exp\left(-\frac{(|\mu| - \mu' - a \Delta q)^2}{2 \Delta q}\right)
\end{aligned}
```
"""

# ╔═╡ 18866c6e-b147-4cda-9dcd-c4edcefa5651
md"""
The two last expressions above suggest two different ways of looking at the integral:

* As a one-sided integral over ``e^{a\mathcal{G}}`` weighted by a mixture of shifted (and truncated by integration limits) Gaussians
* As a sum of convolutions of each half of ``e^{a\mathcal{G}}`` with differently shifted Gaussians
  - Since ``e^{a\mathcal{G}'}`` is not square integrable, in practice we'd want to consider convolutions with ``e^{a\mathcal{G}'} - 1`` and deal with the constant term analytically, giving us a couple of ``\operatorname{erf}``s

It's tempting to gravitate towards the convolution formulation to take advantage of the ``\mathcal{O}(N \log N)`` scaling of FFT rather than the ``\mathcal{O}(N^2)`` scaling of a naive approach (here ``N`` is the number of points at which the functions are sampled in their discrete representation). One potential problem with this is that ``\Theta(\mu')(e^{a\mathcal{G}'(\mu')} - 1)`` is discontinuous at ``\mu' = 0``, requiring a lot of high-frequency content for an accurate frequency space representation. For example, the Fourier transform of ``\Theta(\mu')e^{-b\mu'}`` asymptotes to ``1 / i\omega`` at large ``|\omega|``, a very slow decay. Moreover, convolutions and FFT require uniform sampling, while numerical integration (quadrature) can sample non-uniformly to achieve better accuracy with fewer samples. Thus, the gain in scaling might ultimately be negated by the need for a larger ``N``.
"""

# ╔═╡ ba321c1e-d0da-4339-9426-0683d9979745
begin
	const β = inv(0.03)
	const k = 37
	const ξ0 = a2ξ(β)
	const ξs = (range(ξ0, zero(ξ0), k + 2)[begin:(end - 1)]).^inv(sqrt(2))
	const as_ = ξ2a.(ξs[(begin + 1):end])
	const qs_ = if isinf(β)  # q1 = 1 fixed?
		sin.(range(π / 2, 0.0, k + 2)[begin:(end - 1)])
	else
		sin.(range(π / 2, 0.0, k + 3)[(begin + 1):(end - 1)])
	end
	const dqs_ = diff(-qs_)
end;

# ╔═╡ d6bbf6f7-8b39-46fc-b15c-3a9f53f630d9
md"""
#### Reference implementation of recursive integrals

Using QuadGK.jl with tight tolerances along with ApproxFun.jl to obtain an accurate implementation of the recursive integrals, i.e., the forward pass. This may be slow and poorly suited for optimization, but it's relatively straightforward to implement.
"""

# ╔═╡ 6f44e78b-05fb-4d63-b540-de2cdd4e4b81
let quadgkcounter=ScopedValue{Base.RefValue{Int}}()
	# Scoped integrand call counting for QuadGK
	global function quad(args...; atol=0.0, rtol=16^(-11), order=18, kwargs...)
		I, E, count = quadgk_count(args...; atol, rtol, order, kwargs...)
		tol = max(atol, rtol * norm(I))
		if E > 2tol  # QuadGK is a mystery sometimes, hence factor of 2
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
end;

# ╔═╡ 7c955c2a-54b1-489d-b953-629f6d8878d6
# Heuristic to maintain accuracy all the way to G_k
const μmax = 8.2;  #6.2 + 8.5sqrt(maximum(dqs_));

# ╔═╡ 5036c3f1-e753-47db-8612-0dd3becb600c
const Gdomain = 0.0..μmax;  # For plots

# ╔═╡ 92fbae7f-07ee-48c6-aed7-0865b05df6d7
@stable function G0(μ)
	isinf(β) && return zero(μ)
	return log1p(exp(-2β * abs(μ))) / β
end;

# ╔═╡ ef8f4ca8-5f80-4c5f-9490-f97f34db517e
@stable function wgaussian(dx, var, expshift=nothing)
	exponent = -dx^2 / (2var)
	if !isnothing(expshift)
		exponent += expshift
	end
	return exp(exponent) / sqrt(2(π * var))
end;

# ╔═╡ 9d1c8bf4-153c-47b7-9397-11cfbc9ccc1a
@stable function Gintegrand(G, μp, μabs, a, dq, expshift=nothing)
	# TODO: Trouble ensues for large μabs and small dq (e.g., large k => small dq).
	# Investigate. Substitute?
	dμp = μp - a * dq
	es = a * G(μp)
	if !isnothing(expshift)
		es += expshift
	end
	es2 = es - 2a * μabs
	return wgaussian(dμp - μabs, dq, es) + wgaussian(dμp + μabs, dq, es2)
end;

# ╔═╡ d399f6b4-3c9d-4c07-aa7a-f6f03964032a
begin
	# Scale integration variable by 8σ of kernel support
	@stable Gmapscale(μabs, a, dq) = μabs + a * dq + 8sqrt(dq)
	# EnzymeRules.inactive(::typeof(Gmapscale), args...) = nothing
end;

# ╔═╡ ee72bb96-446e-41f3-9016-b68e1f4e58b8
@stable function Gmap_quadgk_lazy(G, a, dq)
	return let G=G, a=a, dq=dq
		function Gnext_quadgk(μ)
			μabs = abs(μ)
			scale = Gmapscale(μabs, a, dq)
			I, _ = quad(x -> Gintegrand(G, x * scale, μabs, a, dq), 0.0, Inf)
			return log(I * scale) / a
		end
	end
end;

# ╔═╡ b571d11f-e7ac-4ed0-8e9f-c5f00a7dc295
begin
	@stable function μtrunc(G)
		# Do a binary search for a good location to truncate an interpolation of G
		local μ
		μl, μr = 0.0, 10.0
		Gmax = G(μl)
		while true
			μ = (μl + μr) / 2
			r = G(μ) / Gmax
			if r >= 2.0^(-26)
				μl = μ
			elseif r <= 2.0^(-27)
				μr = μ
			else
				break
			end
		end
		return 2μ
	end
	# EnzymeRules.inactive(::typeof(μtrunc), args...) = nothing
end;

# ╔═╡ ec713b7c-626b-46c8-a3ea-1be1afaa3c50
@stable function Gmap_quadgk(G, a, dq)
	_Gnext_quadgk = Gmap_quadgk_lazy(G, a, dq)
	# Return sanitized interpolant
	return let μm=μtrunc(_Gnext_quadgk), itp=Fun(_Gnext_quadgk, Chebyshev(0.0..μm))
		function Gnext_quadgk(μ)
			μabs = abs(μ)
			(μabs > μm) && return zero(itp(zero(μabs)))
			ret = itp(μabs)
			return max(ret, zero(ret))
		end
	end
end;

# ╔═╡ 5ac52242-0e89-4a6a-a254-bb8f1685348d
const G1_quadgk_lazy = Gmap_quadgk_lazy(G0, as_[1], dqs_[1]);

# ╔═╡ 6cacb213-1886-4385-bb52-8b34dcda686e
begin
	const Gs_quadgk = let 
		ncalls, Gs = countintegrandcalls() do
			Gs_ = [G0, Gmap_quadgk(G0, as_[1], dqs_[1])]
			for i in 2:k
				push!(Gs_, Gmap_quadgk(last(Gs_), as_[i], dqs_[i]))
			end
			Gs_
		end
		@show ncalls
		Gs
	end
	const G1_quadgk = Gs_quadgk[2]
end;

# ╔═╡ 1a064d9a-ff05-4097-891f-0c97b1493850
function Gplot(I::Interval, Gs...; legend=true, legendkws...)
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"\mu")
	local p
	for (i, G) in enumerate(Gs)
		p = lines!(ax, I, G; label=L"\mathcal{G}_%$(i - 1)(\mu)")
	end
	if legend
		axislegend(ax; legendkws...)
	end
	return Makie.FigureAxisPlot(fig, ax, p)
end;

# ╔═╡ 69dea82e-fa2a-480b-92d3-a184f87a48eb
Gplot(Gdomain, G0, G1_quadgk_lazy)

# ╔═╡ 2606a0a5-6062-4f7b-97ea-80dfe1aa4927
md"""
#### DFT approach

Let's give the DFT approach a try. Due to realness and symmetry, the cosine transform is the obvious way to find the frequency space representation of the full ``e^{a \mathcal{G}'} - 1``. Can we use this single transform to calculate both convolutions? No---we also need the sine transform; each half will correspond to their sum and difference, respectively. At that point, it's probably better to just do a plain `rfft` of the one-sided function ``\Theta(\mu') (e^{a \mathcal{G}'(\mu')} - 1)`` and exploit the fact that reflection around the origin in position space is equivalent to the same in frequency space: ``\mathcal{F}\{f(-t)\}(\omega) = \mathcal{F}\{f(t)\}(-\omega)`` (with the obvious abuse of notation). Let's do some experiments with the correct Gaussian envelope and a simple exponential decay as the signal taking the place of ``e^{a \mathcal{G}'(\mu')} - 1``.
"""

# ╔═╡ 31b8c01c-309f-4293-937a-2e5bfd055e86
let n=32, rate=10.0, λ=5.0, a=16.0, σ=0.1, tc=-(a * σ^2)
	T = n / rate
	ts = (0:(n - 1)) ./ rate
	fs = rfftfreq(n, rate)
	
	sR(t) = exp(-λ * abs(t)) * (1 + sign(t)) / 2
	sRf(f) = inv(λ + 2π * im * f)
	signalR = sR.(ts)
	signalRf = rfft(signalR) / rate

	_wR(t) = exp(-(t - tc)^2 / (2σ^2)) / sqrt(2π * σ^2)
	wR(t) = _wR(t) + _wR(t - T)
	wRf(f) = exp(-2π * im * tc * f - (2π * σ * f)^2 / 2)
	envelopeR = wR.(ts)
	envelopeRf = rfft(envelopeR) / rate
	
	convolutionRf = signalRf .* envelopeRf
	convolutionR = irfft(convolutionRf, n) * rate
	cRf(f) = sRf(f) * wRf(f)
	cR(t) = exp(λ * (tc + σ^2 * λ / 2 - t)) * (1 - erf((tc + σ^2 * λ - t) / sqrt(2σ^2))) / 2

	sL(t) = exp(-λ * abs((T - t) % T)) * (1 + sign((T - t) % T)) / 2
	sLf(f) = inv(λ - 2π * im * f)
	signalL = sL.(ts)
	signalLf = rfft(signalL) / rate

	_wL(t) = exp(-(t + tc)^2 / (2σ^2)) / sqrt(2π * σ^2)
	wL(t) = _wL(t) + _wL(t - T)
	wLf(f) = exp(2π * im * tc * f - (2π * σ * f)^2 / 2)
	envelopeL = wL.(ts)
	envelopeLf = rfft(envelopeL) / rate
	
	convolutionLf = signalLf .* envelopeLf
	convolutionL = irfft(convolutionLf, n) * rate
	cLf(f) = sLf(f) * wLf(f)
	cL(t) = exp(-λ * (-tc - σ^2 * λ / 2 - t)) * (1 + erf((-tc - σ^2 * λ - t) / sqrt(2σ^2))) / 2

	scaleL(t) = exp((-2a) * t)
	convolutionsum = convolutionR .+ scaleL.(ts) .* convolutionL
	csum(t) = cR(t) + scaleL(t) * cL(t)

	tI = 0..T
	fI = first(fs)..last(fs)

	lineoptions = (; linestyle=:dash)
	
	fig = Figure(; size=(600, 900), fontsize=11)
	ax = (
		Axis(fig[1, 1]), Axis(fig[1, 2]),
		Axis(fig[2, 1]), Axis(fig[2, 2]),
		Axis(fig[3, 1]), Axis(fig[3, 2]),
		Axis(fig[4, 1]), Axis(fig[4, 2]),
		Axis(fig[5, 1]), Axis(fig[5, 2]),
		Axis(fig[6, 1]; xlabel=L"t"), Axis(fig[6, 2]; xlabel=L"f"),
		Axis(fig[7, 1:2]),
	)
	linkxaxes!(ax[1:2:end]...)
	linkxaxes!(ax[2:2:end]...)
	
	stem!(ax[1], ts, signalR; label="signal R")
	lines!(ax[1], tI, sR; lineoptions...)
	stem!(ax[2], fs, real(signalRf); label="real ampl.")
	stem!(ax[2], fs, imag(signalRf); label="imag ampl.")
	lines!(ax[2], fI, real ∘ sRf; lineoptions...)
	lines!(ax[2], fI, imag ∘ sRf; lineoptions...)

	stem!(ax[3], ts, envelopeR; label="envelope R")
	lines!(ax[3], tI, wR; lineoptions...)
	stem!(ax[4], fs, real(envelopeRf); label="real ampl.")
	stem!(ax[4], fs, imag(envelopeRf); label="imag ampl.")
	lines!(ax[4], fI, real ∘ wRf; lineoptions...)
	lines!(ax[4], fI, imag ∘ wRf; lineoptions...)

	stem!(ax[5], ts, convolutionR; label="convolution R")
	lines!(ax[5], tI, cR; lineoptions...)
	stem!(ax[6], fs, real(convolutionRf); label="real ampl.")
	stem!(ax[6], fs, imag(convolutionRf); label="imag ampl.")
	lines!(ax[6], fI, real ∘ cRf; lineoptions...)
	lines!(ax[6], fI, imag ∘ cRf; lineoptions...)

	stem!(ax[7], ts, signalL; label="signal L")
	lines!(ax[7], tI, sL; lineoptions...)
	stem!(ax[8], fs, real(signalLf); label="real ampl.")
	stem!(ax[8], fs, imag(signalLf); label="imag ampl.")
	lines!(ax[8], fI, real ∘ sLf; lineoptions...)
	lines!(ax[8], fI, imag ∘ sLf; lineoptions...)

	stem!(ax[9], ts, envelopeL; label="envelope L")
	lines!(ax[9], tI, wL; lineoptions...)
	stem!(ax[10], fs, real(envelopeLf); label="real ampl.")
	stem!(ax[10], fs, imag(envelopeLf); label="imag ampl.")
	lines!(ax[10], fI, real ∘ wLf; lineoptions...)
	lines!(ax[10], fI, imag ∘ wLf; lineoptions...)

	stem!(ax[11], ts, convolutionL; label="convolution L")
	lines!(ax[11], tI, cL; lineoptions...)
	stem!(ax[12], fs, real(convolutionLf); label="real ampl.")
	stem!(ax[12], fs, imag(convolutionLf); label="imag ampl.")
	lines!(ax[12], fI, real ∘ cLf; lineoptions...)
	lines!(ax[12], fI, imag ∘ cLf; lineoptions...)

	stem!(ax[13], ts, convolutionsum; label="weighted sum of convolutions")
	lines!(ax[13], tI, csum; lineoptions...)

	legendoptions = (; patchsize=(12.5f0, 12.5f0), rowgap=0)
	
	axislegend.(ax[1:2:end]; position=:ct, legendoptions...)
	axislegend.(ax[2:2:end]; position=:rt, legendoptions...)
	
	fig
end

# ╔═╡ 44fdc36d-9c80-4c83-96f1-1fd37c0d9146
md"""
This seems to work. The DFT isn't a good approximation of the Fourier transform of ``e^{-\lambda t}`` at high frequencies, but it seems like a Gaussian covers a multitude of sins with its fast decay, resulting in a good approximation of the convolved DFTs. We just need to be aware of two things:
* The cyclic nature of DFT convolutions result in the convolved signal bouncing back up at the upper end of the interval. The solution is simply to clamp the upper half of the signal to zero. This sets a lower bound for how wide the interval needs to be: convolving with the Gaussian should never appreciably mix the lower and upper half, i.e., the support of the Gaussian should never both reach into the upper half and overlap with non-infinitesimal values in the lower half.
  > Note that for the convolved signals, the upper half of the interval does _not_ represent values of any function of interest evaluated at ``\mu < 0`` (i.e., ``t < 0``), since the prescription for evaluating the transformed signal at ``\mu`` uses convolutions evaluated at ``|\mu|``. In other words, non-infinitesimal values in the upper half are simply junk for our purposes. Don't go looking for their significance, e.g., to automatically produce a symmetric signal if we weight the convolutions appropriately. They can't do it. They represent values at ``|\mu| < 0``, i.e., nothing.
* When splitting the signal into left and right parts, we need to be careful to divide the center value equally between the two parts, otherwise we're not treating odd vs. even parts correctly and the DFT representations will not be complex conjugates of each other. Note that the final output signal (the weighted sum of convolutions) by construction does not respect this prescription, but has the full center value in its first entry. This is expected and necessary, since this signal is due for a nonlinear transformation; however, post transformation, the center value must once again be chopped in half before the next round of convolutions.
"""

# ╔═╡ e25bbce0-e73b-4169-90b6-dfead0d76acf
md"""
Further, the experiments reveal something that is clear from the equations if you look closer: the second convolution is equal to the first with ``|\mu|`` replaced by ``-|\mu|`` (substitute ``\mu' \to -\mu'`` to make it apparent). Thus, their Fourier transforms are just complex conjugates of each other. In other words, once we've computed the right convolution we have everything we need: just take the upper half, reverse, scale by weight, and add on top of lower half (remember the off-by-one: the upper half doesn't include the center value).
"""

# ╔═╡ 59cc072d-a823-417c-ba0f-468531ac337b
md"""
Combining this insight with pulling out the analytical part, we can express the integral for ``e^{a \mathcal{G}}`` as follows:
```math
e^{a \mathcal{G}(\mu)} = H(|\mu|) + e^{-2a|\mu|} H(-|\mu|) \, ,
```
where,
```math
\begin{aligned}
H(\mu) &= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{a\mathcal{G}'(\mu')} \exp\left(-\frac{(a \Delta q + \mu - \mu')^2}{2 \Delta q}\right) \\
&= \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{a \Delta q + \mu}{\sqrt{2 \Delta q}}\right)\right] \\
&+ \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{a\mathcal{G}'(\mu')} - 1\right) \exp\left(-\frac{(a \Delta q + \mu - \mu')^2}{2 \Delta q}\right) \, .
\end{aligned}
```
For quadratures this wouldn't necessarily be the preferred approach, since it requires computing two separate integrals; especially for adaptive quadrature, we'd want to combine the two terms into a single integrand and only perform one integral per ``|\mu|``. However, with FFT the second half comes along for the ride simply because discrete convolution is cyclic; we should obviously take advantage of it.
"""

# ╔═╡ fa5d715b-7ec1-492d-9cba-8b164854868f
md"""
Now let's sample uniformly and try the FFT-based approach to computing this integral.
"""

# ╔═╡ f6de0959-95b7-42e5-b79a-5b09e51abdb0
let μconvmax=0.43, n=2^23
	global const μconv = range(0.0, μconvmax, n + 1)[begin:(end - 1)]

	rate = n / μconvmax
	f = rfftfreq(2n, rate)
	@show n, rate

	global subsample(fμ; keep=128) = fμ[begin:(length(fμ) ÷ keep):end]
	
	gaussianf(mean, var) = exp.(-2π .* im .* mean .* f .- (2π .* f).^2 .* var ./ 2)

	global function Gmap_conv(Gμ, a, dq)  # Not optimized for allocations!
		@assert length(Gμ) == n

		mean, erfd, decay = -a * dq, sqrt(2dq), exp.((-2a) .* μconv)

		# We're implicitly representing the symmetric function
		# μ -> exp(a * G(abs(μ))) - 1 as
		# ifftshift([er[begin:(end - 1)]; er[end] + eμ[begin]; eμ[(begin + 1):end]])
		# where eμ = exp.(a .* Gμ) and er = reverse(eμ). However, we have
		# analytically rewritten our calculations in terms of convolutions involving
		# only the upper half eμ, so we must extend the arrays with zeros
		# representing the excised lower half (er), and cut eμ[begin] in half as the
		# other half belongs to the excised part
		eGm1μ = [exp.(a .* Gμ) .- 1; zero(Gμ)]
		eGm1μ[begin] /= 2
		eGm1f = rfft(eGm1μ) / rate

		Hconvf = eGm1f .* gaussianf(mean, dq)
		Hconvμ = irfft(Hconvf, 2n) * rate

		eGnextμ = Hconvμ[begin:(begin + n - 1)]
		eGnextμ[(begin + 1):end] .+= (
			decay[(begin + 1):end] .* Hconvμ[end:-1:(begin + n + 1)]
		)
		eGnextμ[begin] *= 2  # Restore first element (see comment above)
		eGnextμ .+= (
			1 .+ erf.((μconv .- mean) ./ erfd)
			.+ decay .* erfc.((μconv .+ mean) ./ erfd)
		) ./ 2
		
		return log.(eGnextμ) ./ a
	end
end;

# ╔═╡ 0ed97298-396b-43a7-bb5e-38aee668d50d
let scatteroptions=(; markersize=5)
	μs = subsample(μconv)
	G1μ = subsample(Gmap_conv(G0.(μconv), as_[1], dqs_[1]))
	
	μI = first(μs)..last(μs)

	fig, ax = Gplot(μI, G0, G1_quadgk_lazy; legend=false)
	scatter!(
		ax, μs, G1μ;
		label=L"\mathcal{G}_1^{\text{DFT}}(\mu)",
		color=Cycled(2), scatteroptions...,
	)
	axislegend(ax)
	fig
end

# ╔═╡ bde42de4-9245-4594-844e-16a3a54d3e3f
md"""
That looks decent to the naked eye, but how accurate is it actually? Let's investigate. (Note that I'm subsampling for the plots both above and below; the functions were sampled much more densely in the calculations.)
"""

# ╔═╡ 90f26134-21ee-4328-bdb4-b7e6e479cbcd
begin
	function diffplot(
		fx::AbstractVector, fxapprox::AbstractVector, xs;
		flabel="f", approxlabel="approx", xlabel="x",
		relativeto=identity, relativetolabel="x", legendposition=:lt,
	)
		diff = abs.((fxapprox .- fx) ./ relativeto(fx))
		
		fl = L"%$flabel(%$xlabel)"
		fal = L"%$flabel^{\text{%$approxlabel}}(%$xlabel)"
		frl = L"%$flabel(%$relativetolabel)"
		
		fig = Figure()
		ax = Axis(fig[1, 1]; xlabel=L"%$xlabel", yscale=log10)
		scatter!(
			ax, xs, diff;
			markersize=5, color=Cycled(2),
			label=L"\left|\frac{%$fal - %$fl|}{%$frl}\right|",
		)
		hlines!(
			ax, eps(eltype(diff)) .^ ([7, 8, 9, 10, 11, 12, 13] ./ 13);
			color=:black, linestyle=:dash,
			label=L"\epsilon^{b / 13} : b \in \{7, 8, 9, 10, 11, 12, 13\}",
		)
		ylims!(ax, 1e-16, 1e-8)
		axislegend(ax; position=legendposition)
		fig
	end

	function diffplot(f, f_approx, xs; kwargs...)
		return diffplot(f.(xs), f_approx.(xs), xs; kwargs...)
	end
end;

# ╔═╡ 2986bf5b-fb0f-4791-85f6-2517e9ab2d81
let
	μs = subsample(μconv)
	G1μconv = subsample(Gmap_conv(G0.(μconv), as_[1], dqs_[1]))
	ncalls, G1μ = countintegrandcalls() do
		# Using lazyG1, not interpolated, to count integrand calls
		G1_quadgk_lazy.(μs)
	end
	@show ncalls / length(μs)
	diffplot(
		G1μ, G1μconv, μs;
		flabel=L"\mathcal{G}_1", approxlabel="DFT", xlabel=L"\mu",
		relativeto=first, relativetolabel="0",
	)
end

# ╔═╡ f769c886-ad0d-4d7a-9977-fed541c476cc
md"""
Clearly there are systematic errors from the discretization (near the origin) and from the finite length of the sampled interval (near the upper end, which is the midpoint of the interval used for convolution). It takes a very fine sampling rate to keep these errors down; for these parameters, to comfortably get below ``\epsilon^{11/13}`` error (11 of 13 hexadecimals converged) relative to the maximum value ``\mathcal{G}_1(0)``, we need a sampling rate of ``2 \cdot 10^7`` and a wide enough interval that we end up with well over 8 million samples. Note that these numbers depend on the width of the support of ``\mathcal{G}_{i - 1}``: here we used the shortest possible interval to compute ``\mathcal{G}_1`` to the desired precision, but we know we'll need a much wider interval to make it to ``\mathcal{G}_{37}``, and we'll have to maintain a similar sampling rate, coming out to several times more samples.

In comparison, QuadGK uses less than 250 samples per integral for the same relative error, hence less than 60000 samples for one full convolution (assuming the next layer will require similar resolution to obtain the desired precision). ``N_1^2`` is in fact less than ``N_2 \log N_2`` when ``N_1 \sim 250`` and ``N_2 \sim 8 \cdot 10^6``.

Conlusion: accurate convolution requires too many samples.
"""

# ╔═╡ 4bb351f9-5800-4fdd-a7fe-2a6009315073
md"""
#### Gauss quadrature

Moving on to quadrature-based approaches, we've already seen that QuadGK works well, but it requires an alternating sequence of adaptive quadrature and interpolation (also adaptive if using ApproxFun), which might not be the most straightforward code to differentiate through and optimize over. (To be fair, I haven't tried, one could probably do it with some custom adjoints.) However, with fixed quadrature nodes and weights the nested integrals start to look a lot like feedforward neural networks, and if there's anything we know how to optimize these days...

To this end, let's study some quadrature schemes and compare their relative errors, using the QuadGK implementation as reference.
"""

# ╔═╡ a17151fe-6078-4569-8129-e5242409caf7
md"""
##### Gauss-Laguerre quadrature

Gauss-Laguerre quadrature computes
```math
\int_0^\infty f(x) e^{-x} dx \, .
```
The integral we want to solve is
```math
\int_0^\infty \tilde{f}(\mu) d\mu \, ,
```
where ``\tilde{f}`` represents our entire integrand. To this end, we let ``\mu = \lambda x`` and define ``f(x) = \lambda \tilde{f}(\lambda x) e^x``. However, that exponential is dangerous. In principle its growth is more than canceled by the decay of the Gaussian, but numerically we'll be multiplying a very small and a very large number and lose a lot of accuracy---in fact, `exp(x)` evaluates to infinity at fairly moderate values (``x \gtrsim 710`` for `Float64`). But we can do better by absorbing the exponential in the Gaussian exponents. Here's the analytical treatment:
```math
\begin{aligned}
&e^{a \mathcal{G}(\mu)} \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{-\mu' / \lambda} e^{a\mathcal{G}'(\mu')} \\
&\times \left[\exp\left(-\frac{(\mu' - |\mu| - a \Delta q)^2}{2 \Delta q} + \frac{\mu'}{\lambda}\right) + \exp\left(-\frac{(\mu' + |\mu| - a \Delta q)^2}{2 \Delta q} + \frac{\mu'}{\lambda} - 2a|\mu|\right)\right] \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{-\mu' / \lambda} e^{a\mathcal{G}'(\mu') + (2a + \lambda^{-1}) \Delta q / 2\lambda} \\
&\times \bigg[\exp\left(-\frac{(\mu' - |\mu| - (a + \lambda^{-1}) \Delta q)^2}{2 \Delta q} + \frac{|\mu|}{\lambda} \right) \\
&+ \exp\left(-\frac{(\mu' + |\mu| - (a + \lambda^{-1}) \Delta q)^2}{2 \Delta q} - (2a + \lambda^{-1}) |\mu|\right)\bigg] \\
\end{aligned}
```
It might not be necessary to implement new functions with these completed squares. Adding the extra term to the Gaussian exponents we already have as per the first line above ought to be accurate enough. In fact, it's probably more accurate since the expressions are simpler, i.e., fewer operations.
"""

# ╔═╡ ab0c84db-168e-410b-91e4-2396d4fd1c56
let n=128, (lx, lw)=gausslaguerre(n)
	global const lscale = μmax / (1.5lx[end] - 0.5lx[end - 1])
	global const lnodes = lscale * lx
	global const lweights = lscale * lw
end;

# ╔═╡ 249a427b-c112-4d17-8889-db94c72a7225
begin
	@stable function Gmap_laguerre(G, μ, a, dq)
		lastfval = Gintegrand(G, lnodes[end], μ, a, dq)
		I = if (sqrt(2π * dq) * lastfval) > eps(typeof(lastfval))
			# We're leaking integrand. At this point we know the integral is 1.
			oneunit(eltype(lweights)) * oneunit(lastfval)
		else
			fvals = Gintegrand.(G, lnodes, μ, a, dq, lnodes ./ lscale)
			dot(lweights, fvals)
		end
		return log(I) / a
	end
	
	@stable G1_laguerre(μ) = Gmap_laguerre(G0, μ, as_[1], dqs_[1])
end;

# ╔═╡ 8e10cd1a-4e0f-4be2-8a43-d890df758b8e
Gplot(Gdomain, G0, G1_laguerre)

# ╔═╡ 53d18dc9-bb10-4fdc-a0c9-c46b792764a7
diffplot(
	G1_quadgk, G1_laguerre, lnodes[lnodes .< μmax];
	flabel=L"\mathcal{G}_1", approxlabel="Laguerre", xlabel=L"\mu",
	relativeto=first, relativetolabel="0"
)

# ╔═╡ cdb60bf4-2e1a-4bd8-b4bc-95ec4b85d83a
md"""
Alright, 128 nodes gives pretty good accuracy up to ranges where the true value is as good as zero. At that point convergence suffers until we reach the point where we clamp to zero where the integrand starts to measurably leak outside the nodes (which we can do because we know that the asymptotic value of the integral is ``1``). For this integral we could simply clamp earlier, but if we want to keep the same nodes allt the way up to ``i = 37``, that might be discarding nontrivial non-noise values, so the proper solution is to increase the number of nodes.

From experience, Gauss-Laguerre will always go unstable and start oscillating for large ``\mu``, i.e., when the Gaussian is displaced very far from the origin, and this is most likely the source of the divergence shown above. This is not surprising: the nodes are getting too sparse, so we're undersampling and allowing lots of oscillations in the effective polynomial interpolation (all quadrature rules work via an implicit polynomial interpolation). While the full integrand asymptotes to a shifted but normalized Gaussian with measure ``1``, the Gauss-Laguerre weights within the bulk of the Gaussian decay exponentially as ``\mu`` increases, and the modified integrand with the exponential factor, which is what the polynomial interpolation works on, blows up to compensate (within the bulk of the Gaussian, the quadratic term is small and dominated by the linear growth term). When the bulk is sparsely sampled, it's no surprise that this results in an oscillating interpolant; this tends to be the way of high-order polynomials. In addition, numerical accuracy may suffer from the multiplication of very large and very small numbers.

If we don'w want to clamp, another fix could be to rewrite the integral such that the mass is always located near the origin. One approach would be to split at the top of the Gaussian and use Gauss-Legendre quadrature to the left of this and shifted Gauss-Laguerre to the right. However, this would require knowing ``\mathcal{G}'(\mu')`` at a set of nodes that depends on the ``\mu`` at which we're evaluating ``\mathcal{G}(\mu)``. This is what we're trying to avoid with fixed quadrature rules---we want a neural network-like forward pass. Let's pass on that idea for now.
"""

# ╔═╡ 4d796285-45ef-491e-b9a3-00a27d1fad7a
md"""
Let's look at the integrand and Gauss-Laguerre nodes in the region where accuracy starts suffering. We'll plot the integrand on a logarihtmic axis to more clearly see it's effective support, i.e., the region where it's value is at least `eps(Float64)` times its maximum.
"""

# ╔═╡ 99c700e5-7233-4fb6-b415-6aaa79da0faf
function quadplot(G, μ, a, dq, nodes; integrand=Gintegrand, legendkws...)
	μI = first(nodes)..last(nodes)
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"\mu'", yscale=log10)
	p = lines!(ax, μI, μp -> integrand(G, μp, μ, a, dq); label="Integrand")
	vlines!(ax, nodes; color=(:gray, 0.25), label="Quadrature nodes")
	hlines!(
		ax, [1.0, eps()] ./ sqrt(2π * dq);
		color=:black, linestyle=:dash,
		label=L"b / \sqrt{2\pi\Delta q} : b \in \{1, \epsilon\}",
	)
	ylims!(ax, 1e-16, 1e1)
	axislegend(ax; legendkws...)
	return Makie.FigureAxisPlot(fig, ax, p)
end;

# ╔═╡ 08c3e556-63d3-4f22-b69e-5be3f7a99e98
quadplot(G0, 5.0, as_[1], dqs_[1], lnodes[lnodes .< μmax]; position=:lt)

# ╔═╡ 6a8da0f5-e43f-435f-a326-2eee8560859e
md"""
The clearly shows the increasing undersampling as we increase ``\mu`` (i.e., shift the integrand right).
"""

# ╔═╡ 9285938d-dd43-4132-bdb0-2e8aba84140c
md"""
##### Gauss-Legendre

We already decided on a `μmax` beyond which all the ``\mathcal{G}_i(\mu)`` are approximated by hard zeros. Instead of using tricks to compute integrands over ``[0, \infty)``, we can do a straightforward integral over ``[0, \mu_{\text{max}}]``. This is especially appealing because we have already seen that the bulk of the integrand mass may be located anywhere within this interval depending on the ``\mu`` at which we're evaluating the integral, so ``[0, \infty)``-schemes with decaying node density aren't really appropriate for this problem---their accuracy depends on how close to the origin the mass is located.

The go-to fixed quadrature rule for ``\int_a^b f(x) dx`` is Gauss-Legendre.
"""

# ╔═╡ 3820fe1e-eb76-4f33-9e1b-9e162b76d4c6
let n=128, (ux, uw)=gausslegendre(n)  # [-1, 1]
	tx = (ux .+ 1) ./ 2  # Shifted and scaled to [0, 1]
	tw = uw ./ 2
	global const tnodes = μmax * tx
	global const tweights = μmax * tw
end;

# ╔═╡ 8e6e8a69-1677-4e6f-a0bc-0d5304b83263
begin
	@stable @views function Gmap_legendre(G, μ, a, dq)
		lastfval = Gintegrand(G, tnodes[end], μ, a, dq)
		I = if (sqrt(2π * dq) * lastfval) > eps(typeof(lastfval))
			# We're leaking integrand. At this point we know the integral is 1.
			oneunit(eltype(tweights)) * oneunit(lastfval)
		else
			fvals = Gintegrand.(G, tnodes[begin:(end - 1)], μ, a, dq)
			push!(fvals, lastfval)
			dot(tweights, fvals)
		end
		return log(I) / a
	end
	
	@stable G1_legendre(μ) = Gmap_legendre(G0, μ, as_[1], dqs_[1])
end;

# ╔═╡ 31310c3f-6269-49df-8aef-9a60ace637dc
Gplot(Gdomain, G0, G1_legendre)

# ╔═╡ 4a659c8b-4b36-4a39-a7c8-bc49a0770e51
diffplot(
	G1_quadgk, G1_legendre, tnodes;
	flabel=L"\mathcal{G}_1", approxlabel="Legendre", xlabel=L"\mu",
	relativeto=first, relativetolabel="0",
)

# ╔═╡ b5819306-7e49-4139-a22b-e426980d0ec4
quadplot(G0, 5.0, as_[1], dqs_[1], tnodes; position=:lt)

# ╔═╡ 24bd7873-96b1-4462-87af-0f132752195b
md"""
This is pretty good. It starts out as good as Gauss-Laguerre and remains bounded across the entire range. We can see that uniform weighting of the domain works better for this problem. Still, quadrature order and domain cutoff need attention.
"""

# ╔═╡ fb8f67b2-3abb-44bf-bca7-b84d2f1e8b9f
md"""
#### Stretched Gauss-Legendre

I just wanted to check whether other ways of integrating over the infinite half-line may work better due to different node distribution. To this end, let's take Gauss-Legendre and apply the standard substitution used by, e.g., QuadGK to map ``[, \infty)`` onto ``[0, 1)``. This substitution is algebraic rather than exponential like Gauss-Laguerre, so the density of nodes will decay differently.

Without thinking too much I thought it might perhaps decay slower, and asymptotically it does, but initially it falls off faster---that is, fatter tail but narrower peak. Due to the heavy tail we cannot scale these nodes to such that the largest one is near our selected `μmax`; if we did, the gaps between nodes would be immense except in a small region of small ``\mu`` (since the tail of very low but non-zero node density is so long). Instead, we scale the nodes such that the starting density near ``\mu' = 0`` is the same in this case as for Gauss-Laguerre.
"""

# ╔═╡ 6e0e83d1-9cf8-4e4b-a2f9-48670f9ac54c
let
	sx = tnodes ./ (μmax .- tnodes)  # Stretched to [0, ∞)
	sw = μmax .* tweights ./ ((μmax .- tnodes).^2)
	sscale = (lnodes[begin + 1] - lnodes[begin]) / (sx[begin + 1] - sx[begin])
	global const snodes = sscale * sx
	global const sweights = sscale * sw
	@show snodes[end]
	@show length(snodes[snodes .> μmax])
end;

# ╔═╡ e49c231b-2bfd-4986-8c80-2715e4ebb4d3
begin
	@stable @views function Gmap_legendre_stretched(G, μ, a, dq)
		lastfval = Gintegrand(G, snodes[end], μ, a, dq)
		I = if sqrt(2π * dq) * lastfval > eps(typeof(lastfval))
			# We're leaking integrand. At this point we know the integral is 1.
			oneunit(eltype(sweights)) * oneunit(lastfval)
		else
			fvals = Gintegrand.(G, snodes[begin:(end - 1)], μ, a, dq)
			push!(fvals, lastfval)
			dot(sweights, fvals)
		end
		return log(I) / a
	end
	
	@stable G1_legendre_stretched(μ) = Gmap_legendre_stretched(G0, μ, as_[1], dqs_[1])
end;

# ╔═╡ cdb6f1e9-c420-42dc-b06f-5e235ee07e8f
Gplot(Gdomain, G0, G1_legendre_stretched)

# ╔═╡ d8aa9d5b-4363-4c55-80f1-b7d8eb15d347
diffplot(
	G1_quadgk, G1_legendre_stretched, snodes[snodes .< μmax];
	flabel=L"\mathcal{G}_1", approxlabel="stretched", xlabel=L"\mu",
	relativeto=first, relativetolabel="0", legendposition=:rt,
)

# ╔═╡ d2b13b6e-2b5c-465c-a647-344f3495867b
quadplot(G0, 5.0, as_[1], dqs_[1], snodes[snodes .< μmax]; position=:lt)

# ╔═╡ 9f18c4cd-152c-4ec4-84b3-5e868fc0b5a9
md"""
This is a lot worse than Gauss-Laguerre. Convergence drops off sooner and large oscillations appear way below `μmax`. Node density is way too low in the right 3/4 of the effective domain, while 38 of 128 nodes are wasting away in the barren gloom of ``\mu > \mu_{\text{max}}``; still, if we wanted to increase density across the interval we'd have to increase the scaling factor and push more nodes out to waste. This is abad idea. Moving on.
"""

# ╔═╡ 98660249-52ff-4880-a4fb-71757844cb9b
md"""
##### Gauss-Hermite

Finally, let's address the elephant in the room. We're integrating functions against a Gaussian kernel---clearly, we should do a substitution to bring the Gaussian to the center and use Gauss-Hermite quadrature. That way we'll avoid all issues of shifting integrand mass, and simply use the optimal nodes and weights for integrating a smooth function over a Gaussian measure.

The problem is twofold: first, there's the qualifier "smooth". When we unfold the integrand over ``(-\infty,\infty)`` and center the Gaussian factor, we end up with a kink at ``\mu' = \mu``. In the ``[0, \infty)``-formulation we placed that kink at ``\mu' = 0`` and folded the negative half over. Fixed quadrature rules like are not suited for non-analytic integrands within the domain, they should be placed at endpoints, and that's not possible here if we want the Gauss-Hermite weight to match the actual Gaussian in our integrand.

The second issue is that we would lose the neural network-like property of only needing to know ``\mathcal{G}_i(\mu)`` at the quadrature nodes. The integrand would now contain a factor ``\exp(\mathcal{G}_{i - 1}(\mu - \mu'))``, i.e., so we would need to evaluate ``\mathcal{G}_i`` at the difference between each quadrature node and each new evaluation point. The nodes would not be equally spaced, so these differences wouldn't map onto the nodes themselves.

But that's a lot of talk. Let's see the receipts.
"""

# ╔═╡ e7a500fa-24ea-47bc-aa1e-3a60fabe6435
let n=128
	global const hnodes, hweights = gausshermite(n; normalize=true)
end;

# ╔═╡ 01780f9b-95a9-4601-82ca-10af89a10375
begin
	@stable function Ghintegrand(G, μp, μ, a, dq)
		dμ = μ - μp
		return exp(a * (G(dμ) + abs(dμ) - abs(μ) - a * dq / 2))
	end
	
	@stable function Ghfullintegrand(G, μp, μ, a, dq)  # Just for plots
		return Ghintegrand(G, μp, μ, a, dq) * exp(-μp^2 / (2dq)) / sqrt(2π * dq)
	end
	
	@stable function Gmap_hermite(G, μ, a, dq)
		fvals = Ghintegrand.(G, sqrt(dq) .* hnodes, μ, a, dq)
		return log(dot(hweights, fvals)) / a
	end
	
	@stable G1_hermite(μ) = Gmap_hermite(G0, μ, as_[1], dqs_[1])
end;

# ╔═╡ 8177c265-91fb-4cd7-8652-f4cfb3c8aa2f
Gplot(Gdomain, G0, G1_hermite)

# ╔═╡ b12cd52d-1f43-4def-9a92-b425526e4a28
diffplot(
	G1_quadgk, G1_hermite, sqrt(dqs_[1]) * hnodes;
	flabel=L"\mathcal{G}_1", approxlabel="Hermite", xlabel=L"\mu",
	relativeto=(_ -> G1_quadgk(0.0)), relativetolabel="0",
)

# ╔═╡ 12f69cae-7017-4fde-89ea-54f701b4bd68
quadplot(G0, 0.03, as_[1], dqs_[1], sqrt(dqs_[1]) * hnodes; integrand=Ghfullintegrand)

# ╔═╡ 16e88b60-d354-4ba9-83a9-cbc53c4d17b6
md"""
_Update 2: the observation noted below does not hold at or near zero temperature. Gauss-Hermite still struggles badly there._

_Update: for parameters closer to the free energy optimum, the first ``dq`` is so small that the kink ends up being insignificant and Gauss-Hermite gets excellent accuracy. Since each integrals smoothens the kernel, the problem probably never surfaces if it wasn't present in the first integral, even though ``dq`` increases through the recursion. Now Gauss-Hermite looks awfully good all of a sudden. Shame about that interpolation._

As expected: accuracy suffers due to the kink, but is great in regions where the kink is outside the bulk of the Gaussian.

It's worth noting that each recursion step makes the kink smoother. In fact, it's only truly non-analytic when ``\beta = \infty``, and then only in the first integral. Starting from a higher temperature, or later in the recursion, Gauss-Hermite quadrature would have excellent accuracy.

However, the problem we ultimately want to solve has ``\beta = \infty``, so accuracy in that regime is important. A scheme that puts the kink at the boundary of the region of integration seems like the way to go, especially when that also gives us the feedforward neural network property.

There's an interesting duality where Gauss-Hermite quadrature is least accurate in the first step at ``\beta = \infty``, which is where discrete convolution would be most accurate. But neither is flexible enough to be our choice in the end. The winner is Gauss-Legendre quadrature.
"""

# ╔═╡ 9e33f253-15d9-4405-a7eb-a6c386d246b4
md"""
#### The full forward pass

We've observed decent accuracy from a single Gauss-Legendre pass with 128 nodes. However, we're doing ``k`` steps for some large ``k``, such as ``k = 37`` if we follow Atsushi. Let's find out what it takes to make that entire sequence accurate.
"""

# ╔═╡ 4d3c6872-b462-4efe-abf4-28d4c6cdb332
let	n=512, (ux, uw)=gausslegendre(n)
	tx = (1 .+ ux) ./ 2
	tw = uw / 2
	global const nodes = μmax * tx
	global const weights = μmax * tw
end;

# ╔═╡ c49877b1-e92b-4514-9439-e3c10c6b4e6b
begin
	@stable @views function Gmap_local(Gv, μ, a, dq)
		μabs = abs(μ)
		lastval = exp(a * Gv[end]) * Gintegrand(zero, nodes[end], μabs, a, dq)
		I = if (sqrt(2π * dq) * lastval) > eps(typeof(lastval))
			oneunit(eltype(weights)) * oneunit(lastval)
		else
			vals = Gintegrand.(
				zero, nodes[begin:(end - 1)], μabs, a, dq, a .* Gv[begin:(end - 1)]
			)
			push!(vals, lastval)
			dot(weights, vals)
		end
		return log(I) / a
	end
	
	@stable Gmap(Gv, a, dq) = map(μp -> Gmap_local(Gv, μp, a, dq), nodes)
end;

# ╔═╡ 36c3f5a2-7fcc-4b5a-9109-25e5a994d1c3
Gvs = let
	Gvs = [G0.(nodes)]
	for i in 1:k
		push!(Gvs, Gmap(last(Gvs), as_[i], dqs_[i]))
	end
	Gvs
end;

# ╔═╡ 06ff0708-675d-4008-9812-e35d785fe054
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for Gv in Gvs
		lines!(ax, nodes, Gv)
	end
	fig
end

# ╔═╡ 223aa38b-ee9e-4da8-9d19-9c440ee15ca5
let i=k, a=as_[i]
	diffplot(
		# Gs_quadgk[i + 1].(nodes), Gvs[i + 1], nodes;
		exp.(a .* Gs_quadgk[i + 1].(nodes)), exp.(a .* Gvs[i + 1]), nodes;
		flabel=L"\mathcal{G}_{%$i}", xlabel=L"\mu", approxlabel="Legendre",
		relativeto=first, relativetolabel=L"\mu",
	)
end

# ╔═╡ ad934872-4b43-47a1-b98b-cc848ff2c3b3
md"""
There we go, 256 points and the precision is excellently preserved. This is actually overkill, even 128 performed great, but with 256 the relative error in ``e^{a \mathcal{G}_i(\mu)}`` remained strictly below ``16^{-12}`` for every ``i`` and every ``\mu``. (This is error with respect to the QuadGK solution, but since they are two independent and unrelated ways of computing the same quantity we can assume that this is a decent order-of-magnitude estimate of the true error as well; I don't know of a reason to believe that the errors of the two methods are correlated.)
"""

# ╔═╡ 72459310-d9b1-49a6-b19c-fba1df996773
md"""
#### Setting up the optimization problem

It's time to optimize.

This notebook is getting long and scrolling is tedious. Let's summarize the problem we're solving:
```math
\begin{aligned}
f[\mathbf{a},\mathbf{q},M] &= \frac{J_0}{2} M^2 - \frac{\beta}{4} (1 - q_1)^2 + \frac{1}{4} \sum_{i = 1}^k a_i \left((q_i - 1)^2 - (q_{i + 1} - 1)^2\right) \\
&- \mathcal{I}\left[\mathcal{G}_k(\mu) + |\mu|; h + J_0 M, q_{k + 1}\right] \, ,
\end{aligned}
```
where,
```math
\mathcal{I}\left[g(\mu'); \mu, \Delta q\right] = \int_{-\infty}^\infty \frac{d\mu'}{\sqrt{2 \pi \Delta q}} e^{-(\mu - \mu')^2 / 2 \Delta q} g(\mu') \, ,
```
and the function ``\mathcal{G}_k(\mu)`` is defined through the recursion
```math
\mathcal{G}_i(\mu) = \frac{1}{a_i} \log \mathcal{I}\left[\exp \left[a_i\left(\mathcal{G}_{i - 1}(\mu') + |\mu'| - |\mu| - a_i \Delta q_i / 2\right)\right]; \mu, \Delta q_i\right] \, ,
```
with ``\mathcal{G}_0(\mu) = \log(1 + e^{-2\beta|\mu|}) / \beta``. Since this is symmetric and ``\mathcal{I}`` preserves symmetry, all the ``\mathcal{G}_i(\mu)`` are symmetric, so we only need to evaluate them at ``\mu \ge 0``.

The parameters in these epxressions satisfy
```math
\begin{aligned}
0 &< a_k < \ldots < a_1 < \beta \, , \\
0 &< q_{k + 1} < \ldots < q_1 \le 1 \, , \\
0 &< \Delta q_i = q_i - q_{i + 1} \, , \quad i = 1, \ldots, k \, ,
\end{aligned}
```
and ``q_1 \to 1`` as ``\beta \to \infty``, such that the term in ``f`` that's ``\propto \beta`` vanishes in the limit.


The objective is to minimize ``f`` with respect to ``M``, maximize with respect to the ``q_i``, and extremize with respect to the ``a_i`` in whatever direction gives us a stationary point satisfying the constraints.
"""

# ╔═╡ 6228dbd3-71fd-4d84-8cab-847a255697a5
md"""
We rewrite the recursive integrals as follows for better numerical behavior:
```math
\begin{aligned}
&\mathcal{I}\left[\exp \left[a\left(\mathcal{G}'(\mu') + |\mu'| - |\mu| - a \Delta q / 2\right)\right]; \mu, \Delta q\right] \\
&= \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \exp\left(-\frac{(\mu' - |\mu|)^2}{2 \Delta q} + a\left(\mathcal{G}'(\mu') + |\mu'| - |\mu| - \frac{1}{2} a \Delta q\right)\right) \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \\
&\times \left[\exp\left(a \mathcal{G}'(\mu') - \frac{(\mu' - |\mu| - a \Delta q)^2}{2 \Delta q}\right) + \exp\left(a(\mathcal{G}'(\mu') - 2|\mu|) - \frac{(\mu' + |\mu| - a \Delta q)^2}{2 \Delta q} \right)\right] \, .
\end{aligned}
```
(The absolute value sign on ``|\mu|`` in the Gaussian part in the first line was not present in the definition of ``\mathcal{I}``, but we can add it here because the integrand is even, and it's key to begin able to complete the square in the exponent.)
"""

# ╔═╡ e8c6826b-6d20-4bf2-a63b-12d59f2c1c4e
md"""
Here's how we specify optimization variables in a way that makes the monotonicity constraints enforcible. For the ``a_i`` we define
```math
\begin{aligned}
\xi_i &= \frac{a_i}{1 + a_i} \in [0, 1] \, , \quad i = 0, \ldots, k \, , \\
v_i &= -\log\left(\frac{\xi_{i - 1}}{\xi_i} - 1\right) \, , \quad i = 1, \ldots, k \, ,
\end{aligned}
```
where we take ``a_0 = \beta`` and hence ``\xi_0 = \beta / (1 + \beta)``. The map from ``a_i`` to ``\xi_i`` helps specify constraints when ``\beta = \infty``, and might also help alleviate a vanishing gradient problem. The optimization variables are the ``v_i``, while ``\beta`` is a parameter of the problem. Notice that the map from ``v_i`` back to the ratios of ``\xi_i`` is a sigmoid:
```math
\frac{\xi_i}{\xi_{i - 1}} = \frac{1}{1 + e^{-v_i}} \in (0, 1) \, .
```
This ensures that the ``\xi_i`` are monotonically ordered in ``(0, \xi_0)``, and hence that the ``a_i`` are monotonically ordered in ``(0, \beta)``, without having to add explicit constraints to the optimization problem.

For the ``q_i`` we use the same trick trick, recalling that ``q_0 = 1``, and for ``\beta = \infty`` there's also ``q_1 = 1`` (we'll probably have to add a special case for this). Thus,
```math
\begin{aligned}
w_i &= -\log\left(\frac{q_{i - 1}}{q_i} - 1\right) \, , \quad i = 1, \ldots, k + 1 \, .
\end{aligned}
```
with the special provision that ``w_0 = 1`` if and only if ``\beta = \infty``.
"""

# ╔═╡ e95c9adf-93f7-4471-a1ca-f24c106012ad
begin
	# @stable a2ξ(a) = 1 - exp(-a)
	# @stable ξ2a(ξ) = -log(1 - ξ)
	@stable a2ξ(a) = isinf(a) ? one(a) : a / (1 + a)
	@stable ξ2a(ξ) = isone(ξ) ? oftype(one(ξ), Inf) : ξ / (1 - ξ)
	@stable sigmoid(x) = inv(1 + exp(-x))
	@stable invsigmoid(x) = -log(inv(x) - 1)
end;

# ╔═╡ c2e34d7e-8989-48e3-945c-8f42fed07ead
begin
	struct Quadrature
		nodes::Vector{Float64}
		weights::Vector{Float64}
	end

	function Quadrature(f::typeof(gausslegendre), n, a, b)
		nodes, weights = f(n)
		scale = abs(b - a) / 2
		weights .*= scale
		nodes .*= scale
		nodes .+= (a + b) / 2
		return Quadrature(nodes, weights)
	end

	@stable Base.broadcastable(quadparams::Quadrature) = (quadparams,)

	@stable function mappedquad(F, values, quadparams::Quadrature)
		# integrate F.(nodes, values)
		(; nodes, weights) = quadparams
		wF(w, n, v) = w * F(n, v)
		return mapreduce(wF, +, weights, nodes, values)
	end
end;

# ╔═╡ 76fe38c8-5a24-40d5-abd2-bddc4fe981da
@stable function G02(μ, β)
	isinf(β) && return zero(μ)
	return log1p(exp(-2β * abs(μ))) / β
end;

# ╔═╡ 29ae03f4-8653-4fd8-94be-710b124e66b4
@stable function Gweight(μp, μabs, ai, dqi, expshift=nothing)
	dμp = μp - ai * dqi
	if isnothing(expshift)
		return wgaussian(μabs - dμp, dqi) + wgaussian(μabs + dμp, dqi, -2ai * μabs)
	end
	es2 = expshift - 2ai * μabs
	return wgaussian(μabs - dμp, dqi, expshift) + wgaussian(μabs + dμp, dqi, es2)
end;

# ╔═╡ 304d7c54-2e35-411d-adda-3f3e6f0d00a2
begin
	@stable function Gmap2_local(Gv, μ, ai, dqi, quadparams)
		μabs = abs(μ)
		integrand = let μabs=μabs, ai=ai, dqi=dqi
			integrand(μp, Gμp) = Gweight(μp, μabs, ai, dqi, ai * Gμp)
		end
		(; nodes, weights) = quadparams
		lastval = Gweight(nodes[end], μabs, ai, dqi, ai * Gv[end])
		I = if (sqrt(2(π * dqi)) * lastval) > eps(typeof(lastval))
			oneunit(eltype(weights)) * oneunit(lastval)
		else
			mappedquad(integrand, Gv, quadparams)
		end
		return log(I) / ai
	end

	@stable function Gmap2(Gv, a, dq, quadparams)
		return Gmap2_local.((Gv,), quadparams.nodes, a, dq, quadparams)
	end
	
	@stable function Gmap2!(Gvnext, Gv, a, dq, quadparams)
		Gvnext .= Gmap2_local.((Gv,), quadparams.nodes, a, dq, quadparams)
		return nothing
	end
end;

# ╔═╡ 439a8caa-fee3-44f9-9a00-7520d0cf85cd
@stable function RSBrecursion(G0v, a, dq, quadparams)
	Gvprev = Gmap2(G0v, a[begin], dq[begin], quadparams)
	Gv = Gmap2(Gvprev, a[begin + 1], dq[begin + 1], quadparams)
	for i in eachindex(a, dq)[(begin + 2):end]
		Gv, Gvprev = Gvprev, Gv
		Gmap2!(Gv, Gvprev, a[i], dq[i], quadparams)
	end
	return Gv
	# # Non-mutating version for Zygote
	# Gv = Gmap2(G0v, a[begin], dq[begin], quadparams)
	# for i in eachindex(a, dq)[(begin + 1):end]
	# 	Gv = Gmap2(Gv, a[i], dq[i], quadparams)
	# end
	# return Gv
end;

# ╔═╡ 1c0094fd-b2a4-40a4-be8b-bdaebeeff044
@stable function traceterm(β, heff, a, dq, qlast, quadparams)
	G0v = G02.(quadparams.nodes, β)
	Gkv = RSBrecursion(G0v, a, dq, quadparams)
	integrand = let habs=abs(heff), za=zero(eltype(a)), qlast=qlast
		integrand(μp, Gμp) = (Gμp + abs(μp)) * Gweight(μp, habs, za, qlast)
	end
	return -mappedquad(integrand, Gkv, quadparams)
	# G = Gmap_quadgk(μp -> G02(μp, β), a[begin], dq[begin])
	# for i in eachindex(a, dq)[(begin + 1):end]
	# 	G = Gmap_quadgk(G, a[i], dq[i])
	# end
	# habs = abs(heff)
	# za = zero(eltype(a))
	# scale = Gmapscale(habs, za, qlast)
	# integrand = let G=G, habs=habs, za=za, qlast=qlast, scale=scale
	# 	function integrand(x) 
	# 		μp = x * scale
	# 		return (G(μp) + abs(μp)) * Gweight(μp, habs, za, qlast)
	# 	end
	# end
	# return -quad(integrand, 0.0, Inf)[1] * scale
end;

# ╔═╡ 0edc06a7-f4d3-4d14-8c74-b4bf17531202
@stable function cumprodf(
	fpre, fpost, ratios, start=one(typeof(fpre(oneunit(eltype(ratios)))))
)
	out = similar(
	# out = Zygote.Buffer(
		ratios, typeof(fpost(fpre(oneunit(eltype(ratios))) * oneunit(start)))
	)
	prev = start
	for i in eachindex(out, ratios)
		new = fpre(ratios[i]) * prev
		out[i] = fpost(new)
		prev = new
	end
	return out
	# return copy(out)  # Goes with Zygote.Buffer
end;

# ╔═╡ 95f6d0d0-8ecf-41c7-b176-55015ce10b5e
@stable @views function RSBparams(β, RSBweights)
	k = length(RSBweights) ÷ 2
	v = RSBweights[begin:(begin + k - 1)]
	w = RSBweights[(begin + k):end]
	a = cumprodf(sigmoid, ξ2a, v, a2ξ(β))
	q = if isinf(β)  # q1 = 1 fixed, w1 trivial and not included
		[one(sigmoid(oneunit(eltype(w)))); cumprodf(sigmoid, identity, w)]
	else
		cumprodf(sigmoid, identity, w)
	end
	return (; a, q)
end;

# ╔═╡ ec340551-8322-4af4-8117-280cbd5b67a2
@stable @views function freeenergySK(
	β, h, J0, M, RSBweights, quadparams
)
	(; a, q) = RSBparams(β, RSBweights)
	fieldterm = sum(
		a[i] * ((q[i] - 1)^2 - (q[i + 1] - 1)^2)
		for i in eachindex(a, q[begin:(end - 1)])
	) / 4
	fieldterm += J0 * (M^2) / 2
	if !isinf(β)
		fieldterm -= β * ((1 - q[1])^2) / 4
	end
	dq = q[begin:(end - 1)] .- q[(begin + 1):end]
	return fieldterm + traceterm(β, h + J0 * M, a, dq, q[end], quadparams)
end;

# ╔═╡ 874e4462-fdfb-4a8d-92aa-fcaf3780f28a
@stable function RSBobjectiveSK(RSBweights, (β, h, J0, M, quadparams))
	return freeenergySK(β, h, J0, M, RSBweights, quadparams)
end;

# ╔═╡ 2c091404-d4d3-4f76-845d-9ad04aa2fbae
@views function initRSBweightsSK(β, kRSB)
	ξ0 = a2ξ(β)
	nq = kRSB + (isinf(β) ? 2 : 3)  # q1 = 1 fixed? Then discard q0
	ξ = (range(ξ0, zero(ξ0), kRSB + 2)[begin:(end - 1)]).^inv(sqrt(2))
	q = sin.(range(π / 2, 0.0, nq)[begin:(end - 1)])
	v = invsigmoid.(ξ[(begin + 1):end] ./ ξ[begin:(end - 1)])
	w = invsigmoid.(q[(begin + 1):end] ./ q[begin:(end - 1)])
	return [v; w]
end;

# ╔═╡ 10839b94-c3ab-4dd6-8880-601a7a0d2839
function solveRSBweightsSK(β, h, J0, M, RSBweights0, quadparams; tol=16.0^(-11))
	f = OptimizationFunction{true}(RSBobjectiveSK, AutoEnzyme())
	p = (β, h, J0, M, quadparams)
	prob = OptimizationProblem(f, RSBweights0, p; sense=MaxSense)
	sol = solve(
		prob,
		## Optim.jl
		# BFGS(linesearch=Optim.LineSearches.BackTracking()); reltol=tol, g_tol=tol,
		# LBFGS(linesearch=Optim.LineSearches.BackTracking()); reltol=tol, g_tol=tol,
		# IPNewton(); reltol=tol, g_tol=tol,  # bounds only
		# NewtonTrustRegion(); reltol=tol, g_tol=tol,  # unbounded only, brittle
		## Ipopt.jl
		# Ipopt.Optimizer(); tol=tol, acceptable_tol=tol,  # brittle
		## NLopt.jl
		NLopt.LD_LBFGS(); abstol=tol, reltol=tol,  # Winner so far
		# NLopt.LD_VAR1(); abstol=tol, reltol=tol,
		# NLopt.LD_VAR2(); abstol=tol, reltol=tol,
		# NLopt.LD_TNEWTON(); abstol=tol, reltol=tol,  # bounds only
		# NLopt.LD_TNEWTON_RESTART(); abstol=tol, reltol=tol, # bounds only
		# NLopt.LD_TNEWTON_PRECOND(); abstol=tol, reltol=tol,
		# NLopt.LD_TNEWTON_PRECOND_RESTART(); abstol=tol, reltol=tol,
		# NLopt.LD_SLSQP(); abstol=tol, reltol=tol,
		## OptimizationPolyalgorithms.jl
		# PolyOpt(); reltol=tol, maxiters=80,
	)
	if !SciMLBase.successful_retcode(sol)
		error("Optimization failed with retcode $(sol.retcode)")
	end
	return sol.u
end;

# ╔═╡ 144aa951-56d1-4c07-a346-1d420be789a4
RSBweights = let β=inv(0.0), h=0.0, J0=0.0, M=0.0, kRSB=50
	RSBweights = initRSBweightsSK(β, kRSB)
	μmax = 8.25 #6.5 + 17sqrt(maximum(diff(-q))) / 2  # Heuristic that works(?)
	for nquad in (900, 1260)
		quadparams = Quadrature(gausslegendre, nquad, zero(μmax), μmax)
	# @profview begin
		RSBweights = solveRSBweightsSK(β, h, J0, M, RSBweights, quadparams)
	# end
	end
	RSBweights
end;

# ╔═╡ dff4b549-461b-4ae1-ad83-b8ffa2c75179
let β=inv(0.0)
	(; a, q) = RSBparams(β, RSBweights)
	afull = isinf(β) ? [a; 0.0; 0.0] : [β; a; 0.0; 0.0]
	qfull = isinf(β) ? [q; 0.0] : [1.0; q; 0.0]
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel=L"a", ylabel=L"q(a)")
	stairs!(ax, afull, qfull; step=:pre)
	# stairs!(ax, a2ξ.(afull).^sqrt(2), asin.(qfull) / (π / 2); step=:pre)
	# xlims!(ax, 0.0, 10.0)
	fig
end

# ╔═╡ 506da3f5-5744-4a7c-839c-49703fa87b28
Ginspect, Ginspect_quadgk, qparams = let β=inv(0.0), nquad=1260
	(; a, q) = RSBparams(β, RSBweights)
	dq = q[begin:(end - 1)] .- q[(begin + 1):end]
	μmax = 8.25  #6.6 + 17sqrt(maximum(diff(-q))) / 2  # Heuristic that works(?)
	@show μmax
	quadparams = Quadrature(gausslegendre, nquad, zero(μmax), μmax)
	Gvs = [G02.(quadparams.nodes, β)]
	for i in eachindex(a, dq)
		push!(Gvs, Gmap2(last(Gvs), a[i], dq[i], quadparams))
	end
	Gs_quadgk = let
		G0 = μ -> G02(μ, β)
		Gs = [G0, Gmap_quadgk(G0, a[1], dq[1])]
		for i in eachindex(a, dq)[(begin + 1):end]
			push!(Gs, Gmap_quadgk(last(Gs), a[i], dq[i]))
		end
		Gs
	end
	Gvs, Gs_quadgk, quadparams
end;

# ╔═╡ 507c83bc-b3f2-4368-8a5d-917ef81f2d13
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for Gv in Ginspect
		lines!(ax, qparams.nodes, Gv)
	end
	fig
end

# ╔═╡ b0161634-b100-4142-8326-38d03b1d9479
let
	(; a) = RSBparams(β, RSBweights)
	(; nodes) = qparams
	i = lastindex(a)
	diffplot(
		Ginspect_quadgk[i + 1].(nodes), Ginspect[i + 1], nodes;
		# exp.(a[i] .* Ginspect_quadgk[i + 1].(nodes)),
		# exp.(a[i] .* Ginspect[i + 1]),
		# nodes;
		flabel=L"\mathcal{G}_{%$i}",
		xlabel=L"\mu",
		approxlabel="Legendre",
		relativeto=first,
		relativetolabel=L"%$(first(nodes))",
	)
end

# ╔═╡ e671622a-9be2-4b90-82d5-5b6f85b76730
RSBweights

# ╔═╡ a0d70a8b-1303-4e4c-8268-5812836a40f3
(; a, q) = RSBparams(inv(0.0), RSBweights)

# ╔═╡ c3fc4ba5-fec9-4611-b2dc-add0ff09c499
dq = q[begin:(end - 1)] .- q[(begin + 1):end]

# ╔═╡ aac1aaa8-a679-425f-a649-10401e97bbc8
md"""
We incur significant loss of precision in ``\mathcal{G}(\mu)`` for large values of ``\mu`` by first computing ``\exp(a\mathcal{G}(\mu))`` and then taking the logarithm. A better implementation is to compute the equivalent of `expm1(a * G(μ))` and then use the `log1p` function to take the logarithm. This can be accomplished simply by using `expm1` instead of `exp` in the ``\mathcal{G}``-dependent part of the integrand, leaving the normalized Gaussian kernel unchanged. Thus:
"""

# ╔═╡ 00aaa315-2e4e-4b66-b8da-b9704cae2a35
md"""
```math
\begin{aligned}
&e^{a \mathcal{G}(\mu)} - 1 \\
&= \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{-(\mu' - |\mu|)^2 / 2 \Delta q} \left(\exp a\left(\mathcal{G}'(\mu') + |\mu'| - |\mu| - \frac{1}{2} a \Delta q\right) - 1\right) \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(|\mu| - \mu')^2 / 2 \Delta q} + e^{-(|\mu| + \mu')^2 / 2 \Delta q}\right) \\
&\times \left(\exp a\left(\mathcal{G}'(\mu') + \mu' - |\mu| - \frac{1}{2} a \Delta q\right) - 1\right) \\
\end{aligned}
```
"""

# ╔═╡ 47a0dbbc-111f-4b28-a73c-b8bc3a82e455
md"""
This could be the accuracy gain from the asymptotic renormalization group reformulation that @Schmidt2008Replica talked about! However, away from ``\mu' = |\mu|`` it relies on multiplying a large number by an even smaller number in order to obtain small values. This is what we avoided by aggressively combining exponents in our previous rewrite, a capability we lose now that we're computing values of `expm1` instead of `exp`. But in floating point, large times small isn't bad, the main things to avoid are (large + small) - large, log(1 + small), exp(small) - 1, et cetera. So as long as we don't end up in territory where `expm1` overflows to infinity, we're good.

Turns out it's not too hard to end up where `expm1` overflows. Let's look into absorbing that exponent in the quadratic term in a way that preserves the `expm1` form of the integrand. We got quite far earlier, we just need to extract some ``\operatorname{erf}``s.
"""

# ╔═╡ d91dffe5-f9c2-4bbc-864d-d167af7d1582
md"""
```math
\begin{aligned}
&e^{a \mathcal{G}(\mu)} \\
&= \int_{-\infty}^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} e^{-(\mu' - \mu)^2 / 2 \Delta q} \exp a\left(\mathcal{G}'(\mu') + |\mu'| - \mu - \frac{1}{2} a \Delta q\right) \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \Big(e^{-(\mu' - \mu)^2 / 2 \Delta q + a(\mu' - \mu) - a^2 \Delta q / 2)} + e^{-(\mu' + \mu)^2 / 2 \Delta q + a(\mu' - \mu) - a^2 \Delta q / 2)} \Big) e^{a \mathcal{G}'(\mu')} \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(\mu' - \mu - a \Delta q)^2 / 2 \Delta q} + e^{-(\mu' + \mu - a \Delta q)^2 / 2 \Delta q - 2a\mu} \right) e^{a \mathcal{G}'(\mu')} \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(\mu' - \mu - a \Delta q)^2 / 2 \Delta q} + e^{-(\mu' + \mu - a \Delta q)^2 / 2 \Delta q - 2a\mu} \right) \left(e^{a \mathcal{G}'(\mu')} - 1\right) \\
&+ \int_{-\infty}^0 \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(\mu' + \mu + a \Delta q)^2 / 2 \Delta q} + e^{-(\mu' - \mu + a \Delta q)^2 / 2 \Delta q - 2a\mu} \right) \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(\mu' - \mu - a \Delta q)^2 / 2 \Delta q} + e^{-(\mu' + \mu - a \Delta q)^2 / 2 \Delta q - 2a\mu} \right) \left(e^{a \mathcal{G}'(\mu')} - 1\right) \\
&+ \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{a \Delta q + \mu}{\sqrt{2\Delta q}}\right) + e^{-2a\mu} \left(1 + \operatorname{erf}\left(\frac{a \Delta q - \mu}{\sqrt{2\Delta q}}\right)\right)\right] \, ,
\end{aligned}
```
and thus,
```math
\begin{aligned}
&e^{a \mathcal{G}(\mu)} - 1 \\
&= \int_0^\infty \frac{d \mu'}{\sqrt{2 \pi \Delta q}} \left(e^{-(\mu' - \mu - a \Delta q)^2 / 2 \Delta q} + e^{-(\mu' + \mu - a \Delta q)^2 / 2 \Delta q - 2a\mu} \right) \left(e^{a \mathcal{G}'(\mu')} - 1\right) \\
&+ \frac{1}{2}\left[e^{-2a\mu}\operatorname{erfc}\left(\frac{\mu - a \Delta q}{\sqrt{2\Delta q}}\right) - \operatorname{erfc}\left(\frac{\mu + a \Delta q}{\sqrt{2\Delta q}}\right)\right] \, .
\end{aligned}
```
"""

# ╔═╡ eb91148a-81e1-44e1-8528-2d273e6deadc
md"""
Unfortunately, the exponential prefactor prevents us from taking advantage of the two-arg library function for difference of erfs, but the alternative is to have a negative integral term added to a positive difference-of-erfs term, opening the door to catastrophic cancellation.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ApproxFun = "28f2ccd6-bb30-5033-b560-165f7b14dc2f"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DispatchDoctor = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationMOI = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
OptimizationNLopt = "4e6fcdb7-1186-4e1f-a706-475e75c168bb"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProfileView = "c46f51b8-102a-5cf2-8d2c-8597cb0e0da7"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
ScopedValues = "7e506255-f358-4e82-b7e4-beb19740aa63"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
ApproxFun = "~0.13.27"
CairoMakie = "~0.12.5"
DispatchDoctor = "~0.4.11"
Enzyme = "~0.12.23"
FFTW = "~1.8.0"
FastGaussQuadrature = "~1.0.2"
Optimization = "~3.27.0"
OptimizationMOI = "~0.4.2"
OptimizationNLopt = "~0.2.2"
ProfileView = "~1.7.2"
QuadGK = "~2.9.4"
ScopedValues = "~1.2.1"
SpecialFunctions = "~2.4.0"
StaticArrays = "~1.9.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "8b470f6eac80b678549dd3ece515d39095a09b0a"

[[deps.ADTypes]]
git-tree-sha1 = "6778bcc27496dae5723ff37ee30af451db8b35fe"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.6.2"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.ATK_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a2ecb68d240333fe63bea1965b71884e98c2d0f0"
uuid = "7b86fcea-f67b-53e1-809c-8f1719c154e8"
version = "2.38.0+0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "f61b15be1d76846c0ce31d3fcfac5380ae53db6a"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.37"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7d5da5dd472490d048b081ca1bda4a7821b06456"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.1.1"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ApproxFun]]
deps = ["AbstractFFTs", "ApproxFunBase", "ApproxFunFourier", "ApproxFunOrthogonalPolynomials", "ApproxFunSingularities", "Calculus", "DomainSets", "FastTransforms", "LinearAlgebra", "RecipesBase", "Reexport", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "756514707948262303d65d0807c0ed01788e9bd5"
uuid = "28f2ccd6-bb30-5033-b560-165f7b14dc2f"
version = "0.13.27"
weakdeps = ["DualNumbers"]

    [deps.ApproxFun.extensions]
    ApproxFunDualNumbersExt = "DualNumbers"

[[deps.ApproxFunBase]]
deps = ["AbstractFFTs", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "Calculus", "Combinatorics", "DSP", "DomainSets", "DualNumbers", "FFTW", "FillArrays", "InfiniteArrays", "IntervalSets", "LazyArrays", "LinearAlgebra", "LowRankMatrices", "SparseArrays", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "709cf88109bf9babe765875629b93b423addf763"
uuid = "fbd15aa5-315a-5a7d-a8a4-24992e37be05"
version = "0.9.28"
weakdeps = ["Test"]

    [deps.ApproxFunBase.extensions]
    ApproxFunBaseTestExt = "Test"

[[deps.ApproxFunFourier]]
deps = ["AbstractFFTs", "ApproxFunBase", "BandedMatrices", "DomainSets", "FFTW", "FastTransforms", "InfiniteArrays", "IntervalSets", "LinearAlgebra", "Reexport", "StaticArrays"]
git-tree-sha1 = "73b8ebe2ae2ce377196088fcf7f1138d6f652c18"
uuid = "59844689-9c9d-51bf-9583-5b794ec66d30"
version = "0.3.28"

[[deps.ApproxFunOrthogonalPolynomials]]
deps = ["ApproxFunBase", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "DomainSets", "FastGaussQuadrature", "FastTransforms", "FillArrays", "HalfIntegers", "IntervalSets", "LinearAlgebra", "OddEvenIntegers", "Reexport", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "179e94caa8a8a9812b932a165cf42e198f9ddb42"
uuid = "b70543e2-c0d9-56b8-a290-0d4d6d4de211"
version = "0.6.56"
weakdeps = ["Polynomials", "Static"]

    [deps.ApproxFunOrthogonalPolynomials.extensions]
    ApproxFunOrthogonalPolynomialsPolynomialsExt = "Polynomials"
    ApproxFunOrthogonalPolynomialsStaticExt = "Static"

[[deps.ApproxFunSingularities]]
deps = ["ApproxFunBase", "ApproxFunOrthogonalPolynomials", "BlockBandedMatrices", "DomainSets", "HalfIntegers", "IntervalSets", "LinearAlgebra", "OddEvenIntegers", "Reexport", "SpecialFunctions"]
git-tree-sha1 = "7061a7d8f22a40cca68216d9af91c03880ba91ac"
uuid = "f8fcb915-6b99-5be2-b79a-d6dbef8e6e7e"
version = "0.3.21"
weakdeps = ["StaticArrays"]

    [deps.ApproxFunSingularities.extensions]
    ApproxFunSingularitiesStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "f54c23a5d304fb87110de62bace7777d59088c34"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.15.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "ce2ca959f932f5dad70697dd93133d1167cf1e4e"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "014bc22d6c400a7703c0f5dc1fdc302440cf88be"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.4"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "71f605effb24081b09cae943ba39ef9ca90c04f4"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.7.2"
weakdeps = ["SparseArrays"]

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bijections]]
git-tree-sha1 = "95f5c7e2d177b7ba1a240b0518038b975d72a8c0"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "5c0ffe1dff8cb7112de075f1b1cb32191675fcba"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.1.0"
weakdeps = ["BandedMatrices"]

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

[[deps.BlockBandedMatrices]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "FillArrays", "LinearAlgebra", "MatrixFactorizations"]
git-tree-sha1 = "cff6f0710fad7e0117b1df30760f54b2666378c0"
uuid = "ffab5731-97b5-5995-9138-79e8c1846df0"
version = "0.13.1"
weakdeps = ["SparseArrays"]

    [deps.BlockBandedMatrices.extensions]
    BlockBandedMatricesSparseArraysExt = "SparseArrays"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSTParser]]
deps = ["Tokenize"]
git-tree-sha1 = "0157e592151e39fa570645e2b2debcdfb8a0f112"
uuid = "00ebfdb7-1f24-5e51-bd34-a7502290713f"
version = "3.4.3"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "e4da5095557f24713bae4c9f50e34ff4d3b959c0"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "f8889d1770addf59d0a015c49a473fa2bdb9f809"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "b8fe8546d52ca154ac556809e10c75e6e7430ac8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.5"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonMark]]
deps = ["Crayons", "JSON", "PrecompileTools", "URIs"]
git-tree-sha1 = "532c4185d3c9037c0237546d817858b23cf9e071"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.12"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d8a9c0b6ac2d9081bf76324b39c78ca3ce4f0c98"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.6"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "f7f4319567fe769debfcf7f8c03d8da1dd4e2fb0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.9"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "8630ce72d9d2028cfdf6b55ffec1fc7515ef5ce2"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.1.2"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "35b66b6744b2d92c778afd3a88d2571875664a2a"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.2"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PreallocationTools", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Static", "StaticArraysCore", "Statistics", "Tricks", "TruncatedStacktraces"]
git-tree-sha1 = "d1e8a4642e28b0945bde6e2e1ac569b9e0abd728"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.151.5"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDistributionsExt = "Distributions"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "Functors", "LinearAlgebra", "Markdown", "NonlinearSolve", "Parameters", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "91e10deadcf1e33168bac3140eb0ea8cc4dfa5d7"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "3.7.0"

    [deps.DiffEqCallbacks.weakdeps]
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
    Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"

[[deps.DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "GPUArraysCore", "LinearAlgebra", "Markdown", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "Requires", "ResettableStacks", "SciMLBase", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "ed0158e758723b4d429afbbb5d98c5afd3458dc1"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.22.0"

    [deps.DiffEqNoiseProcess.extensions]
    DiffEqNoiseProcessReverseDiffExt = "ReverseDiff"

    [deps.DiffEqNoiseProcess.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "FillArrays", "LinearAlgebra", "PackageExtensionCompat", "SparseArrays", "SparseMatrixColorings"]
git-tree-sha1 = "6bd550abccb7aa156141e10d5367c580af9128af"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.5.11"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = "Enzyme"
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = "ForwardDiff"
    DifferentiationInterfacePolyesterForwardDiffExt = "PolyesterForwardDiff"
    DifferentiationInterfaceReverseDiffExt = "ReverseDiff"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTapirExt = "Tapir"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tapir = "07d77754-e150-4737-8c94-cd238a1fb45b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "739111000382a82ef7c94440c72f65ffca98cef2"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.12"
weakdeps = ["ChainRulesCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "490392af2c7d63183bfa2c8aaa6ab981c5ba7561"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.14"
weakdeps = ["Makie"]

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "30a1848c4f4fc35d1d4bbbd125650f6a11b5bc6c"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.7"

[[deps.DynamicQuantities]]
deps = ["Compat", "PackageExtensionCompat", "Tricks"]
git-tree-sha1 = "412b25c7d99ec6b06967d315c7b29bb8e484f092"
uuid = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
version = "0.13.2"

    [deps.DynamicQuantities.extensions]
    DynamicQuantitiesLinearAlgebraExt = "LinearAlgebra"
    DynamicQuantitiesMeasurementsExt = "Measurements"
    DynamicQuantitiesScientificTypesExt = "ScientificTypes"
    DynamicQuantitiesUnitfulExt = "Unitful"

    [deps.DynamicQuantities.weakdeps]
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random"]
git-tree-sha1 = "d182ebea02378543ebf83397d42c22e2e5fb4645"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.12.26"
weakdeps = ["ChainRulesCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "d445df66dd8761a4c27df950db89c6a3a0629fe7"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.7.7"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b5084294bdbdbb2cf1f102bc399872fcebc7e887"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.140+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Expronicon]]
deps = ["MLStyle", "Pkg", "TOML"]
git-tree-sha1 = "fc3951d4d398b5515f91d7fe5d45fc31dccb3c9b"
uuid = "6b7a57c9-7cc1-4fdf-b7f5-e857abae3636"
version = "0.8.5"

[[deps.Extents]]
git-tree-sha1 = "94997910aca72897524d2237c41eb852153b0f65"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.3"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "ab1b34570bcdf272899062e1a56285a53ecaae08"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.5"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FastLapackInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cbf5edddb61a43669710cbc2241bc08b36d9e660"
uuid = "29a986be-02c6-4525-aec4-84b980013641"
version = "2.0.4"

[[deps.FastTransforms]]
deps = ["AbstractFFTs", "BandedMatrices", "FFTW", "FastGaussQuadrature", "FastTransforms_jll", "FillArrays", "GenericFFT", "Libdl", "LinearAlgebra", "Reexport", "SpecialFunctions", "ToeplitzMatrices"]
git-tree-sha1 = "8ef4a98b14c07bf5ebb8c848e6634555523d49a6"
uuid = "057dd010-8810-581a-b7be-e3fc3b93f78c"
version = "0.16.4"

[[deps.FastTransforms_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "MPFR_jll", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "8cec4285578b83cec4f5aa13a13a4963595846df"
uuid = "34b6f7d7-08f9-5794-9e10-3819e4c7e49a"
version = "0.6.2+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
git-tree-sha1 = "fa0ba2042021409deb144f868abafde0b06be8f0"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "1.3.0"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "2de436b72c3422940cbe1367611d137008af7ec3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.23.1"
weakdeps = ["BandedMatrices", "BlockBandedMatrices", "StaticArrays"]

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "bd1aaf448be998ea427b1c7213b8acf2e278498b"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "1.0.0"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "64d8e93700c7a3f28f717d265382d52fac9fa1c1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.12"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.2.1+6"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ab29216184312f99ff957b32cd63c2fe9c928b91"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.26.7"

[[deps.GTK3_jll]]
deps = ["ATK_jll", "Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Libepoxy_jll", "Pango_jll", "Pkg", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXcomposite_jll", "Xorg_libXcursor_jll", "Xorg_libXdamage_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "Xorg_libXrender_jll", "at_spi2_atk_jll", "gdk_pixbuf_jll", "iso_codes_jll", "xkbcommon_jll"]
git-tree-sha1 = "b080a592525632d287aee4637a62682576b7f5e4"
uuid = "77ec8976-b24b-556a-a1bf-49a033a670a6"
version = "3.24.31+0"

[[deps.GenericFFT]]
deps = ["AbstractFFTs", "FFTW", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "1bc01f2ea9a0226a60723794ff86b8017739f5d9"
uuid = "a8297547-1b15-4a5a-a998-a2ac5f1cef28"
version = "0.1.6"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "9fff8990361d5127b770e3454488360443019bb3"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.5"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ebd18c326fa6cee1efb7da9a3b45cf69da2ed4d9"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.11.2"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "fc713f007cff99ff9e50accba6373624ddd33588"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Gtk]]
deps = ["Cairo", "Cairo_jll", "Dates", "GTK3_jll", "Glib_jll", "Graphics", "JLLWrappers", "Libdl", "Librsvg_jll", "Pkg", "Reexport", "Scratch", "Serialization", "Test", "Xorg_xkeyboard_config_jll", "adwaita_icon_theme_jll", "gdk_pixbuf_jll", "hicolor_icon_theme_jll"]
git-tree-sha1 = "b502c9f626930385658f818edc62218956204fb4"
uuid = "4c0ca9eb-093a-5379-98c5-f87ac0bbbf44"
version = "1.3.0"

[[deps.GtkObservables]]
deps = ["Cairo", "Colors", "Dates", "FixedPointNumbers", "Graphics", "Gtk", "IntervalSets", "LinearAlgebra", "Observables", "PrecompileTools", "Reexport", "RoundingIntegers"]
git-tree-sha1 = "7abc9367c964ed75a51ee85f8d65fad47af1ae77"
uuid = "8710efd8-4ad6-11eb-33ea-2d5ceb25a41c"
version = "1.2.9"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "6c3e57bc26728b99f470b267a437f0d380eac4fc"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "f672ebc641b814d7c8e5d62023c5c3f5f0f6ac50"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.14.2"
weakdeps = ["DSP", "Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysDSPExt = "DSP"
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "6336676b91409ec0a2c82723de9bdf4cabd02d78"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.8"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14eb2b542e748570b56446f4c50fbfb2306ebc45"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "433b0bb201cd76cb087b017e49244f10394ebe9c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.14"
weakdeps = ["DiffRules", "ForwardDiff", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "18c59411ece4838b18cd7f537e56cf5e41ce5bfd"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.15"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.JuliaFormatter]]
deps = ["CSTParser", "CommonMark", "DataStructures", "Glob", "Pkg", "PrecompileTools", "Tokenize"]
git-tree-sha1 = "4b0b04fe63e02eed776181e14ebc1f48d787ceec"
uuid = "98e50ef6-434e-11e9-1051-2b60c6c9e899"
version = "1.0.59"

[[deps.JumpProcesses]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "FunctionWrappers", "Graphs", "LinearAlgebra", "Markdown", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SciMLBase", "StaticArrays", "SymbolicIndexingInterface", "UnPack"]
git-tree-sha1 = "f12000093078e3dea1ee15de8bb35cfdc0014d97"
uuid = "ccbc3e58-028d-4f4c-8cd5-9ae44345cda5"
version = "9.12.0"
weakdeps = ["FastBroadcast"]

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "07649c499349dad9f08dde4243a4c597064663e9"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.6.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "9253429e28cceae6e823bec9ffde12460d79bb38"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.5"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "267dad6b4b7b5d529c76d40ff48d33f7e94cb834"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LBFGSB]]
deps = ["L_BFGS_B_jll"]
git-tree-sha1 = "e2e6f53ee20605d0ea2be473480b7480bd5091b5"
uuid = "5be7bae1-8223-5378-bac3-9e7378a2f6e6"
version = "0.4.1"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

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

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.L_BFGS_B_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "77feda930ed3f04b2b0fbb5bea89e69d3677c6b0"
uuid = "81d17ec3-03a1-5e46-b53e-bddc35a13473"
version = "3.0.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "e459fda6b68ea8684b3fcd513d2fd1e5130c4402"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.16.0"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "5b0d630f3020b82c0775a51d05895852f8506f50"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.4"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "b8ea0abe6cc872996e87356951d286d25d485aba"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.1.9"
weakdeps = ["BandedMatrices", "BlockArrays", "BlockBandedMatrices", "StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

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

[[deps.Libepoxy_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "7a0158b71f8be5c771e7a273183b2d0ac35278c5"
uuid = "42c93a91-0102-5b3f-8f9d-e41de60ac950"
version = "1.5.10+0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pango_jll", "Pkg", "gdk_pixbuf_jll"]
git-tree-sha1 = "ae0923dab7324e6bc980834f709c4cd83dd797ed"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.54.5+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "6355fb9a4d22d867318db186fd09b09b35bd2ed7"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.6.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "CpuId", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "00d9dc2ef7abb0575ec2b59fedbbc5685127e6ea"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "2.31.2"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveEnzymeExt = ["Enzyme", "EnzymeCore"]
    LinearSolveFastAlmostBandedMatricesExt = ["FastAlmostBandedMatrices"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolveRecursiveArrayToolsExt = "RecursiveArrayTools"

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.LowRankMatrices]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7c8664b2f3d5c3d9b77605c03d53b18813e79b0f"
uuid = "e65ccdef-c354-471a-8090-89bec1c20ec3"
version = "1.0.1"
weakdeps = ["FillArrays"]

    [deps.LowRankMatrices.extensions]
    LowRankMatricesFillArraysExt = "FillArrays"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.0+1"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "863b9e666b5a099c8835e85476a5834f9d77c4c1"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.5"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c1c950560397ee68ad7302ee0e3efa1b07466a2f"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.4"

[[deps.ManifoldDiff]]
deps = ["LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "Requires"]
git-tree-sha1 = "fcdcba81649ccbda6d28d55dd0ca03b3bb93df1f"
uuid = "af67fdf4-a580-4b9f-bbec-742ef357defd"
version = "0.3.10"

    [deps.ManifoldDiff.extensions]
    ManifoldDiffFiniteDiffExt = "FiniteDiff"
    ManifoldDiffFiniteDifferencesExt = "FiniteDifferences"
    ManifoldDiffForwardDiffExt = "ForwardDiff"
    ManifoldDiffReverseDiffExt = "ReverseDiff"
    ManifoldDiffZygoteExt = "Zygote"

    [deps.ManifoldDiff.weakdeps]
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Manifolds]]
deps = ["Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldDiff", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "58eb09899273a3ed17aae1f56435f440669b810c"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.9.20"

    [deps.Manifolds.extensions]
    ManifoldsBoundaryValueDiffEqExt = "BoundaryValueDiffEq"
    ManifoldsNLsolveExt = "NLsolve"
    ManifoldsOrdinaryDiffEqDiffEqCallbacksExt = ["DiffEqCallbacks", "OrdinaryDiffEq"]
    ManifoldsOrdinaryDiffEqExt = "OrdinaryDiffEq"
    ManifoldsRecipesBaseExt = ["Colors", "RecipesBase"]
    ManifoldsTestExt = "Test"

    [deps.Manifolds.weakdeps]
    BoundaryValueDiffEq = "764a87c0-6b3e-53db-9096-fe964310641d"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown", "Printf", "Random", "Requires"]
git-tree-sha1 = "3c196fc3c47bd8a852386aca443eda5874d536c1"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.15.12"

    [deps.ManifoldsBase.extensions]
    ManifoldsBasePlotsExt = "Plots"
    ManifoldsBaseQuaternionsExt = "Quaternions"
    ManifoldsBaseRecursiveArrayToolsExt = "RecursiveArrayTools"
    ManifoldsBaseStatisticsExt = "Statistics"

    [deps.ManifoldsBase.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    Quaternions = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "91b08d27a27d83cf1e63e50837403e7f53a0fd74"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.31.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "e1641f32ae592e415e3dbae7f4a188b5316d4b62"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.1"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "f765b4eda3ea9be8e644b9127809ca5151f3d9ea"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.4.2"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "07c98fdf57c9b45b987cf250c4bdc7200fa39eb2"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "3.0.0"
weakdeps = ["BandedMatrices"]

    [deps.MatrixFactorizations.extensions]
    MatrixFactorizationsBandedMatricesExt = "BandedMatrices"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "1b9e613f2ca3b6cdcbfe36381e17ca2b66d4b3a1"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MethodAnalysis]]
deps = ["AbstractTrees"]
git-tree-sha1 = "c2ee9b8f036c951f9ed0a47503a7f7dc0905b256"
uuid = "85b6ec6f-f7df-4429-9514-a64bcd9ee824"
version = "0.4.13"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.ModelingToolkit]]
deps = ["AbstractTrees", "ArrayInterface", "Combinatorics", "Compat", "ConstructionBase", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DiffEqNoiseProcess", "DiffRules", "Distributed", "Distributions", "DocStringExtensions", "DomainSets", "DynamicQuantities", "ExprTools", "Expronicon", "FindFirstFunctions", "ForwardDiff", "FunctionWrappersWrappers", "Graphs", "InteractiveUtils", "JuliaFormatter", "JumpProcesses", "LabelledArrays", "Latexify", "Libdl", "LinearAlgebra", "MLStyle", "NaNMath", "NonlinearSolve", "OrderedCollections", "PrecompileTools", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "SciMLStructures", "Serialization", "Setfield", "SimpleNonlinearSolve", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils", "Symbolics", "URIs", "UnPack", "Unitful"]
git-tree-sha1 = "f5830116c54bc0f5f9653f204ce719aa6cec477e"
uuid = "961ee093-0014-501f-94e3-6117800e7a78"
version = "9.29.0"

    [deps.ModelingToolkit.extensions]
    MTKBifurcationKitExt = "BifurcationKit"
    MTKDeepDiffsExt = "DeepDiffs"

    [deps.ModelingToolkit.weakdeps]
    BifurcationKit = "0f109fa4-8a5d-4b75-95aa-f515264e7665"
    DeepDiffs = "ab62b9b5-e342-54a8-a765-a90f495de1a6"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "5c1d1d9361e1417e5a065e1f84dc3686cbdaea21"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.6"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "d0a6b1096b584a2b88efb70a92f8cb8c881eb38a"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.6"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NLopt]]
deps = ["NLopt_jll"]
git-tree-sha1 = "3b887e2ef56be3309e68d2546c6178e2d2fa9a60"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.0.2"
weakdeps = ["MathOptInterface"]

    [deps.NLopt.extensions]
    NLoptMathOptInterfaceExt = ["MathOptInterface"]

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedDims]]
deps = ["LinearAlgebra", "Pkg", "Statistics"]
git-tree-sha1 = "90178dc801073728b8b2d0d8677d10909feb94d8"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "1.2.2"

    [deps.NamedDims.extensions]
    AbstractFFTsExt = "AbstractFFTs"
    ChainRulesCoreExt = "ChainRulesCore"
    CovarianceEstimationExt = "CovarianceEstimation"
    TrackerExt = "Tracker"

    [deps.NamedDims.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CovarianceEstimation = "587fd27a-f159-11e8-2dae-1979310e6154"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "LazyArrays", "LineSearches", "LinearAlgebra", "LinearSolve", "MaybeInplace", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SparseArrays", "SparseDiffTools", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "3adb1e5945b5a6b1eaee754077f25ccc402edd7f"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "3.13.1"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveBandedMatricesExt = "BandedMatrices"
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = "NLsolve"
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSymbolicsExt = "Symbolics"
    NonlinearSolveZygoteExt = "Zygote"

    [deps.NonlinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "195e0a19842f678dd3473ceafbe9d82dfacc583c"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.1"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OddEvenIntegers]]
git-tree-sha1 = "9980c824d61373cd473149480d3ef59119f382bf"
uuid = "8d37c425-f37a-4ca2-9b9d-a61bc06559d2"
version = "0.1.12"
weakdeps = ["HalfIntegers"]

    [deps.OddEvenIntegers.extensions]
    OddEvenIntegersHalfIntegersExt = "HalfIntegers"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a028ee3cb5641cccc4c24e90c36b0a4f7707bdf5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"
weakdeps = ["MathOptInterface"]

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LBFGSB", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Pkg", "Printf", "ProgressLogging", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "e51c209075bf413ee1dbf4a2473972276965022c"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "3.27.0"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "Manifolds", "PDMats", "Reexport", "Requires", "SciMLBase", "SparseArrays", "SymbolicAnalysis", "SymbolicIndexingInterface", "Symbolics"]
git-tree-sha1 = "c154d3d569c47bbc1da356b8e5e981b08dc068cd"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "1.3.3"

    [deps.OptimizationBase.extensions]
    OptimizationEnzymeExt = "Enzyme"
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMTKExt = "ModelingToolkit"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSparseDiffExt = ["SparseDiffTools", "ReverseDiff"]
    OptimizationTrackerExt = "Tracker"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseDiffTools = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationMOI]]
deps = ["MathOptInterface", "ModelingToolkit", "Optimization", "Reexport", "SciMLStructures", "SparseArrays", "SymbolicIndexingInterface", "Symbolics"]
git-tree-sha1 = "065ed47c8bb8012e8fefc0d723f4b8e1511ad768"
uuid = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
version = "0.4.2"

[[deps.OptimizationNLopt]]
deps = ["NLopt", "Optimization", "Reexport"]
git-tree-sha1 = "83b7f5218c6ff83f65fff32923630fe8be9f5131"
uuid = "4e6fcdb7-1186-4e1f-a706-475e75c168bb"
version = "0.2.2"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cb5a2ab6763464ae0f19c86c56c63d4a2b0f5bda"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PoissonRandom]]
deps = ["Random"]
git-tree-sha1 = "a0f1159c33f846aa77c3f30ebbc69795e5327152"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.4"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "9ff799e8fb8ed6717710feee3be3bc20645daa97"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.15"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "1a9cfb2dc2c2f1bd63f1906d72af39a79b49b736"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.11"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "d7f3f63331c7c8c81245b4ee2815b7d496365833"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.23"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileView]]
deps = ["Cairo", "Colors", "Dates", "FileIO", "FlameGraphs", "Graphics", "Gtk", "GtkObservables", "InteractiveUtils", "IntervalSets", "MethodAnalysis", "PrecompileTools", "Preferences", "Profile", "Requires", "UUIDs"]
git-tree-sha1 = "64acb8d47eb0f7a3f9389fac00b4c7520b962e5c"
uuid = "c46f51b8-102a-5cf2-8d2c-8597cb0e0da7"
version = "1.7.2"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "b450d967a770fb13d0e26358f58375e20361cf9c"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.26.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "6db1a75507051bc18bfa131fbc7c3f169cc4b2f6"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.23"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RoundingIntegers]]
git-tree-sha1 = "99acd97f396ea71a5be06ba6de5c9defe188a778"
uuid = "d5f540fe-1c90-5db3-b776-2e2f362d9394"
version = "1.1.0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "2803cab51702db743f3fda07dd1745aadfbf43bd"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.5.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "Expronicon", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "7f0e208db50f5fee2386b6d8dc9608d580059331"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.48.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Setfield", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "10499f619ef6e890f3f4a38914481cc868689cd5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.8"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "20ad3e7c137156c50c93c888d0f2bc5b7883c729"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.4.2"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "eef2fbac9538ee6cc60ee1489f028d2f8a1a5249"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.2.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "DiffResults", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "Reexport", "SciMLBase", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "03c21a4c373c7c3aa77611430068badaa073d740"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "1.11.0"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"
    SimpleNonlinearSolveZygoteExt = "Zygote"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseDiffTools]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "PackageExtensionCompat", "Random", "Reexport", "SciMLOperators", "Setfield", "SparseArrays", "StaticArrayInterface", "StaticArrays", "Tricks", "UnPack", "VertexSafeGraphs"]
git-tree-sha1 = "469f51f8c4741ce944be2c0b65423b518b1405b0"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "2.19.0"

    [deps.SparseDiffTools.extensions]
    SparseDiffToolsEnzymeExt = "Enzyme"
    SparseDiffToolsPolyesterExt = "Polyester"
    SparseDiffToolsPolyesterForwardDiffExt = "PolyesterForwardDiff"
    SparseDiffToolsSymbolicsExt = "Symbolics"
    SparseDiffToolsZygoteExt = "Zygote"

    [deps.SparseDiffTools.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "ad048e784b816e4de5553a13f1daf148412f3dbd"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.3.6"

[[deps.Sparspak]]
deps = ["Libdl", "LinearAlgebra", "Logging", "OffsetArrays", "Printf", "SparseArrays", "Test"]
git-tree-sha1 = "342cf4b449c299d8d1ceaf00b7a49f4fbc7940e7"
uuid = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
version = "0.3.9"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "8963e5a083c837531298fc41599182a759a87a6d"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
deps = ["Test"]
git-tree-sha1 = "010dc73c7146869c042b49adcdb6bf528c12e859"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymbolicAnalysis]]
deps = ["DSP", "DataStructures", "Dictionaries", "Distributions", "DomainSets", "IfElse", "LinearAlgebra", "LogExpFunctions", "Manifolds", "PDMats", "RecursiveArrayTools", "StatsBase", "SymbolicUtils", "Symbolics"]
git-tree-sha1 = "770c6d4df4ec39d328c12778510fd5bba1c20d31"
uuid = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
version = "0.1.0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "2dd32da03adaf43fd91494e38ef3df0ab2e6c20e"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.27"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "fb099adbd7504f1e68b4512828e9d94197a8b889"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.1"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TermInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "9345b7b8a2923abaf6089d9f7306bb712fecb840"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "2.1.2"

[[deps.Symbolics]]
deps = ["ADTypes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "ForwardDiff", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "653f8337e6605b4a84e5da15add78f1de8f4bae5"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.35.1"

    [deps.Symbolics.extensions]
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxCoreExt = "LuxCore"
    SymbolicsPreallocationToolsExt = "PreallocationTools"
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    LuxCore = "bb33d45b-7691-41d6-9220-0943567d0623"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "6f0cee95e74d1f6891ba6b35b8b219fd3d11b567"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.4.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "bc7fd5c91041f44636b2c134041f7e5263ce58ae"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "df4e499f6321e72f801aab45336ba76ed06e97db"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.3"
weakdeps = ["StatsBase"]

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

[[deps.Tokenize]]
git-tree-sha1 = "468b4685af4abe0e9fd4d7bf495a6554a6276e75"
uuid = "0796e94c-ce3b-5d07-9a54-7f471281c624"
version = "0.5.29"

[[deps.TranscodingStreams]]
git-tree-sha1 = "96612ac5365777520c3c5396314c8cf7408f436a"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.1"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "be986ad9dac14888ba338c2554dcfec6939e1393"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.2.1"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "e7f5b81c65eb858bed630fe006837b935518aca5"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.70"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "d9717ce3518dc68a99e6b96300813760d887a01d"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.1+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcomposite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll"]
git-tree-sha1 = "7c688ca9c957837539bbe1c53629bb871025e423"
uuid = "3c9796d7-64a0-5134-86ad-79f8eb684845"
version = "0.4.5+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdamage_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll"]
git-tree-sha1 = "fe4ffb2024ba3eddc862c6e1d70e2b070cd1c2bf"
uuid = "0aeada51-83db-5f97-b67e-184615cfc6f6"
version = "1.1.5+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libXtst_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "Xorg_libXi_jll"]
git-tree-sha1 = "0c0a60851f44add2a64069ddf213e941c30ed93c"
uuid = "b6f176f1-7aea-5357-ad67-1d3e565ea1c6"
version = "1.2.3+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.adwaita_icon_theme_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "hicolor_icon_theme_jll"]
git-tree-sha1 = "60f62ea463bfd4288ce94d0eac5d521a6e62e10b"
uuid = "b437f822-2cd6-5e08-a15c-8bac984d38ee"
version = "3.33.93+0"

[[deps.at_spi2_atk_jll]]
deps = ["ATK_jll", "Artifacts", "JLLWrappers", "Libdl", "Pkg", "XML2_jll", "Xorg_libX11_jll", "at_spi2_core_jll"]
git-tree-sha1 = "f16ae690aca4761f33d2cb338ee9899e541f5eae"
uuid = "de012916-1e3f-58c2-8f29-df3ef51d412d"
version = "2.34.1+4"

[[deps.at_spi2_core_jll]]
deps = ["Artifacts", "Dbus_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXtst_jll"]
git-tree-sha1 = "d2d540cd145f2b2933614649c029d222fe125188"
uuid = "0fc3237b-ac94-5853-b45c-d43d59a06200"
version = "2.34.0+4"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "86e7731be08b12fa5e741f719603ae740e16b666"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.10+0"

[[deps.hicolor_icon_theme_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b458a6f6fc2b1a8ca74ed63852e4eaf43fb9f5ea"
uuid = "059c91fe-1bad-52ad-bddd-f7b78713c282"
version = "0.17.0+3"

[[deps.iso_codes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d837a5d2a19d54243dafb6cc98d0f590a603dfa1"
uuid = "bf975903-5238-5d20-8243-bc370bc1e7e5"
version = "4.15.1+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═2b6685d9-8b9f-4f58-9f22-cfc2058cabe8
# ╟─6ead3e9a-227f-11ef-1940-cbc927e46201
# ╟─a2bf328f-393e-4c87-a6c5-ff2e3920ffe0
# ╟─006edbbb-f585-4a2a-a351-2d0ce9f5e8e2
# ╟─463c134f-700b-4fa1-8f1e-bb4efd9a3cb8
# ╟─17a08b3a-32b6-43eb-b367-063fde038034
# ╟─57d22519-9cfc-4610-a67f-1054b3b0aacd
# ╟─62a0d0f4-d5e5-4ecd-aaae-8380ce2fe534
# ╟─95ab5021-113d-4ad7-9b12-ced70b0f9820
# ╟─e5089794-3554-446e-b334-2ed9a4516032
# ╟─e3b8887e-226e-4627-ab82-cab722c7a627
# ╟─52fe5149-e50c-48e6-8a00-d78e05369906
# ╟─fec9c08f-e827-42c2-97fe-60a23c791436
# ╠═840d726e-708f-4e01-9632-c605d05cfac2
# ╠═ccfe01dd-9879-43ca-bd10-db3fe99a035c
# ╠═92a23b3b-5ad1-4dda-919f-ea020051242a
# ╟─d48c3912-b7f7-4599-bd3a-d9d52b856cee
# ╟─c3da10d3-d997-4047-8c80-d3d7266c071c
# ╟─c356ae36-56b1-40b9-8e46-d68b03a4ef50
# ╟─3a1afb31-ca60-4655-972b-7e42a1a5eae5
# ╟─c803242d-503d-4af6-b39a-62341ea255f8
# ╟─c6050ea9-836a-4dac-8a39-7be23fbd2dd5
# ╟─73f96e10-d0ca-40f4-a7c9-8a36962ff8ef
# ╟─d10ad4af-d383-429d-a3b0-b679229f6b65
# ╠═a0fc11ad-99e9-4abb-96f1-c1c7a92e4d70
# ╟─9df607cb-4d4b-4d25-9a06-bcf12dc342a6
# ╟─08f3a66e-a6c3-4d76-b817-ad2b27a0ce00
# ╟─ccd108bf-28e9-4385-b9fb-be687be17f09
# ╟─af9c0340-17f2-4fc8-9b0b-aa4816b07d59
# ╟─8d3d6bad-4eab-4061-9a67-3be4e89b72a0
# ╟─0c7a4d65-1c78-4c80-a86a-7357514f775b
# ╟─18866c6e-b147-4cda-9dcd-c4edcefa5651
# ╠═ba321c1e-d0da-4339-9426-0683d9979745
# ╟─d6bbf6f7-8b39-46fc-b15c-3a9f53f630d9
# ╠═6f44e78b-05fb-4d63-b540-de2cdd4e4b81
# ╠═7c955c2a-54b1-489d-b953-629f6d8878d6
# ╠═5036c3f1-e753-47db-8612-0dd3becb600c
# ╠═92fbae7f-07ee-48c6-aed7-0865b05df6d7
# ╠═ef8f4ca8-5f80-4c5f-9490-f97f34db517e
# ╠═9d1c8bf4-153c-47b7-9397-11cfbc9ccc1a
# ╠═d399f6b4-3c9d-4c07-aa7a-f6f03964032a
# ╠═ee72bb96-446e-41f3-9016-b68e1f4e58b8
# ╠═b571d11f-e7ac-4ed0-8e9f-c5f00a7dc295
# ╠═ec713b7c-626b-46c8-a3ea-1be1afaa3c50
# ╠═5ac52242-0e89-4a6a-a254-bb8f1685348d
# ╠═6cacb213-1886-4385-bb52-8b34dcda686e
# ╠═1a064d9a-ff05-4097-891f-0c97b1493850
# ╠═69dea82e-fa2a-480b-92d3-a184f87a48eb
# ╟─2606a0a5-6062-4f7b-97ea-80dfe1aa4927
# ╟─31b8c01c-309f-4293-937a-2e5bfd055e86
# ╟─44fdc36d-9c80-4c83-96f1-1fd37c0d9146
# ╟─e25bbce0-e73b-4169-90b6-dfead0d76acf
# ╟─59cc072d-a823-417c-ba0f-468531ac337b
# ╟─fa5d715b-7ec1-492d-9cba-8b164854868f
# ╠═f6de0959-95b7-42e5-b79a-5b09e51abdb0
# ╟─0ed97298-396b-43a7-bb5e-38aee668d50d
# ╟─bde42de4-9245-4594-844e-16a3a54d3e3f
# ╠═90f26134-21ee-4328-bdb4-b7e6e479cbcd
# ╠═2986bf5b-fb0f-4791-85f6-2517e9ab2d81
# ╟─f769c886-ad0d-4d7a-9977-fed541c476cc
# ╟─4bb351f9-5800-4fdd-a7fe-2a6009315073
# ╟─a17151fe-6078-4569-8129-e5242409caf7
# ╠═ab0c84db-168e-410b-91e4-2396d4fd1c56
# ╠═249a427b-c112-4d17-8889-db94c72a7225
# ╠═8e10cd1a-4e0f-4be2-8a43-d890df758b8e
# ╠═53d18dc9-bb10-4fdc-a0c9-c46b792764a7
# ╟─cdb60bf4-2e1a-4bd8-b4bc-95ec4b85d83a
# ╟─4d796285-45ef-491e-b9a3-00a27d1fad7a
# ╠═99c700e5-7233-4fb6-b415-6aaa79da0faf
# ╠═08c3e556-63d3-4f22-b69e-5be3f7a99e98
# ╟─6a8da0f5-e43f-435f-a326-2eee8560859e
# ╟─9285938d-dd43-4132-bdb0-2e8aba84140c
# ╠═3820fe1e-eb76-4f33-9e1b-9e162b76d4c6
# ╠═8e6e8a69-1677-4e6f-a0bc-0d5304b83263
# ╠═31310c3f-6269-49df-8aef-9a60ace637dc
# ╠═4a659c8b-4b36-4a39-a7c8-bc49a0770e51
# ╠═b5819306-7e49-4139-a22b-e426980d0ec4
# ╟─24bd7873-96b1-4462-87af-0f132752195b
# ╟─fb8f67b2-3abb-44bf-bca7-b84d2f1e8b9f
# ╠═6e0e83d1-9cf8-4e4b-a2f9-48670f9ac54c
# ╠═e49c231b-2bfd-4986-8c80-2715e4ebb4d3
# ╠═cdb6f1e9-c420-42dc-b06f-5e235ee07e8f
# ╠═d8aa9d5b-4363-4c55-80f1-b7d8eb15d347
# ╠═d2b13b6e-2b5c-465c-a647-344f3495867b
# ╟─9f18c4cd-152c-4ec4-84b3-5e868fc0b5a9
# ╟─98660249-52ff-4880-a4fb-71757844cb9b
# ╠═e7a500fa-24ea-47bc-aa1e-3a60fabe6435
# ╠═01780f9b-95a9-4601-82ca-10af89a10375
# ╠═8177c265-91fb-4cd7-8652-f4cfb3c8aa2f
# ╠═b12cd52d-1f43-4def-9a92-b425526e4a28
# ╠═12f69cae-7017-4fde-89ea-54f701b4bd68
# ╟─16e88b60-d354-4ba9-83a9-cbc53c4d17b6
# ╟─9e33f253-15d9-4405-a7eb-a6c386d246b4
# ╠═4d3c6872-b462-4efe-abf4-28d4c6cdb332
# ╠═c49877b1-e92b-4514-9439-e3c10c6b4e6b
# ╠═36c3f5a2-7fcc-4b5a-9109-25e5a994d1c3
# ╠═06ff0708-675d-4008-9812-e35d785fe054
# ╠═223aa38b-ee9e-4da8-9d19-9c440ee15ca5
# ╟─ad934872-4b43-47a1-b98b-cc848ff2c3b3
# ╟─72459310-d9b1-49a6-b19c-fba1df996773
# ╟─6228dbd3-71fd-4d84-8cab-847a255697a5
# ╟─e8c6826b-6d20-4bf2-a63b-12d59f2c1c4e
# ╠═e95c9adf-93f7-4471-a1ca-f24c106012ad
# ╠═c2e34d7e-8989-48e3-945c-8f42fed07ead
# ╠═76fe38c8-5a24-40d5-abd2-bddc4fe981da
# ╠═29ae03f4-8653-4fd8-94be-710b124e66b4
# ╠═304d7c54-2e35-411d-adda-3f3e6f0d00a2
# ╠═439a8caa-fee3-44f9-9a00-7520d0cf85cd
# ╠═1c0094fd-b2a4-40a4-be8b-bdaebeeff044
# ╠═0edc06a7-f4d3-4d14-8c74-b4bf17531202
# ╠═95f6d0d0-8ecf-41c7-b176-55015ce10b5e
# ╠═ec340551-8322-4af4-8117-280cbd5b67a2
# ╠═874e4462-fdfb-4a8d-92aa-fcaf3780f28a
# ╠═2c091404-d4d3-4f76-845d-9ad04aa2fbae
# ╠═10839b94-c3ab-4dd6-8880-601a7a0d2839
# ╠═144aa951-56d1-4c07-a346-1d420be789a4
# ╠═dff4b549-461b-4ae1-ad83-b8ffa2c75179
# ╠═506da3f5-5744-4a7c-839c-49703fa87b28
# ╠═507c83bc-b3f2-4368-8a5d-917ef81f2d13
# ╠═b0161634-b100-4142-8326-38d03b1d9479
# ╠═e671622a-9be2-4b90-82d5-5b6f85b76730
# ╠═a0d70a8b-1303-4e4c-8268-5812836a40f3
# ╠═c3fc4ba5-fec9-4611-b2dc-add0ff09c499
# ╟─aac1aaa8-a679-425f-a649-10401e97bbc8
# ╟─00aaa315-2e4e-4b66-b8da-b9704cae2a35
# ╟─47a0dbbc-111f-4b28-a73c-b8bc3a82e455
# ╟─d91dffe5-f9c2-4bbc-864d-d167af7d1582
# ╟─eb91148a-81e1-44e1-8528-2d273e6deadc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
