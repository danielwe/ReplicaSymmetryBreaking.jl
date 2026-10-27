# ReplicaSymmetryBreaking.jl

The goal of this package is to calculate the free energy of the Sherrington-Kirkpatrick (SK) model to arbitrary finite order of replica symmetry breaking (RSB), using adaptive interpolants and quadrature routines to control the error in the recursive integrals, while keeping the implementation differentiable such that the optimization over order parameters can obtain the Jacobian through autodifferentiation. The approach otherwise follows the development in Schmidt (2008)[^Schmidt2008Replica].

The subsequent goal is to generalize the approach to other models that yield different integral kernels and asymptotic renormalization group flows, such as the soft-spin model studied in Yamamura et al. (2024)[^Yamamura2024Geometric].

The approach relies on <https://github.com/EnzymeAD/Enzyme.jl/pull/1852> and a variant of <https://github.com/JuliaMath/QuadGK.jl/pull/120>.

The basic goal of solving for the SK replica order parameters was achieved, but the project was put on the back burner in the middle of subsequent work to improve and generalize the implementation. Thus, the package may be difficult to use in its current state.

[^Schmidt2008Replica]: Schmidt, “Replica Symmetry Breaking at Low Temperatures.” <https://opus.bibliothek.uni-wuerzburg.de/frontdoor/index/index/docId/2621>

[^Yamamura2024Geometric]: Yamamura et al., “Geometric Landscape Annealing as an Optimization Principle Underlying the Coherent Ising Machine.” <https://doi.org/10.1103/PhysRevX.14.031054>
