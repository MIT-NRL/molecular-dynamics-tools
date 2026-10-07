# Carbon diffraction and crystallite-size comparisons

The lazy public helpers in `molecular_dynamics_tools.carbon_diffraction` follow
selected measurements from Putman et al., DOI
[10.1016/j.carbon.2023.03.040](https://doi.org/10.1016/j.carbon.2023.03.040),
[author preprint](https://arxiv.org/abs/2212.06354). They do not reproduce the
authors' TOPAS profile model or million-atom simulations.

```python
import molecular_dynamics_tools as mdt

# Pure-carbon elastic coherent intensity per atom; this is not raw detector intensity.
intensity = mdt.carbon_xray_form_factor(q)**2 * sq
peak = mdt.fit_carbon_diffraction_peak(q, intensity, reflection="10")
distribution = mdt.summarize_crystallite_size_distribution(
    individual_sizes_A,
    support_min_A=minimum_accepted_domain_size_A,
    diffraction_size_A=peak["summary"]["La_A"],
)
calibration = mdt.calibrate_graphene_diffraction()
```

## Peak fitting

Q uses `4*pi*sin(theta)/wavelength`, in inverse Å. The default windows are
(10): 2.5–3.6, (002): 1.2–2.2, (11): 4.5–5.7 and (004): 3.2–4.2 inverse Å.
`fit_carbon_diffraction_peak` returns scalar `summary` and tabular `curve`.
The empirical split pseudo-Voigt has independent left/right half-widths,
a common mixing fraction and a linear background. It does **not** separate
strain, stacking disorder or Warren line-shape physics.

Q-space Scherrer conversion is `L=2*pi*K/FWHM_Q`, with overridable default
K=1 for (10)/(11) and 0.9 for (002)/(004). An optional wavelength uses the
angular-width relation instead. Formal errors assume independent,
equal-variance residuals; they exclude correlated MD frames, finite-cell
effects, profile choice and shape-factor uncertainty.

Sparse, low-contrast, unresolved, boundary-truncated, missing-bin or
unidentifiable fits retain diagnostics but withhold the size. NaN observations
and residuals stay unavailable. A supplied instrument width uses explicitly
labelled Gaussian quadrature subtraction, not a general pseudo-Voigt
deconvolution. A measured instrumental profile should instead be convolved
with the prediction. The caller must establish finite-cell and Fourier-cutoff
convergence before interpreting apparent coherence lengths physically.

`carbon_xray_form_factor` supplies neutral-carbon elastic f0(Q) in electrons,
without anomalous corrections. Multiplying S(Q) by f0(Q)^2 excludes detector
background, Compton, polarization and instrumental terms.

## Fig. 8-style distributions

`summarize_crystallite_size_distribution` accepts **individual finite-domain
sizes**, including repeated observations, rather than histogram centers.
Periodic spanning domains must be excluded by the caller. It returns number
mean/median, the exact weighted mean, empirical weighted modal histogram bin
and optional fractions below a diffraction size. The default `L**2.5` weight
is a proxy, not a calculated diffraction pattern. Histogram densities
integrate to one over their bin widths.

The lower-truncated exponential maximum-likelihood scale is
`mean(L-lower_support)` and the weighted fitted mode is
`max(lower_support, exponent*scale)`. This estimator differs from the paper's
histogram least-squares fit. Supply physical lower support from the geometric
minimum-ring criterion when possible; otherwise sample minimum is used and
recorded. Fewer than 20 observations, fewer than five distinct sizes or
effectively zero width leave the fit unavailable. The empirical distribution
remains available.

A descriptive CDF deviation assesses the exponential assumption, without a
significance test that would incorrectly assume independent domains from
correlated frames. A deviation above 0.2 flags the fit for review while retaining
the fitted parameters and curves; this is a descriptive threshold, not a
statistical significance criterion. `curve` includes histogram densities and bin-averaged
fitted densities; `fit_curve` includes analytic curves spanning lower support
and weighted mode. Fitted tails beyond the largest observed domain are model
extrapolations.

## Small Fig. 3-style calibration

`calibrate_graphene_diffraction` cuts isolated circular atom discs from a
perfect honeycomb lattice, computes their powder Debye patterns and freely
fits the maximum (10) intensity to a size power law. Both **per-fragment**
and **per-atom** exponents are exported. The paper's 2.5 is a reference value,
never imposed. Per-atom normalization removes the area contribution to the
exponent, so the per-fragment result supplies the appropriate comparison.

The explicit size convention is `L=sqrt(N*bulk_graphene_area_per_atom)`, with
area per atom `3*sqrt(3)*bond_length**2/4`. This counts edge atoms with their
bulk area, differing from summing fully enclosed hexagon areas. Actual and
requested sizes and atom counts are exported; small fragments have important
perimeter corrections. This bounded small suite does not reproduce the paper's
50 nm example or establish a universal exponent. A maximum at the Q-window
edge withholds the calibration fit. Size-range and Q-grid convergence are
required before applying a newly fitted exponent to production domains.

`carbon_debye_pattern` exposes the isolated-particle calculation:

* `S_q=sum_ij sinc(Q*r_ij)/N`, dimensionless with S(0)=N;
* `I_per_atom=fC(Q)**2*S_q`, electron squared per atom;
* `I_per_fragment=N*I_per_atom`, electron squared per fragment.

It is nonperiodic, intended for bounded fragment calibration. Blocked pair
histograms avoid N-by-N matrices; spectral scratch arrays are bounded.
Defaults limit fragments to 5000 atoms; explicit limits also cover pair block
size, Q points and histogram bins. Pair histogram midpoint errors depend on
the recorded distance-bin width, rather than arbitrary smoothing.
