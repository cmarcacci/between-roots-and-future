# Between Roots and Future: An Evolutionary Approach to Charismatic Communities

Code reproducing the figures and the numerical results of the paper by Luigino Bruni and Carmine Marcacci.

## Files

Each script exists in two equivalent versions: a notebook (`.ipynb`, ready for Google Colab or Jupyter) and a plain Python file (`.py`).

| File | Output | Running time |
|---|---|---|
| `figures_1-4_scenarios` | Figures 1–4 (`Figure_1.pdf` … `Figure_4.pdf`): time series of Scenarios A–D, with the long-run states reported in the captions | a few seconds |
| `figures_5-7_robustness` | Figures 5–7 (`Figure_5.pdf`, `Figure_6.pdf`, `Figure_7.pdf`): two-parameter diagram, one-parameter diagrams, selection intensity | a few minutes |
| `figures_8-12_simplices` | Figures 8–12 (`Figure_8.pdf` … `Figure_12.pdf`): ternary plots of the faces of the simplex | under a minute |
| `reproduce_numerical_results` | Checks of all numerical values of the paper and the Supplementary Material: each line prints the value reported in the paper, the computed value and `OK` or `MISMATCH` | a few minutes |
| `jacobians_supplementary` | Exact symbolic computations (SymPy) for Section 1 of the Supplementary Material: equilibria $P_2$, $P_3$, $E^*$ (including the reduction of $E^*$ to a cubic), Jacobian matrices, characteristic polynomials and eigenvalues at $P_1$, $P_2$, $P_3$, $E^*$, each compared with the Supplementary (`OK` or `MISMATCH`) | under a minute |

## How to run

* **Google Colab:** open the notebook and choose *Runtime → Run all*.
* **Locally:** `pip install numpy scipy matplotlib sympy python-ternary`, then `python <file>.py`. The package `python-ternary` is needed only for `figures_8-12_simplices`, and `sympy` only for `jacobians_supplementary`. On Google Colab only `python-ternary` has to be installed, which the notebook does in its first cell.

## Notation

The state vector is `x = (x_N, x_DL, x_D, x_H)`, the shares of Nostalgic, Disillusioned, Depressed and Hopeful members. All scripts share the same model block (parameters, payoff matrix, right-hand side of the system, Community Preservation Index).

| Code | Paper | Meaning |
|---|---|---|
| `m, p, q, c, b, a` | $m, p, q, c, b, a$ | payoff parameters (Table 1) |
| `w` | $w$ | selection intensity |
| `kH` | $\kappa_H$ | Fading, H → N |
| `kN` | $\kappa_N$ | spontaneous exit, N → DL |
| `kDH` | $\kappa_{D,H}$ | Rescue, D → H |
| `kD` | $\kappa_D$ | spontaneous exit, D → DL |
| `kDLD` | $\kappa_{DL,D}$ | Contagion, DL → D |
| `kDLH` | $\kappa_{DL,H}$ | Testimony, DL → H |
| `Qstar`, `Fstar` | $Q^*$, $F^*$ | equilibria of Propositions 1 and 2 (closed forms) |
| `CPI` | $L$ | Community Preservation Index |

## Numerical settings

* Long-run states: LSODA (`scipy.integrate.solve_ivp`), relative tolerance 1e-10, absolute tolerance 1e-13, integration up to t = 40000.
* Jacobian matrices of Section 1.2 of the Supplementary: computed symbolically with SymPy and evaluated at the exact equilibria. In the bifurcation analysis the Jacobian restricted to the simplex is computed by central differences with step 1e-7.
* Stable equilibria: Newton's method (`scipy.optimize.fsolve`) from 66 starting points; a point is accepted as an equilibrium if the residual is below 1e-10, and as stable if all eigenvalues of the Jacobian restricted to the simplex have real part below −1e-7.
* Global convergence at the baseline: 1200 initial conditions drawn uniformly from the interior of the simplex, integrated up to t = 6000; along each trajectory the shares stay non-negative and their sum stays equal to one within 1e-9.
* Time series (Figures 1–4) and ternary plots (Figures 8–12): `scipy.integrate.odeint` with default tolerances, on t ∈ [0, 300] and t ∈ [0, 600] respectively; in the ternary plots the projection is not drawn when the displayed shares sum to less than 1e-5.
