# %% [markdown]
# # Between Roots and Future: equilibria and Jacobian matrices of the Supplementary Material (Section 1)
# Exact symbolic computations (SymPy) at the baseline (Scenario D, $w=1$):
# * the equilibria of Section 1.1: the quadratic equations for $P_2$ and $P_3$, and the reduction of the interior
#   equilibrium $E^*$ to a cubic, including the case $16x_3=5$;
# * the Jacobian matrices of Section 1.2: symbolic Jacobian of the reduced system with $x_D$ eliminated, its value at
#   $P_1$, $P_2$, $P_3$, $E^*$, characteristic polynomials and eigenvalues.
#
# Each check prints `OK` if the result agrees with the Supplementary Material, `MISMATCH` otherwise.
# Requires `sympy` (preinstalled on Google Colab).

# %%
import numpy as np
import sympy as sp

# ------------------------------ Model --------------------------------
# State vector x = (x_N, x_DL, x_D, x_H): shares of Nostalgic, Disillusioned, Depressed and Hopeful.
TYPES = ['N', 'DL', 'D', 'H']
NAMES = ['Nostalgic (N)', 'Disillusioned (DL)', 'Depressed (D)', 'Hopeful (H)']

# Payoff parameters (Table 1): common good multiplier m; probabilities p (Nostalgic) and q (Disillusioned)
# of preserving when meeting a Depressed member; endowment c; investments b (regenerating) and a (preserving).
PAY = dict(m=1.8, p=0.8, q=0.6, c=5.0, b=3.0, a=2.0)

# Conversion rates, with the notation of the paper:
#   kH   = kappa_H       Fading              H  -> N
#   kN   = kappa_N       Spontaneous exit    N  -> DL
#   kDH  = kappa_{D,H}   Rescue              D  -> H
#   kD   = kappa_D       Spontaneous exit    D  -> DL
#   kDLD = kappa_{DL,D}  Contagion           DL -> D
#   kDLH = kappa_{DL,H}  Testimony           DL -> H
BASE = dict(kH=0.15, kN=0.15, kDH=0.45, kD=0.05, kDLD=0.65, kDLH=0.30)          # baseline (Scenario D)
SCEN = {'A': dict(kH=0.05, kN=0.15, kDH=0.20, kD=0.20, kDLD=0.21, kDLH=0.70),   # Thriving
        'B': dict(kH=0.20, kN=0.10, kDH=0.05, kD=0.05, kDLD=0.60, kDLH=0.20),   # Deteriorating
        'C': dict(kH=0.70, kN=0.00, kDH=0.05, kD=0.70, kDLD=0.05, kDLH=0.05),   # Stagnant
        'D': dict(BASE)}                                                         # Baseline

def rates(**changes):
    """Baseline conversion rates with optional changes, e.g. rates(kDLH=0)."""
    k = dict(BASE); k.update(changes)
    return k

def payoff(m=1.8, p=0.8, q=0.6, c=5.0, b=3.0, a=2.0):
    """Payoff matrix: entry (i, j) is the payoff of type i meeting type j (Section 3.2)."""
    al, be = m/2 - 1, m/2                                                   # alpha, beta
    return np.array([[c+(al+be)*a, c+(al+be)*a, c+al*a*p, c+al*a+be*b],     # N
                     [c+(al+be)*a, c+(al+be)*a, c+al*a*q, c+al*a+be*b],     # DL
                     [c+be*p*a,    c+be*q*a,    c,        c+be*b],          # D
                     [c+al*b+be*a, c+al*b+be*a, c+al*b,   c+(al+be)*b]])    # H

def rhs(t, x, M, w, k):
    """Right-hand side of the system of Section 3.4: replicator term w*x_i*E_i(x) plus conversion flows."""
    N, DL, D, H = x
    f = M @ x; E = f - f @ x                                                # excess payoffs E_i = f_i - phi
    return np.array([w*N*E[0]  - k['kN']*N + k['kH']*H,
                     w*DL*E[1] + k['kN']*N + k['kD']*D - k['kDLD']*DL*D - k['kDLH']*DL*H,
                     w*D*E[2]  + k['kDLD']*DL*D - k['kDH']*D*H - k['kD']*D,
                     w*H*E[3]  + k['kDLH']*DL*H + k['kDH']*D*H - k['kH']*H])

def CPI(x, pay=PAY):
    """Community Preservation Index L (Section 3.2)."""
    a, b, p, q = pay['a'], pay['b'], pay['p'], pay['q']
    N, DL, D, H = x
    return N*(1-(1-p)*D) + DL*(1-(1-q)*D) + (b/a)*H

M0 = payoff(**PAY)                                                          # matrix M* of Table 1

def check(name, ok, detail=''):
    print(f"{'OK      ' if ok else 'MISMATCH'}  {name}" + (f":  {detail}" if detail else ''))

Q = lambda v: sp.Rational(str(v))                     # exact rational from a decimal value
PAYQ = {key: Q(v) for key, v in PAY.items()}          # payoff parameters of Table 1, exact
KQ = {key: Q(v) for key, v in BASE.items()}           # baseline conversion rates, exact
W = 1                                                 # selection intensity
MQ = payoff(**PAYQ)                                   # matrix M* of Table 1 with exact entries
assert isinstance(MQ[0, 0], sp.Basic)
print('Matrix M* (Table 1):')
sp.pprint(sp.Matrix(MQ))

# Reduced system: x_D = 1 - x_N - x_DL - x_H, as in Section 1.2 of the Supplementary.
# The right-hand side is the same function used in all the other scripts.
xN, xDL, xH = sp.symbols('x_N x_DL x_H')
xD = 1 - xN - xDL - xH
field = rhs(0, np.array([xN, xDL, xD, xH], dtype=object), MQ, W, KQ)
F = sp.Matrix([sp.expand(field[i]) for i in (0, 1, 3)])          # equations for x_N, x_DL, x_H
J = F.jacobian([xN, xDL, xH])                                    # symbolic Jacobian

# %% [markdown]
# ## 1. Boundary equilibria (Section 1.1.1)

# %%
s = sp.symbols('s')
P1 = {xN: 0, xDL: 1, xH: 0}                                     # all-Disillusioned vertex
# Case 1 (face x_D = 0): x_H solves 8 x_H^2 - 32 x_H + 3 = 0, then x_N = 3x_H/(3-2x_H), x_DL = 9/(12-8x_H)
hH = [r for r in sp.solve(8*s**2 - 32*s + 3, s) if 0 < float(r) < 1][0]
P2 = {xN: 3*hH/(3 - 2*hH), xDL: 9/(12 - 8*hH), xH: hH}
# Case 3 (edge x_N = x_H = 0): x_DL solves -0.64 x_DL^2 + 0.77 x_DL - 0.05 = 0
hDL = [r for r in sp.solve(-Q(0.64)*s**2 + Q(0.77)*s - Q(0.05), s) if 0 < float(r) < 1][0]
P3 = {xN: 0, xDL: hDL, xH: 0}
print('x_H at P2 =', hH, '   x_DL at P3 =', hDL)

def coords(pt):
    """(x_N, x_DL, x_D, x_H) as floats."""
    n, l, h = (float(sp.N(pt[v], 30)) for v in (xN, xDL, xH))
    return np.array([n, l, 1 - n - l - h, h])

check('P2 = (0.103, 0.801, 0, 0.096)', np.abs(coords(P2) - [0.103, 0.801, 0, 0.096]).max() < 6e-4, str(np.round(coords(P2), 4)))
check('P3 = (0, 0.069, 0.931, 0)', np.abs(coords(P3) - [0, 0.069, 0.931, 0]).max() < 6e-4, str(np.round(coords(P3), 4)))

# %% [markdown]
# ## 2. Interior equilibrium (Section 1.1.2)
# Coordinates $x_1=x_N$, $x_2=x_{DL}$, $x_3=x_D$, $x_4=x_H=1-x_1-x_2-x_3$; stationarity conditions [N], [D], [H].

# %%
x1, x2, x3 = sp.symbols('x1 x2 x3')
x4 = 1 - x1 - x2 - x3
Y = np.array([x1, x2, x3, x4], dtype=object)
f = MQ @ Y
E = f - f @ Y                                                     # excess payoffs E_i = f_i - phi
EN, ED, EH = (sp.expand(E[i]) for i in (0, 2, 3))
stat_N = sp.expand(x1*EN - KQ['kN']*x1 + KQ['kH']*x4)
stat_D = sp.expand(ED + KQ['kDLD']*x2 - KQ['kDH']*x4 - KQ['kD'])
stat_H = sp.expand(EH + KQ['kDLH']*x2 + KQ['kDH']*x3 - KQ['kH'])

check('E_D and E_H contain no terms in x1^2, x2^2, x1*x2',
      all(sp.Poly(e, x1, x2, x3).coeff_monomial(mon) == 0 for e in (ED, EH) for mon in (x1**2, x2**2, x1*x2)))
check('[D] and [H] are linear in x1', sp.degree(stat_D, x1) == 1 and sp.degree(stat_H, x1) == 1)
aH = sp.factor(stat_H.coeff(x1, 1))
check('in [H] the coefficient of x1 is proportional to 16 x3 - 5', sp.cancel(aH/(16*x3 - 5)).is_number, str(aH))

# Case 16 x3 = 5
H5 = sp.expand(stat_H.subs(x3, Q(5)/16))
check('[H] with 16 x3 = 5 is proportional to 165/16 - 40 x2', sp.cancel(H5/(Q(165)/16 - 40*x2)).is_number, str(H5))
x2c = sp.solve(H5, x2)[0]
check('x2 = 33/128', x2c == Q(33)/128, str(x2c))
rootsN = [float(r) for r in sp.Poly(stat_N.subs({x3: Q(5)/16, x2: x2c}), x1).nroots() if r.is_real]
rootsD = [float(r) for r in sp.Poly(stat_D.subs({x3: Q(5)/16, x2: x2c}), x1).nroots() if r.is_real]
print('   [N] = 0 at x1 =', np.round(rootsN, 4), '   [D] = 0 at x1 =', np.round(rootsD, 4), '  (Supplementary: about 0.25 and 0.33)')
check('[N] and [D] have no common root: no equilibrium with 16 x3 = 5',
      all(abs(a - b) > 1e-6 for a in rootsN for b in rootsD))

# Case 16 x3 != 5
x1_sol = sp.solve(stat_H, x1)[0]
check('x1 = (15 - 64 x2 x3 - 20 x2 - 15 x3) / (2 (16 x3 - 5))',
      sp.simplify(x1_sol - (15 - 64*x2*x3 - 20*x2 - 15*x3)/(2*(16*x3 - 5))) == 0)
numD = sp.fraction(sp.together(stat_D.subs(x1, x1_sol)))[0]
rel = -64*x2*x3 - 52*x2 - 59*x3 + 37
check('[D] reduces to -64 x2 x3 - 52 x2 - 59 x3 + 37 = 0', sp.cancel(numD/rel).is_number, str(sp.factor(numD)))
x3_sol = sp.solve(rel, x3)[0]
check('x3 = (37 - 52 x2) / (64 x2 + 59)', sp.simplify(x3_sol - (37 - 52*x2)/(64*x2 + 59)) == 0)
numN = sp.factor(sp.fraction(sp.together(stat_N.subs(x1, x1_sol).subs(x3, x3_sol)))[0])
Rx = 5120*x2**3 - 6192*x2**2 + 7467*x2 - 2030
check('[N] reduces to the cubic R(x2) = 5120 x2^3 - 6192 x2^2 + 7467 x2 - 2030',
      sp.Poly(numN, x2).rem(sp.Poly(Rx, x2)).is_zero)
print('   other factors of the numerator:', sp.factor(sp.cancel(numN/Rx)),
      '(they vanish only at x2 < 0 and at x2 = 33/128, the case 16 x3 = 5)')
disc = sp.discriminant(sp.diff(Rx, x2), x2)
check("R'(x2) has negative discriminant, so R is strictly increasing", disc < 0, str(disc))
rr = sp.Poly(Rx, x2).real_roots()
check('R has exactly one real root', len(rr) == 1)
x2E = rr[0].evalf(30)
x3E = x3_sol.subs(x2, x2E)
x1E = x1_sol.subs({x2: x2E, x3: x3E})
Estar = {xN: x1E, xDL: x2E, xH: 1 - x1E - x2E - x3E}
check('E* = (0.2523, 0.3411, 0.2383, 0.1683)',
      np.abs(coords(Estar) - [0.2523, 0.3411, 0.2383, 0.1683]).max() < 6e-5, str(np.round(coords(Estar), 4)))

# %% [markdown]
# ## 3. Jacobian matrices, characteristic polynomials and eigenvalues (Section 1.2)

# %%
EQUILIBRIA = {'P1': P1, 'P2': P2, 'P3': P3, 'E*': Estar}
PAPER_J = {'P1': [[-0.15, 0.0, 0.15], [0.23, 0.08, -0.12], [0.0, 0.0, 0.05]],
           'P2': [[-0.17998, -0.03958, 0.12068], [0.27975, 0.11054, -0.03051], [-0.07643, -0.04761, -0.06682]],
           'P3': [[-0.2579, 0.0, 0.15], [0.1705, -0.6349, -0.0138], [0.0, 0.0, 0.0445]],
           'E*': [[-0.1159, 0.0034, 0.1401], [0.2867, 0.0168, 0.0424], [-0.0796, -0.0163, -0.0756]]}
PAPER_EIG = {'P1': [-0.15, 0.08, 0.05],
             'P2': [0.0233, -0.0798+0.0308j, -0.0798-0.0308j],
             'P3': [-0.6349, -0.2579, 0.0445],
             'E*': [-0.0738+0.0835j, -0.0738-0.0835j, -0.0271]}
TOL_J = {'P1': 1e-12, 'P2': 6e-6, 'P3': 6e-5, 'E*': 6e-5}
lam = sp.symbols('lambda')

def same_spectrum(ev, paper, tol=6e-5):
    """Every reported eigenvalue is close to a computed one, and vice versa."""
    paper = np.asarray(paper, complex)
    return all(np.abs(ev - p).min() < tol for p in paper) and all(np.abs(paper - e).min() < tol for e in ev)

for name, pt in EQUILIBRIA.items():
    print(f'\n--- {name} = {np.round(coords(pt), 4)} ---')
    residual = max(abs(float(sp.N(Fi.subs(pt), 30))) for Fi in F)
    check(f'{name} is an equilibrium', residual < 1e-20, f'residual {residual:.1e}')
    Jn = J.subs(pt).evalf(30)
    J_np = np.array(Jn.tolist(), dtype=float)
    print('Jacobian (coordinates x_N, x_DL, x_H):'); print(np.round(J_np, 5))
    check(f'J({name}) as in the Supplementary', np.abs(J_np - np.array(PAPER_J[name])).max() <= TOL_J[name])
    print('characteristic polynomial det(J - lambda I):', sp.expand((Jn - lam*sp.eye(3)).det()).evalf(8))
    ev = np.linalg.eigvals(J_np)
    ev = ev[np.lexsort((ev.imag, np.round(ev.real, 10)))]
    print('eigenvalues:', np.round(ev, 4))
    check(f'eigenvalues of {name} as in the Supplementary', same_spectrum(ev, PAPER_EIG[name]))
    print('locally asymptotically stable' if ev.real.max() < 0 else 'unstable: at least one eigenvalue with positive real part')
