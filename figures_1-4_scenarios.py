# %% [markdown]
# # Between Roots and Future: Figures 1–4 (time series of the four scenarios)
# Each scenario of Section 4.2 is integrated from the uniform initial condition $x_i(0)=0.25$ over $t\in[0,300]$,
# with the payoff matrix $M^*$ of Table 1 and selection intensity $w=1$.
# The script saves `Figure_1.pdf` … `Figure_4.pdf` and prints the long-run state reported in each caption.
# Only `numpy`, `scipy` and `matplotlib` are required.

# %%
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

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

W = 1.0                                   # selection intensity
X0 = [0.25, 0.25, 0.25, 0.25]             # uniform initial condition (x_N, x_DL, x_D, x_H)
t = np.linspace(0, 300, 1500)
COLORS = ['#2c3e50', '#95a5a6', '#c0392b', '#f1c40f']

def long_run_state(k, x0=X0, T=40000):
    """State reached at time T (LSODA, rtol 1e-10, atol 1e-13): the value reported in the captions."""
    s = solve_ivp(rhs, [0, T], x0, args=(M0, W, k), method='LSODA', rtol=1e-10, atol=1e-13)
    return np.clip(s.y[:, -1], 0, None)

# %% [markdown]
# ## Figures 1–4

# %%
for n, s in enumerate('ABCD', start=1):
    k = SCEN[s]
    sol = odeint(rhs, X0, t, args=(M0, W, k), tfirst=True)

    plt.figure(figsize=(12, 5))
    for i in range(4):
        plt.plot(t, sol[:, i], label=NAMES[i], color=COLORS[i], lw=2.5)
    plt.xlabel("Time ($t$)", fontsize=12)
    plt.ylabel("Relative Frequency", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'Figure_{n}.pdf', format='pdf', bbox_inches='tight')   # vector PDF, as required by the journal
    plt.show()

    x = long_run_state(k)
    print(f"Scenario {s} (Figure {n}): state at t = {t[-1]:.0f}: ({', '.join(f'{v:.4f}' for v in sol[-1])})")
    print(f"   long-run state: ({', '.join(f'{v:.4f}' for v in x)}),  CPI L = {CPI(x):.2f}\n")

# %% [markdown]
# ## Other initial conditions (optional)
# Integrates one scenario from a list of initial conditions $(x_N, x_{DL}, x_D, x_H)$ and prints the final states.
# Each initial condition must sum to one.

# %%
scenario = 'D'
initial_conditions = [
    [0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 1.00],
    [0.80, 0.00, 0.20, 0.00],
    [0.00, 0.10, 0.10, 0.80],
    [0.09, 0.10, 0.81, 0.00],
]
for x0 in initial_conditions:
    x = long_run_state(SCEN[scenario], x0=x0)
    print(f"Scenario {scenario}, x(0) = {x0}  ->  ({', '.join(f'{v:.4f}' for v in x)}),  CPI L = {CPI(x):.2f}")
