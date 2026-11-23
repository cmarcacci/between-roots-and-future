# %% [markdown]
# # Between Roots and Future: Figures 5–7 (robustness and bifurcation analysis, Section 4.3)
# * `Figure_5.pdf`: two-parameter diagram in the contagion rate $\kappa_{DL,D}$ and the testimony rate $\kappa_{DL,H}$.
# * `Figure_6.pdf`: one-parameter diagrams in testimony, contagion, rescue and the multiplier $m$.
# * `Figure_7.pdf`: stable equilibria of the four scenarios as functions of the selection intensity $w$.
#
# Stable equilibria are located by Newton's method from 66 starting points, followed by the eigenvalues of the
# Jacobian restricted to the simplex. The whole script runs in a few minutes; the two-parameter grid
# (101 × 101 points) takes the longest.

# %%
import numpy as np, warnings, zipfile
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from multiprocessing import Pool

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

# ------------------------ Analysis helpers ---------------------------
def longrun(k, w=1.0, M=M0, x0=(.25, .25, .25, .25), T=40000):
    """State reached at time T (LSODA, rtol 1e-10, atol 1e-13)."""
    s = solve_ivp(rhs, [0, T], x0, args=(M, w, k), method='LSODA', rtol=1e-10, atol=1e-13)
    return np.clip(s.y[:, -1], 0, None)

def jac_red(x, k, w=1.0, M=M0, h=1e-7):
    """Jacobian in the coordinates (x_N, x_DL, x_H), with x_D = 1 - x_N - x_DL - x_H (as in the Supplementary).
    Its eigenvalues are those of the Jacobian restricted to the tangent space of the simplex."""
    def F(y):
        z = np.array([y[0], y[1], 1-y.sum(), y[2]]); d = rhs(0, z, M, w, k); return d[[0, 1, 3]]
    y0 = np.array([x[0], x[1], x[3]])
    return np.array([(F(y0+h*e) - F(y0-h*e))/(2*h) for e in np.eye(3)]).T

def eigenvalues(x, k, w=1.0, M=M0):
    """Eigenvalues of the Jacobian restricted to the simplex, sorted by real and then imaginary part."""
    ev = np.linalg.eigvals(jac_red(x, k, w, M))
    return ev[np.lexsort((ev.imag, np.round(ev.real, 10)))]   # rounding keeps conjugate pairs in a fixed order

def stable_eq(k, w=1.0, M=M0, n=60, seed=0):
    """All locally stable equilibria in the simplex: Newton's method from n+6 starting points, then eigenvalues."""
    rng = np.random.default_rng(seed); found = []
    res = lambda y: rhs(0, np.append(y, 1-y.sum()), M, w, k)[:3]
    starts = [rng.dirichlet(np.ones(4)*al) for al in (1, 0.3) for _ in range(n//2)]
    starts += [np.array(v, float) for v in ([0,1,0,0], [0,.1,.9,0], [.1,.8,0,.1], [.3,.2,0,.5], [.25]*4, [0,.5,.5,0])]
    for x0 in starts:
        y, info, ier, msg = fsolve(res, x0[:3], full_output=True, xtol=1e-13)
        x = np.append(y, 1-y.sum())
        if ier == 1 and np.abs(rhs(0, x, M, w, k)).max() < 1e-10 and x.min() > -1e-9:
            x = np.clip(x, 0, None)
            if not any(np.abs(x-z).max() < 1e-5 for z in found): found.append(x)
    return [x for x in found if eigenvalues(x, k, w, M).real.max() < -1e-7]

def deltas(pay=PAY):
    """Payoff differences delta_H, delta_D, gamma_D (Section 4.2)."""
    a, b, m, q = pay['a'], pay['b'], pay['m'], pay['q']
    al, be = m/2 - 1, m/2
    return abs(al)*(b-a), a*((m-1)-be*q), abs(al)*a*q

def Qstar(k, w=1.0, pay=PAY):
    """Equilibrium Q* without Depressed (Proposition 1, closed form for x_H*)."""
    u = w*deltas(pay)[0]; K = k['kDLH'] - u
    A, B, C = u*K, -K*(k['kN']+u+k['kH']), k['kN']*K - k['kN']*k['kH']
    h = -C/B if abs(A) < 1e-14 else (-B - np.sqrt(B*B - 4*A*C))/(2*A)
    dl = (k['kH'] + u*(1-h))/k['kDLH']
    return np.array([1-h-dl, dl, 0.0, h])

def Fstar(k, w=1.0, pay=PAY):
    """Equilibrium F* without Hopeful on the DL-D edge (Proposition 2, closed form for x_D*)."""
    dH, dD, gD = deltas(pay); s = w*(dD+gD)
    A, B, C = k['kDLD'] + w*gD, w*(2*dD+gD) - k['kDLD'], 4*s*k['kD']
    d = (B + np.sqrt(A*A - C))/(2*s)
    return np.array([0.0, 1-d, d, 0.0])

def growth_D(x, k, w=1.0, M=M0):
    """Per-capita growth rate of a small share of Depressed at state x (invasion condition)."""
    f = M @ x; E = f - f @ x
    return w*E[2] + k['kDLD']*x[1] - k['kDH']*x[3] - k['kD']

def growth_H(x, k, w=1.0, M=M0):
    """Per-capita growth rate of a small share of Hopeful at state x (invasion condition)."""
    f = M @ x; E = f - f @ x
    return w*E[3] + k['kDLH']*x[1] + k['kDH']*x[2] - k['kH']

def label(x, th=1e-4):
    """Types present at state x, e.g. 'N+DL+H'."""
    return '+'.join(n for n, v in zip(TYPES, x) if v > th)

k0 = rates(); dH, dD, gD = deltas()
COL = {'N': '#2c3e50', 'DL': '#95a5a6', 'D': '#c0392b', 'H': '#e6b800'}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
def save(fig, name):
    fig.savefig(name + '.pdf', dpi=300); fig.savefig(name + '.png', dpi=170); plt.show()

# %% [markdown]
# ## Figure 7: selection intensity

# %%
def w_branches(k, ws):
    """Primary branch (continues the w=0 attractor) and second stable equilibrium, if any."""
    prim, sec, prev = [], [], None
    for w in ws:
        s = stable_eq(k, w)
        if prev is None: prev = s[0] if s else longrun(k, w)
        d = [np.abs(x - prev).sum() for x in s]
        if s and min(d) < 0.15:
            i = int(np.argmin(d)); x1 = s[i]; others = [x for j, x in enumerate(s) if j != i]
        else:                                   # primary not found by Newton: follow it by simulation
            x1 = longrun(k, w, x0=0.98*prev + 0.005)
            others = [x for x in s if np.abs(x - x1).sum() > 1e-3]
        prim.append(x1); sec.append(others[0] if others else np.full(4, np.nan)); prev = x1
    return np.array(prim), np.array(sec)

def first_bistable(k, lo, hi, n=14):
    """Smallest w in [lo, hi] with two stable equilibria (bisection)."""
    for _ in range(n):
        mid = (lo + hi)/2
        if len(stable_eq(k, mid, n=200)) >= 2: hi = mid
        else: lo = mid
    return hi

kA, kB, kC, kD_ = SCEN['A'], SCEN['B'], SCEN['C'], SCEN['D']
thrA = [brentq(lambda w: Qstar(kA, w)[3] - 0.5, 1.0, 2.0), (kA['kDLH'] - kA['kH'])/dH]
thrB = [(kB['kDLD'] - kB['kD'])/dD]
thrD = [brentq(lambda w: growth_D(Qstar(kD_, w), kD_, w), 1.0, 1.1),
        first_bistable(kD_, 1.12, 1.19), (kD_['kDLH'] - kD_['kH'])/dH]
print('thresholds in w   A:', np.round(thrA, 3), ' B:', np.round(thrB, 3), ' D:', np.round(thrD, 3))

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6))
cfg = [('A', 'Scenario A (Thriving)', kA, 8, thrA, [1.02, 1.02]),
       ('B', 'Scenario B (Deteriorating)', kB, 2, thrB, [1.02]),
       ('C', 'Scenario C (Stagnant)', kC, 2, [], []),
       ('D', 'Scenario D (Baseline)', kD_, 2, thrD, [1.09, 1.02, 1.02])]
for ax, (s, title, k, wmax, thr, ypos) in zip(axes.flat, cfg):
    if s == 'C':                                    # continuum of equilibria: use x_i(0)=0.25
        ws = np.linspace(0, wmax, 41); X = np.array([longrun(k, w) for w in ws]); S = np.full_like(X, np.nan)
    else:
        ws = np.unique(np.concatenate([np.linspace(0, wmax, 241)] + [np.linspace(t-0.02, t+0.02, 40) for t in thr]))
        ws = ws[(ws >= 0) & (ws <= wmax)]; X, S = w_branches(k, ws)
    for i, key in enumerate(TYPES):
        ax.plot(ws, X[:, i], color=COL[key], lw=1.8, label=NAMES[i])
        ax.plot(ws, np.where(S[:, i] > 1e-6, S[:, i], np.nan), '--', color=COL[key], lw=1.3)   # second equilibrium
    two = ~np.isnan(S[:, 0])
    if two.any(): ax.axvspan(ws[two].min(), wmax, color='0.9', lw=0, zorder=0)
    for t, y in zip(thr, ypos):
        ax.axvline(t, color='0.35', lw=0.8, ls=':')
        ax.text(t, y, ('%.2f' % t) if abs(t-round(t, 1)) > 1e-6 else ('%.1f' % t), ha='center', va='bottom',
                fontsize=7, color='0.2', transform=ax.get_xaxis_transform())
    ax.axvline(1.0, color='0.6', lw=0.8); ax.set_xlim(0, wmax); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, ls='--', lw=0.4, color='0.85'); ax.set_title(title, loc='left', pad=24, fontsize=9)
    ax.set_xlabel(r'Selection intensity $w$'); ax.set_ylabel('Equilibrium share')
h, l = axes[0, 0].get_legend_handles_labels()
h = h[:4] + [Line2D([], [], color='0.3', ls='--', lw=1.3), Patch(color='0.9')]
l = l[:4] + ['Second stable equilibrium', 'Two stable equilibria coexist']
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=2.0)
save(fig, 'Figure_7')

# %% [markdown]
# ## Figure 6: one parameter at a time

# %%
def pm(v):
    """Payoff parameters with multiplier m = v."""
    pay = dict(PAY); pay['m'] = v; return pay

def branch(vals, fn):
    X, L = [], []
    for v in vals:
        Mv, k, pay = fn(v); s = stable_eq(k, 1.0, Mv)
        assert len(s) <= 1, ('multistability at', v)            # multistability check
        x = s[0] if s else longrun(k, 1.0, Mv)
        X.append(x); L.append(CPI(x, pay))
    return np.array(X), np.array(L)

thr1 = {'kDLH': [brentq(lambda v: growth_D(Qstar(rates(kDLH=v)), rates(kDLH=v)), 0.26, 0.9)],
        'kDLD': [brentq(lambda v: growth_D(Qstar(rates(kDLD=v)), rates(kDLD=v)), 0.3, 0.9)],
        'kDH':  [brentq(lambda v: growth_H(Fstar(rates(kDH=v)), rates(kDH=v)), 0.1, 0.9),
                 brentq(lambda v: growth_D(Qstar(rates(kDH=v)), rates(kDH=v)), 0.5, 0.95)],
        'm':    [brentq(lambda v: growth_H(Fstar(k0, pay=pm(v)), k0, M=payoff(**pm(v))), 1.6, 1.8),
                 brentq(lambda v: growth_D(Qstar(k0, pay=pm(v)), k0, M=payoff(**pm(v))), 1.75, 1.9)]}
print('thresholds around the baseline:', {n: [round(t, 3) for t in v] for n, v in thr1.items()})
panels = [(r'Testimony rate $\kappa_{DL,H}$', np.linspace(0, 1, 201), lambda v: (M0, rates(kDLH=v), PAY), 0.30, thr1['kDLH'], [0]),
          (r'Contagion rate $\kappa_{DL,D}$', np.linspace(0, 1, 201), lambda v: (M0, rates(kDLD=v), PAY), 0.65, thr1['kDLD'], [0]),
          (r'Rescue rate $\kappa_{D,H}$', np.linspace(0, 1, 201), lambda v: (M0, rates(kDH=v), PAY), 0.45, thr1['kDH'], [0, 0]),
          (r'Multiplier $m$', np.linspace(1.02, 1.98, 193), lambda v: (payoff(**pm(v)), k0, pm(v)), 1.80, thr1['m'], [-0.035, 0.035])]
fig = plt.figure(figsize=(6.8, 7.0))
gs = fig.add_gridspec(5, 2, height_ratios=[3, 1.15, 1.25, 3, 1.15], hspace=0.10, wspace=0.30)
for n, (xl, vals, fn, base, ths, off) in enumerate(panels):
    r = 0 if n < 2 else 3; c = n % 2
    X, L = branch(vals, fn)
    ax = fig.add_subplot(gs[r, c]); ax2 = fig.add_subplot(gs[r+1, c], sharex=ax)
    for i, key in enumerate(TYPES): ax.plot(vals, X[:, i], color=COL[key], lw=1.8, label=NAMES[i])
    ax2.plot(vals, L, color='0.15', lw=1.6); ax2.axhline(1.0, color='0.5', lw=0.8, ls='--')
    for a_ in (ax, ax2):
        a_.axvline(base, color='0.6', lw=0.8); a_.grid(True, ls='--', lw=0.4, color='0.85')
        for t in ths: a_.axvline(t, color='0.35', lw=0.8, ls=':')
    for t, o in zip(ths, off):
        ax.text(t+o, 1.03, '%.2f' % t, ha='center', va='bottom', fontsize=7, color='0.2', transform=ax.get_xaxis_transform())
    ax.set_ylim(-0.02, 1.02); ax.set_ylabel('Equilibrium share'); plt.setp(ax.get_xticklabels(), visible=False)
    ax2.set_ylim(0, 1.55); ax2.set_yticks([0, 0.5, 1, 1.5]); ax2.set_ylabel('CPI $L$'); ax2.set_xlabel(xl)
    ax.set_xlim(vals[0], vals[-1])
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h + [Line2D([], [], color='0.15', lw=1.6)], l + ['CPI $L$'], loc='lower center', ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.15)
save(fig, 'Figure_6')

# %% [markdown]
# ## Figure 5: contagion versus testimony

# %%
G = 101; xs = np.linspace(0, 1, G); ys = np.linspace(0, 1, G)       # xs: kDLD (contagion), ys: kDLH (testimony)
def grid_column(i):
    out = []
    for j in range(G):
        k = rates(kDLD=xs[i], kDLH=ys[j]); s = stable_eq(k)
        u = s[0] if len(s) == 1 else longrun(k, T=20000)            # attractor reached from x_i(0)=0.25
        out.append(([label(x) for x in s], label(u), CPI(u)))
    return out
try:
    with Pool(2) as pool: cols = pool.map(grid_column, range(G))
except Exception:
    cols = [grid_column(i) for i in range(G)]
code = {'DL': 0, 'N+DL+H': 1, 'N+DL+D+H': 2, 'DL+D': 2}
reg = np.zeros((G, G)); bis = np.zeros((G, G), bool); Lu = np.zeros((G, G))
for i in range(G):
    for j in range(G):
        labs, lu, Lval = cols[i][j]; Lu[j, i] = Lval
        if len(labs) == 2: bis[j, i] = True; reg[j, i] = 2
        else: reg[j, i] = code.get(labs[0] if labs else lu, 2)
yy = np.linspace(k0['kH'] + dH + 1e-4, 1, 300); xb = []           # boundary: the Depressed invade Q*
for y in yy:
    x = Qstar(rates(kDLH=y)); f = M0 @ x; E = f - f @ x
    xb.append((k0['kD'] + k0['kDH']*x[3] - E[2]) / x[1])
xb = np.array(xb); ok = xb <= 1
print('boundary: kDLD - kDLH between %.3f and %.3f' % ((xb-yy)[ok].min(), (xb-yy)[ok].max()))
print('multistability wedge at kDLH=0: kDLD from %.2f to %.2f' % (xs[bis[0]].min(), xs[bis[0]].max()))
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6))
ax = axs[0]
ax.pcolormesh(xs, ys, reg, cmap=ListedColormap(['#d9dddd', '#f6e7a8', '#edc3bd']), vmin=-0.5, vmax=2.5,
              shading='nearest', rasterized=True)
ax.contourf(xs, ys, bis.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
ax.contour(xs, ys, bis.astype(float), levels=[0.5], colors='0.3', linewidths=0.6)
xD0, yH0 = k0['kD'] + dD, k0['kH'] + dH                              # invasion thresholds of the all-DL state
ax.plot([xD0, xD0], [0, yH0], color='0.15', lw=1.0, ls='--'); ax.plot([0, xD0], [yH0, yH0], color='0.15', lw=1.0, ls='--')
ax.plot(xb[ok], yy[ok], color='0.15', lw=1.0)
ax.text(0.26, 0.12, 'all\nDisillusioned', ha='center', va='center', fontsize=7.5)
ax.text(0.40, 0.66, 'no Depressed\n(N, DL, H)', ha='center', va='center', fontsize=7.5)
ax.text(0.82, 0.12, 'four types\ncoexist', ha='center', va='center', fontsize=7.5)
ax.set_ylabel(r'Testimony rate $\kappa_{DL,H}$'); ax.set_title('Long-run regime', loc='left', fontsize=9)
ax = axs[1]
pc = ax.pcolormesh(xs, ys, Lu, cmap='Greys_r', vmin=0, vmax=1.3, shading='nearest', rasterized=True)
ax.contour(xs, ys, np.where((reg == 0) | bis, np.nan, Lu), levels=[1.0], colors='k', linewidths=1.0)
ax.contourf(xs, ys, bis.astype(float), levels=[0.5, 1.5], colors='none', hatches=['////'])
fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03).set_label('CPI $L$')
ax.set_title('CPI $L$', loc='left', fontsize=9); ax.set_yticklabels([])
for a_ in axs:
    a_.plot([0.65], [0.30], 'o', ms=5, mfc='white', mec='k', mew=1.1)
    a_.annotate('baseline', (0.65, 0.30), xytext=(0.72, 0.36), fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3'))
    a_.set_xlabel(r'Contagion rate $\kappa_{DL,D}$'); a_.set_xlim(0, 1); a_.set_ylim(0, 1); a_.set_aspect('equal')
    for s in ('top', 'right'): a_.spines[s].set_visible(True)
fig.legend([Line2D([], [], color='0.15', ls='--', lw=1.0), Line2D([], [], color='0.15', lw=1.0),
            Patch(fc='white', ec='0.3', hatch='////'), Line2D([], [], color='k', lw=1.0)],
           ['invasion thresholds of the all-DL state', 'invasion threshold of the Depressed',
            'two stable equilibria', '$L=1$ (right panel)'], loc='lower center', ncol=2, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.09, right=0.93, top=0.93, bottom=0.26, wspace=0.12)
save(fig, 'Figure_5')

# %% [markdown]
# ## Download (Google Colab)

# %%
names = ['Figure_5', 'Figure_6', 'Figure_7']
with zipfile.ZipFile('figures_5-7.zip', 'w') as z:
    for n in names: z.write(n + '.pdf'); z.write(n + '.png')
try:
    from google.colab import files; files.download('figures_5-7.zip')
except ImportError:
    print('figures_5-7.zip written to the current folder')
