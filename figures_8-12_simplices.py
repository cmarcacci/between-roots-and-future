# %% [markdown]
# # Between Roots and Future: Figures 8–12 (ternary plots of the faces of the simplex, Section 4.4)
# * Faces $x_D=0$ and $x_H=0$ are invariant: Figures 8 and 10 are exact phase portraits (single colour).
# * Faces $x_N=0$ and $x_{DL}=0$ are not invariant: Figures 9 and 11 show the projection $\pi_h$ of four-type
#   trajectories (composition among the three displayed types); the colour reports the share of the hidden type.
# * Figure 12 places the four faces in a single diagram.
#
# Trajectories are integrated with `odeint` on $t\in[0,600]$ at the baseline (Scenario D), $w=1$.
# Requires the package `python-ternary`.

# %%
# On Google Colab, uncomment the next line:
# !pip install python-ternary

# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import ternary
import ternary.helpers as th
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

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

W = 1.0                     # selection intensity
K = rates()                 # baseline conversion rates (Scenario D)

def trajectory(x0, t):
    return odeint(rhs, x0, t, args=(M0, W, K), tfirst=True)

# %% [markdown]
# ## Settings, equilibria and initial conditions

# %%
t = np.linspace(0, 600, 3000)
THRESHOLD = 1e-5            # below this sum of the displayed shares the projection is not defined (singular point)

# Equilibria of the full system (N, DL, D, H), computed by long integration
t_long = np.linspace(0, 5000, 5000)
EQUILIBRIA = {
    r"$P_1$": np.array([0.0, 1.0, 0.0, 0.0]),
    r"$P_2$": trajectory([0.3, 0.4, 0.0, 0.3], t_long)[-1],
    r"$P_3$": trajectory([0.0, 0.5, 0.5, 0.0], t_long)[-1],
    r"$E^*$": trajectory([0.25, 0.25, 0.25, 0.25], t_long)[-1],
}
print("Equilibria (N, DL, D, H):")
for name, x in EQUILIBRIA.items():
    print("  ", name, np.round(x, 3))

# Case 1: face x_D = 0 (invariant). No initial condition with Depressed.
IC_CASE1 = [
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00],

    [0.10, 0.00, 0.00, 0.90],
    [0.30, 0.00, 0.00, 0.70],
    [0.50, 0.00, 0.00, 0.50],
    [0.70, 0.00, 0.00, 0.30],
    [0.90, 0.00, 0.00, 0.10],

    [0.00, 0.10, 0.00, 0.90],
    [0.00, 0.30, 0.00, 0.70],
    [0.00, 0.50, 0.00, 0.50],
    [0.00, 0.70, 0.00, 0.30],
    [0.00, 0.90, 0.00, 0.10],

    [0.90, 0.10, 0.00, 0.00],
    [0.70, 0.30, 0.00, 0.00],
    [0.50, 0.50, 0.00, 0.00],
    [0.30, 0.70, 0.00, 0.00],
    [0.10, 0.90, 0.00, 0.00],
]

# Case 2: initial conditions without Disillusioned (face x_DL = 0, NOT invariant).
IC_CASE2 = [
    [0.00, 0.00, 0.00, 1.00],
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00],

    [0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.70, 0.30],
    [0.00, 0.00, 0.50, 0.50],
    [0.00, 0.00, 0.30, 0.70],
    [0.00, 0.00, 0.10, 0.90],

    [0.90, 0.00, 0.10, 0.00],

    [0.10, 0.00, 0.00, 0.90],
    [0.30, 0.00, 0.00, 0.70],
    [0.50, 0.00, 0.00, 0.50],
    [0.70, 0.00, 0.00, 0.30],
    [0.90, 0.00, 0.00, 0.10],
]

# Case 3: face x_H = 0 (invariant). No initial condition with Hopeful.
IC_CASE3 = [
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00],

    [0.00, 0.10, 0.90, 0.00],
    [0.00, 0.30, 0.70, 0.00],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.70, 0.30, 0.00],
    [0.00, 0.90, 0.10, 0.00],

    [0.10, 0.90, 0.00, 0.00],
    [0.30, 0.70, 0.00, 0.00],
    [0.50, 0.50, 0.00, 0.00],
    [0.70, 0.30, 0.00, 0.00],
    [0.90, 0.10, 0.00, 0.00],

    [0.90, 0.00, 0.10, 0.00],
    [0.70, 0.00, 0.30, 0.00],
    [0.50, 0.00, 0.50, 0.00],
    [0.30, 0.00, 0.70, 0.00],
    [0.10, 0.00, 0.90, 0.00],
]

# Case 4: initial conditions without Nostalgic (face x_N = 0, NOT invariant).
# The condition [1, 0, 0, 0] is omitted: it is the hidden vertex, where the projection is not defined.
IC_CASE4 = [
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 1.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00],

    [0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.70, 0.30],
    [0.00, 0.00, 0.50, 0.50],
    [0.00, 0.00, 0.30, 0.70],
    [0.00, 0.00, 0.10, 0.90],

    [0.00, 0.90, 0.10, 0.00],
    [0.00, 0.70, 0.30, 0.00],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.30, 0.70, 0.00],
    [0.00, 0.10, 0.90, 0.00],

    [0.00, 0.10, 0.00, 0.90],
    [0.00, 0.30, 0.00, 0.70],
    [0.00, 0.50, 0.00, 0.50],
    [0.00, 0.70, 0.00, 0.30],
    [0.00, 0.90, 0.00, 0.10],
]

INVARIANT_FACES = (2, 3)    # hidden type D or H: the face is invariant

# %% [markdown]
# ## Projection and plotting functions

# %%
def project(x, indices):
    """Projection pi_h: shares of the three displayed types divided by their sum.
    Returns None at the singular point (zero sum: the whole community is of the hidden type)."""
    v = np.array([x[indices[0]], x[indices[1]], x[indices[2]]])
    total = v.sum()
    if total <= THRESHOLD:
        return None
    return tuple(v / total)

def projected_trajectory(x0, indices, hidden):
    """Projected points, share of the hidden type, final four-component state, and truncation flag."""
    sol = trajectory(x0, t)
    points, hidden_shares, truncated = [], [], False
    for x in sol:
        p = project(x, indices)
        if p is None:
            truncated = True
            break
        points.append(p)
        hidden_shares.append(x[hidden])
    return points, hidden_shares, sol[-1], truncated

def label_equilibria(ax, final_states, indices, transform=None, **kwargs):
    """Writes P_1, P_2, P_3, E* next to the end points, where the projection is defined."""
    for name, eq in EQUILIBRIA.items():
        reached = any(np.abs(xf - eq).max() < 1e-2 for xf in final_states)
        p = project(eq, indices)
        if reached and p is not None:
            if transform is not None:
                p = transform(p)
            x, y = th.project_point(p)
            offset = (10, -10) if (name == r"$P_3$" and indices == [3, 1, 2]) else (7, 7)
            ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=11, **kwargs)

def ternary_plot(title, labels, indices, initial_conditions, filename):
    fig, tax = ternary.figure(scale=1.0)
    fig.set_size_inches(9, 6)
    tax.boundary(linewidth=1.0, alpha=0.7)
    tax.gridlines(multiple=0.1, color="blue", alpha=0.2)
    tax.set_title(title, fontsize=14, pad=20)
    tax.bottom_axis_label(labels[0], offset=0.01)
    tax.right_axis_label(labels[1], offset=0.14)
    tax.left_axis_label(labels[2], offset=0.14)

    ax = tax.get_axes()

    hidden = [i for i in range(4) if i not in indices][0]
    hidden_name = NAMES[hidden]
    invariant = hidden in INVARIANT_FACES

    # The hidden share can reach 1: colour scale from 0 to 1
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(0, 1)

    final_states = []
    has_crosses = False
    for x0 in initial_conditions:
        if project(x0, indices) is None:
            print("Skipped", x0, ": it is the hidden vertex, where the projection is not defined")
            continue
        points, hidden_shares, x_final, truncated = projected_trajectory(x0, indices, hidden)
        final_states.append(x_final)

        if len(points) > 1:
            points_2d = [th.project_point(p) for p in points]
            segments = [[points_2d[i], points_2d[i+1]] for i in range(len(points_2d)-1)]
            if invariant:
                # invariant face: the fourth type stays at zero, single colour
                lc = LineCollection(segments, colors='#2c3e50')
            else:
                segment_colors = [(hidden_shares[i] + hidden_shares[i+1])/2 for i in range(len(hidden_shares)-1)]
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(np.array(segment_colors))
            lc.set_linewidth(2.5)
            lc.set_alpha(0.8)
            ax.add_collection(lc)

        # Start (white circle); end (black triangle) or exit from the domain (red cross)
        tax.scatter([points[0]], marker='o', color='white', edgecolor='black', s=30, zorder=5)
        if truncated:
            tax.scatter([points[-1]], marker='x', color='red', s=70, linewidths=2, zorder=6)
            has_crosses = True
        else:
            tax.scatter([points[-1]], marker='^', color='black', s=60, zorder=5)

    label_equilibria(ax, final_states, indices)

    tax.ticks(axis='lbr', multiple=0.1, linewidth=1, tick_formats="%.1f")
    tax.get_axes().axis('off')

    handles = [Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=6, label='initial condition'),
               Line2D([], [], marker='^', color='w', markerfacecolor='black', markersize=8, label='equilibrium reached')]
    if has_crosses:
        handles.append(Line2D([], [], marker='x', color='w', markeredgecolor='red', markersize=8, markeredgewidth=2,
                              label='trajectory tends to the hidden vertex'))
    if not invariant:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(f'Fraction of {hidden_name}', fontsize=12)
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Figures 8–11: the four faces

# %%
# Figure 8 - Case 1: invariant face x_D = 0
ternary_plot("", ["Disillusioned (DL)", "Hopeful (H)", "Nostalgic (N)"], [1, 3, 0],
             IC_CASE1, 'Figure_8.pdf')

# Figure 9 - Case 2: projection pi_DL (trajectories starting without Disillusioned)
ternary_plot("", ["Depressed (D)", "Nostalgic (N)", "Hopeful (H)"], [2, 0, 3],
             IC_CASE2, 'Figure_9.pdf')

# Figure 10 - Case 3: invariant face x_H = 0
ternary_plot("", ["Depressed (D)", "Nostalgic (N)", "Disillusioned (DL)"], [2, 0, 1],
             IC_CASE3, 'Figure_10.pdf')

# Figure 11 - Case 4: projection pi_N (trajectories starting without Nostalgic)
ternary_plot("", ["Hopeful (H)", "Disillusioned (DL)", "Depressed (D)"], [3, 1, 2],
             IC_CASE4, 'Figure_11.pdf')

# %% [markdown]
# ## Figure 12: the four faces of the tetrahedron in one diagram

# %%
def map_to_sector(p, vertices):
    """Maps a point of the triangle (barycentric coordinates) to the sector with the given vertices."""
    P = [0.0, 0.0, 0.0]
    for i in range(3):
        P[0] += p[i] * vertices[i][0]
        P[1] += p[i] * vertices[i][1]
        P[2] += p[i] * vertices[i][2]
    return tuple(P)

V_TOP    = (0.0, 1.0, 0.0)
V_RIGHT  = (1.0, 0.0, 0.0)
V_LEFT   = (0.0, 0.0, 1.0)
M_RIGHT  = (0.5, 0.5, 0.0)
M_LEFT   = (0.0, 0.5, 0.5)
M_BOTTOM = (0.5, 0.0, 0.5)

fig_full, tax = ternary.figure(scale=1.0)
fig_full.set_size_inches(12, 10)
tax.boundary(linewidth=2.0, alpha=0.5)
tax.gridlines(multiple=0.05, color="black", alpha=0.2, linewidth=1.0)
tax.line(M_RIGHT, M_BOTTOM, linewidth=1.5, color='black', linestyle='--')
tax.line(M_BOTTOM, M_LEFT, linewidth=1.5, color='black', linestyle='--')
tax.line(M_LEFT, M_RIGHT, linewidth=1.5, color='black', linestyle='--')

# Indices: 0=N, 1=DL, 2=D, 3=H
SECTORS = [
    {'idx': [3, 1, 0], 'hidden': 2, 'vertices': [V_TOP,   M_RIGHT, M_LEFT],   'ics': IC_CASE1},   # D absent (invariant)
    {'idx': [3, 1, 2], 'hidden': 0, 'vertices': [V_RIGHT, M_RIGHT, M_BOTTOM], 'ics': IC_CASE4},   # N absent
    {'idx': [3, 0, 2], 'hidden': 1, 'vertices': [V_LEFT,  M_LEFT,  M_BOTTOM], 'ics': IC_CASE2},   # DL absent
    {'idx': [1, 0, 2], 'hidden': 3, 'vertices': [M_RIGHT, M_LEFT,  M_BOTTOM], 'ics': IC_CASE3},   # H absent (invariant)
]

ax_full = tax.get_axes()
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(0, 1)

LABEL_OFFSETS = {r"$P_1$": (-24, 6), r"$P_3$": (10, -4)}   # label offsets (in points); default (7, 7)

def label_equilibria_sector(final_states, s, written):
    """As label_equilibria, but in sector s of the full diagram.
    A label already written near the same point (same name, distance < 0.06) is not repeated."""
    for name, eq in EQUILIBRIA.items():
        reached = any(np.abs(xf - eq).max() < 1e-2 for xf in final_states)
        p = project(eq, s['idx'])
        if reached and p is not None:
            x, y = th.project_point(map_to_sector(p, s['vertices']))
            if any(n == name and np.hypot(x - u, y - v) < 0.06 for n, u, v in written):
                continue
            written.append((name, x, y))
            ax_full.annotate(name, (x, y), xytext=LABEL_OFFSETS.get(name, (7, 7)),
                             textcoords="offset points", fontsize=11, zorder=7)

written = []
for s in SECTORS:
    final_states = []
    for x0 in s['ics']:
        if project(x0, s['idx']) is None:
            continue
        points, hidden_shares, x_final, truncated = projected_trajectory(x0, s['idx'], s['hidden'])
        if truncated:
            continue   # the same trajectory is shown in full in the adjacent sector
        final_states.append(x_final)
        sector_points = [map_to_sector(p, s['vertices']) for p in points]

        if len(sector_points) > 1:
            points_2d = [th.project_point(p) for p in sector_points]
            segments = [[points_2d[i], points_2d[i+1]] for i in range(len(points_2d)-1)]
            segment_colors = [(hidden_shares[i] + hidden_shares[i+1])/2 for i in range(len(hidden_shares)-1)]
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(segment_colors))
            lc.set_linewidth(2.0)
            lc.set_alpha(0.8)
            ax_full.add_collection(lc)

        tax.scatter([sector_points[0]], marker='o', color='white', edgecolor='black', s=30, zorder=5)
        tax.scatter([sector_points[-1]], marker='^', color='black', s=60, zorder=5)

    label_equilibria_sector(final_states, s, written)

def put_label(point, text, **kwargs):
    x, y = th.project_point(point)
    tax.get_axes().text(x, y, text, **kwargs)

put_label((-0.02,  1.03,  0.00), "H", fontsize=14, ha='center', fontweight='bold', color='#1a5276')
put_label(( 1.05, -0.02,  0.00), "H", fontsize=14, ha='center', fontweight='bold', color='#1a5276')
put_label((-0.02, -0.02,  1.05), "H", fontsize=14, ha='center', fontweight='bold', color='#1a5276')
put_label((0.53,  0.50,  0.00), "DL", fontsize=12, fontweight='bold', color='darkred')
put_label((-0.05, 0.51,  0.53), "N",  fontsize=12, fontweight='bold', color='darkred')
put_label((0.53, -0.06,  0.50), "D",  fontsize=12, fontweight='bold', color='darkred')

tax.get_axes().axis('off')
tax.set_title("", fontsize=15, pad=20)

handles = [
    Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=6, label='initial condition'),
    Line2D([], [], marker='^', color='w', markerfacecolor='black', markersize=8, label='equilibrium reached')
]
ax_full.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_full, fraction=0.03, pad=0.04)
cbar.set_label('Fraction of the type absent from the face at $t=0$', fontsize=12)

plt.tight_layout()
plt.savefig('Figure_12.pdf', format='pdf', bbox_inches='tight')
plt.show()
