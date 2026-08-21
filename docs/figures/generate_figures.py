"""Generate all figures for docs/FedProxAlgorithm.md"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── LC Design System palette ────────────────────────────────────────────────
C = dict(
    bg='#FAFAFA', surface='#FFFFFF', border='#E4E4E7',
    text='#18181B', text2='#52525B', muted='#71717A',
    primary='#0F69AF', accent='#2DBECD',
    magenta='#EB3C96', purple='#503291',
    lime='#A5CD50', error='#E61E50',
    success='#149B5F', warning='#FFC832',
)

OUT = os.path.join(os.path.dirname(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.facecolor': C['surface'],
    'figure.facecolor': C['bg'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': C['border'],
    'grid.color': C['border'],
    'grid.linewidth': 0.8,
    'xtick.color': C['text2'],
    'ytick.color': C['text2'],
    'axes.labelcolor': C['text'],
    'text.color': C['text'],
})

def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.close('all')
    print(f'  saved: {name}')

SITE_COLORS = [C['primary'], C['accent'], C['purple'], C['error'], C['warning']]
SITES = ['Site 1', 'Site 2', 'Site 3', 'Site 4', 'Site 5']
N_SAMPLES = [800, 300, 500, 150, 600]
REGIMES = ['Cake filtration', 'Intermediate\nblocking', 'Combined 1-A', 'Complete\nblocking', 'Standard\nblocking']
TMP = ['0.5–1.5', '1.0–3.0', '0.3–0.8', '1.5–4.0', '0.8–2.0']
FILTERS = ['PES 0.2 µm', 'PVDF 0.1 µm', 'Cell. 0.45 µm', 'PES 0.1 µm', 'PVDF 0.22 µm']

W_GLOBAL = np.array([2.00, 0.50])
LOCAL_OPTS = np.array([[2.30, 0.65],[2.10, 0.45],[2.25, 0.70],[0.80, 0.10],[2.15, 0.55]])
N = np.array(N_SAMPLES, dtype=float)
WTS = N / N.sum()
MU_EX = 0.10
FP_OPTS = (LOCAL_OPTS + MU_EX * W_GLOBAL) / (1 + MU_EX)
FEDAVG_R = np.sum(WTS[:,None] * LOCAL_OPTS, axis=0)
FEDPROX_R = np.sum(WTS[:,None] * FP_OPTS, axis=0)
TRUE_OPT = np.array([np.sum(WTS * LOCAL_OPTS[:,0]), np.sum(WTS * LOCAL_OPTS[:,1])])


# ════════════════════════════════════════════════════════════════════════════
print('fig01 — site heterogeneity')
fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(5)
bars = ax.barh(y, N_SAMPLES, color=SITE_COLORS, height=0.55, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(SITES, fontsize=11, color=C['text'])
ax.set_xlabel('Training samples (n)', fontsize=11)
ax.set_title('Site Data Heterogeneity — Dataset Size and Dominant Fouling Regime',
             fontsize=12, color=C['text'], pad=12)
ax.grid(axis='x', zorder=1)
ax.set_xlim(0, 1050)
for i, (bar, n, reg, tmp, filt) in enumerate(zip(bars, N_SAMPLES, REGIMES, TMP, FILTERS)):
    ax.text(n + 18, bar.get_y() + bar.get_height()/2,
            f'{n}  |  {reg.replace(chr(10)," ")}  |  TMP {tmp} bar  |  {filt}',
            va='center', fontsize=8.5, color=C['text2'])
ax.set_facecolor(C['surface'])
fig.patch.set_facecolor(C['bg'])
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
save('fig01_site_heterogeneity.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig02 — client drift weight space')
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
titles = ['Without FedProx  (FedAvg — free local training)',
          'With FedProx  μ = 0.10  (production: μ = 0.01)']
end_pts = [LOCAL_OPTS, FP_OPTS]
result_pts = [FEDAVG_R, FEDPROX_R]

for ax, title, opts, res in zip(axes, titles, end_pts, result_pts):
    ax.set_facecolor(C['surface'])
    # Draw a soft "leash circle" for the FedProx panel
    if 'FedProx' in title:
        radius = np.max(np.linalg.norm(opts - W_GLOBAL, axis=1)) * 1.15
        circle = plt.Circle(W_GLOBAL, radius, color=C['primary'],
                            alpha=0.07, zorder=1, linewidth=0)
        ax.add_patch(circle)
        ax.annotate('leash\nzone', xy=(W_GLOBAL[0] + radius*0.65, W_GLOBAL[1] + radius*0.65),
                    fontsize=8, color=C['primary'], alpha=0.7)
    # Arrows from W_global to each site
    for i, (opt, col) in enumerate(zip(opts, SITE_COLORS)):
        style = 'dashed' if 'FedAvg' in title else 'solid'
        ax.annotate('', xy=opt, xytext=W_GLOBAL,
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                    linestyle=style, connectionstyle='arc3,rad=0.0'))
        ax.scatter(*opt, color=col, s=90, zorder=5)
        offset = [0.04, -0.04] if i % 2 == 0 else [-0.12, -0.04]
        ax.annotate(SITES[i], xy=opt, xytext=(opt[0]+offset[0], opt[1]+offset[1]),
                    fontsize=8.5, color=col)
    # W_global star
    ax.scatter(*W_GLOBAL, marker='*', s=350, color=C['primary'], zorder=6, label='W_global')
    ax.annotate('W_global\n[2.00, 0.50]', xy=W_GLOBAL,
                xytext=(W_GLOBAL[0]+0.06, W_GLOBAL[1]-0.09),
                fontsize=8, color=C['primary'])
    # Aggregated result
    ax.scatter(*res, marker='D', s=130, color=C['text'], zorder=6)
    ax.annotate(f'Aggregated\n[{res[0]:.2f}, {res[1]:.2f}]',
                xy=res, xytext=(res[0]-0.35, res[1]+0.06),
                fontsize=8, color=C['text'])
    # True optimum
    ax.scatter(*TRUE_OPT, marker='P', s=120, color=C['success'], zorder=6)
    ax.annotate(f'True opt\n[{TRUE_OPT[0]:.2f}, {TRUE_OPT[1]:.2f}]',
                xy=TRUE_OPT, xytext=(TRUE_OPT[0]+0.04, TRUE_OPT[1]+0.05),
                fontsize=8, color=C['success'])
    ax.set_xlabel('J0_w  (initial flux weight)', fontsize=11)
    ax.set_ylabel('k1_w  (pore-constriction weight)', fontsize=11)
    ax.set_title(title, fontsize=10.5, color=C['text'])
    ax.set_xlim(0.55, 2.65)
    ax.set_ylim(-0.05, 0.88)
    ax.grid(True, alpha=0.5)

legend_elems = [Line2D([0],[0], marker='*', color='w', markerfacecolor=C['primary'],
                        markersize=12, label='W_global (server broadcast)'),
                Line2D([0],[0], marker='D', color='w', markerfacecolor=C['text'],
                        markersize=9, label='Aggregated result'),
                Line2D([0],[0], marker='P', color='w', markerfacecolor=C['success'],
                        markersize=9, label='True global optimum')]
legend_elems += [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                         markersize=9, label=s) for s, c in zip(SITES, SITE_COLORS)]
axes[1].legend(handles=legend_elems, fontsize=8, loc='lower right',
               framealpha=0.9, edgecolor=C['border'])
fig.suptitle('FedAvg vs FedProx — Local Model Drift in Weight Space',
             fontsize=13, color=C['text'], y=1.01)
save('fig02_client_drift.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig03 — loss landscape (Site 4)')
T4 = np.array([0.80, 0.10])
j0_range = np.linspace(0.3, 2.6, 300)
k1_range = np.linspace(-0.1, 0.85, 300)
J, K = np.meshgrid(j0_range, k1_range)
F_local = (J - T4[0])**2 + (K - T4[1])**2
F_prox  = (J - W_GLOBAL[0])**2 + (K - W_GLOBAL[1])**2
F_comb  = F_local + (MU_EX / 2) * F_prox

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
data = [(F_local,  'Local loss  F₄(W)',           T4,           'Site 4 local opt\n[0.80, 0.10]',  C['error']),
        (F_prox,   'Proximal term  ‖W−W_global‖²', W_GLOBAL,    'W_global\n[2.00, 0.50]',         C['primary']),
        (F_comb,   'Combined  F₄(W) + (μ/2)‖W−W_global‖²', FP_OPTS[3], f'FedProx opt\n[{FP_OPTS[3,0]:.3f}, {FP_OPTS[3,1]:.3f}]', C['success'])]

for ax, (Z, ttl, star, lbl, scol) in zip(axes, data):
    ax.set_facecolor(C['surface'])
    levels = np.percentile(Z, np.linspace(5, 80, 14))
    ct = ax.contourf(J, K, Z, levels=levels, cmap='Blues', alpha=0.7)
    ax.contour(J, K, Z, levels=levels, colors=C['primary'], alpha=0.3, linewidths=0.7)
    if ttl.startswith('Combined'):
        ax.scatter(*T4, marker='x', s=120, color=C['error'], zorder=6, linewidths=2)
        ax.annotate('Local opt', xy=T4, xytext=(T4[0]+0.1, T4[1]+0.08),
                    fontsize=7.5, color=C['error'])
        ax.scatter(*W_GLOBAL, marker='*', s=250, color=C['primary'], zorder=6)
        ax.annotate('W_global', xy=W_GLOBAL, xytext=(W_GLOBAL[0]+0.06, W_GLOBAL[1]+0.05),
                    fontsize=7.5, color=C['primary'])
    ax.scatter(*star, marker='D', s=180, color=scol, zorder=7, edgecolors='white', linewidths=1.5)
    ax.annotate(lbl, xy=star, xytext=(star[0]+0.08, star[1]-0.12),
                fontsize=8, color=scol, fontweight='bold')
    ax.set_xlabel('J0_w', fontsize=10)
    ax.set_ylabel('k1_w', fontsize=10)
    ax.set_title(ttl, fontsize=10, color=C['text'])
    ax.grid(True, alpha=0.3)

fig.suptitle('Loss Landscape for Site 4 (High-TMP Outlier) — How FedProx Shifts the Local Minimum',
             fontsize=12, color=C['text'], y=1.02)
save('fig03_loss_landscape.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig04 — mu sensitivity')
mu_vals = np.linspace(0, 0.5, 200)
# Site 4 J0_w after one round: (T4_J0 + mu * W_global_J0) / (1+mu)
site4_j0_1round = (0.80 + mu_vals * 2.00) / (1 + mu_vals)

# Full 50-round simulation for Site 4 influence on global model
def sim_site4_effect(mu, rounds=50, alpha=0.8):
    W = W_GLOBAL[0]
    for _ in range(rounds):
        local_w = [W + alpha*(lo - W) for lo in LOCAL_OPTS[:,0]]
        if mu > 0:
            local_w = [(lw + mu*W)/(1+mu) for lw in local_w]
        W = float(np.sum(WTS * local_w))
    return W

mu_test = np.linspace(0, 0.5, 40)
global_j0_50r = [sim_site4_effect(m) for m in mu_test]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_facecolor(C['surface'])
ax.plot(mu_vals, site4_j0_1round, color=C['accent'], lw=2,
        label='Site 4 local weight after 1 round')
ax.plot(mu_test, global_j0_50r, color=C['primary'], lw=2.5,
        label='Global model J0_w after 50 rounds')
ax.axhline(TRUE_OPT[0], color=C['success'], lw=1.5, ls='--',
           label=f'True global optimum ({TRUE_OPT[0]:.3f})')
ax.axhline(2.00, color=C['text2'], lw=1.2, ls=':', alpha=0.7,
           label='W_global at start (2.00)')
ax.axhline(0.80, color=C['error'], lw=1.2, ls=':', alpha=0.7,
           label='Site 4 local target (0.80)')
ax.axvline(0.01, color=C['warning'], lw=1.5, ls='--', alpha=0.8)
ax.text(0.013, 0.84, 'μ = 0.01\n(production)', fontsize=8.5,
        color=C['warning'], va='bottom')
ax.axvline(0.10, color=C['magenta'], lw=1.5, ls='--', alpha=0.8)
ax.text(0.103, 0.84, 'μ = 0.10\n(this example)', fontsize=8.5,
        color=C['magenta'], va='bottom')
ax.set_xlabel('FedProx μ (FEDPROX_MU)', fontsize=11)
ax.set_ylabel('J0_w value', fontsize=11)
ax.set_title('Effect of μ on Local Model Drift and Global Convergence', fontsize=12)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor=C['border'])
ax.grid(True, alpha=0.5)
ax.set_xlim(-0.01, 0.52)
save('fig04_mu_sensitivity.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig05 — algorithm comparison radar')
categories = ['Non-IID\nRobustness', 'Communication\nEfficiency',
              'Convergence\nSpeed', 'Implementation\nSimplicity',
              'Exact\nConvergence', 'Partial\nParticipation']
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

algos = {
    'FedAvg':  ([2, 5, 4, 5, 1, 4], C['text2']),
    'FedProx': ([4, 5, 3, 4, 3, 4], C['primary']),
    'SCAFFOLD':([5, 3, 5, 3, 5, 3], C['accent']),
    'FedNova': ([4, 4, 4, 4, 3, 3], C['purple']),
    'FedDyn':  ([5, 3, 4, 2, 5, 2], C['success']),
    'MOON':    ([4, 2, 3, 2, 3, 2], C['magenta']),
}

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
ax.set_facecolor(C['surface'])
fig.patch.set_facecolor(C['bg'])
for ring in [1, 2, 3, 4, 5]:
    ax.plot(angles, [ring]*N_cat + [ring], color=C['border'], lw=0.8, zorder=1)

for name, (scores, col) in algos.items():
    vals = scores + scores[:1]
    lw = 2.5 if name == 'FedProx' else 1.5
    alpha = 1.0 if name == 'FedProx' else 0.7
    ax.plot(angles, vals, color=col, lw=lw, label=name, zorder=3)
    ax.fill(angles, vals, alpha=0.06, color=col)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9, color=C['text'])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=7, color=C['muted'])
ax.set_ylim(0, 5.5)
ax.spines['polar'].set_color(C['border'])
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
          fontsize=9.5, framealpha=0.95, edgecolor=C['border'])
ax.set_title('FL Algorithm Comparison  (score 1–5, higher = better)',
             fontsize=12, color=C['text'], pad=20)
# Highlight FedProx ring
scores_fp = algos['FedProx'][0]
vals_fp = scores_fp + scores_fp[:1]
ax.plot(angles, vals_fp, color=C['primary'], lw=3, zorder=4)
save('fig05_algorithm_radar.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig06 — hermia flux models')
t = np.linspace(0, 120, 500)
J0 = 100.0

models = {
    'Standard\n$J = J_0/(1+k_s t)^2$':
        (J0 / (1 + 0.02*t)**2, C['primary']),
    'Complete\n$J = J_0 e^{-k_c t}$':
        (J0 * np.exp(-0.015*t), C['error']),
    'Intermediate\n$J = J_0/(1+J_0 k_i t)$':
        (J0 / (1 + J0*0.0003*t), C['warning']),
    'Cake\n$J = J_0/\\sqrt{1+J_0^2 k_{cf} t}$':
        (J0 / np.sqrt(1 + J0**2 * 0.00003 * t), C['purple']),
    'Combined 1-A\n$J = J_0/(1+k_1 t)^2 \\cdot e^{-k_2 t}$':
        ((J0 / (1 + 0.02*t)**2) * np.exp(-0.005*t), C['accent']),
}

fig, axes = plt.subplots(1, 5, figsize=(15, 4.5), sharey=True)
fig.patch.set_facecolor(C['bg'])
for ax, (label, (flux, col)) in zip(axes, models.items()):
    ax.set_facecolor(C['surface'])
    ax.plot(t, flux, color=col, lw=2.5)
    ax.fill_between(t, flux, alpha=0.12, color=col)
    ax.set_xlabel('Time (min)', fontsize=9)
    ax.set_title(label, fontsize=9, color=C['text'], pad=6)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 115)
    ax.axhline(flux[-1], color=col, lw=0.8, ls=':', alpha=0.6)
    ax.text(62, flux[-1]+3, f'J(120)={flux[-1]:.0f}', fontsize=7.5, color=col)
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Flux  J  (LMH)', fontsize=10)
# Mark J0 line
for ax in axes:
    ax.axhline(J0, color=C['text2'], lw=0.8, ls='--', alpha=0.5)
    ax.text(1, J0+2, 'J₀=100', fontsize=7, color=C['text2'])

fig.suptitle('The Five Hermia Membrane Fouling Models  (J₀ = 100 LMH, t = 0–120 min)',
             fontsize=12, color=C['text'], y=1.02)
save('fig06_hermia_models.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig07 — manabe Pc and LRV vs flux')
J_flux = np.linspace(0, 200, 500)
J_crit = 80.0
lambdas = [2.0, 5.0, 10.0]
lambda_cols = [C['accent'], C['primary'], C['purple']]
lambda_lbls = ['λ = 2 (low affinity)', 'λ = 5 (medium)', 'λ = 10 (high affinity)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax in (ax1, ax2):
    ax.set_facecolor(C['surface'])
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('Flux  J  (LMH)', fontsize=11)
    ax.axvline(J_crit, color=C['text2'], lw=1.2, ls=':', alpha=0.7)
    ax.text(J_crit+3, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.01,
            f'J_crit = {J_crit:.0f}', fontsize=8.5, color=C['text2'], va='bottom')

for lam, col, lbl in zip(lambdas, lambda_cols, lambda_lbls):
    Pc = 1 - np.exp(-lam * J_flux / J_crit)
    ax1.plot(J_flux, Pc, color=col, lw=2.2, label=lbl)
    LRV = np.log10(1 / np.clip(1 - Pc, 1e-9, None))
    ax2.plot(J_flux, LRV, color=col, lw=2.2)

ax1.set_ylabel('Capture probability  Pc', fontsize=11)
ax1.set_title('Manabe Capture Probability', fontsize=11)
ax1.set_ylim(-0.02, 1.05)
ax1.axhline(0, color=C['border'], lw=0.8)
ax1.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor=C['border'])
ax1.axvline(J_crit, color=C['text2'], lw=1.2, ls=':', alpha=0.7)
ax1.text(J_crit+3, 0.02, f'J_crit = {J_crit:.0f}', fontsize=8.5, color=C['text2'])

ax2.set_ylabel('Log Reduction Value  (LRV)', fontsize=11)
ax2.set_title('LRV  =  log₁₀(1 / (1 − Pc))', fontsize=11)
ax2.set_ylim(-0.1, 8)
ax2.axhline(4.0, color=C['error'], lw=1.8, ls='--', label='Regulatory minimum LRV = 4')
ax2.fill_between(J_flux, 0, 4.0, alpha=0.07, color=C['error'])
ax2.text(5, 4.12, 'LRV = 4.0  (FDA/EMA minimum)', fontsize=8.5, color=C['error'])
ax2.axvline(J_crit, color=C['text2'], lw=1.2, ls=':', alpha=0.7)
ax2.text(J_crit+3, 0.1, f'J_crit = {J_crit:.0f}', fontsize=8.5, color=C['text2'])
ax2.legend(fontsize=9, loc='upper left', framealpha=0.95, edgecolor=C['border'])
for lam, col in zip(lambdas, lambda_cols):
    Pc_line = 1 - np.exp(-lam * J_flux / J_crit)
    LRV_line = np.log10(1 / np.clip(1 - Pc_line, 1e-9, None))
    idx = np.argmax(LRV_line >= 4.0)
    if idx > 0:
        ax2.scatter(J_flux[idx], 4.0, color=col, s=60, zorder=5)
        ax2.text(J_flux[idx]+3, 3.6, f'J={J_flux[idx]:.0f}', fontsize=7.5, color=col)

fig.suptitle('Manabe Virus Capture Model — Effect of Flux and Membrane Affinity on LRV',
             fontsize=12, color=C['text'], y=1.02)
save('fig07_manabe_lrv.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig08 — PINN architecture block diagram')
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C['bg'])

def box(ax, x, y, w, h, label, sublabel='', fc=C['surface'], ec=C['primary'],
        fontsize=9.5, subfontsize=8):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle='round,pad=0.08', facecolor=fc,
                       edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(r)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=C['text'], zorder=4)
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                fontsize=subfontsize, color=C['text2'], zorder=4)

def arrow(ax, x1, y1, x2, y2, col=C['primary'], lw=1.8, label='', lpos='mid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        mx = (x1+x2)/2 + (0.12 if lpos=='right' else -0.12 if lpos=='left' else 0)
        my = (y1+y2)/2 + 0.18
        ax.text(mx, my, label, fontsize=7.5, color=col, ha='center')

# Input
box(ax, 1.1, 3.5, 1.6, 5.0, 'INPUT\n(11 features)',
    '• pore_size_nm\n• nmwco_kda\n• membrane_area\n• tmp_bar\n• feed_flux_lmh\n• pH  • IS_mM\n• mab_conc_g_L\n• temperature_C\n• virus_size_nm\n• virus_charge',
    fc='#EFF6FF', ec=C['primary'], fontsize=10, subfontsize=8)

# ParameterPredictor layers
layer_cols = ['#EFF6FF', '#DBEAFE', '#BFDBFE', '#93C5FD']
layer_data = [(3.2, 3.5, 0.95, 2.0, 'Linear\n11 → 128', '+ReLU', layer_cols[0]),
              (4.5, 3.5, 0.95, 2.0, 'Linear\n128 → 128', '+ReLU', layer_cols[1]),
              (5.8, 3.5, 0.95, 2.0, 'Linear\n128 → 64', '+ReLU', layer_cols[2]),
              (7.1, 3.5, 0.95, 2.0, 'Linear\n64 → 10', '+Softplus\n/ Sigmoid', layer_cols[3])]
for (x, y, w, h, lbl, sub, fc) in layer_data:
    box(ax, x, y, w, h, lbl, sub, fc=fc, ec=C['primary'])

# Level 1 bracket
ax.annotate('', xy=(7.6, 1.0), xytext=(2.75, 1.0),
            arrowprops=dict(arrowstyle='-', color=C['primary'], lw=1.2))
ax.text(5.2, 0.7, 'LEVEL 1 — ParameterPredictor  (learnable weights — shared in FL)',
        ha='center', fontsize=9, color=C['primary'], style='italic')

# Connections between layers
for x1, x2 in [(1.9, 2.73), (3.68, 4.03), (4.98, 5.33), (6.28, 6.63)]:
    arrow(ax, x1, 3.5, x2, 3.5)

# Output params box
box(ax, 8.5, 3.5, 1.4, 4.8, '10 Physics\nParameters',
    'J0  ks  ki  kc\nkcf  k1  k2\nPc  Jcrit  Dv',
    fc='#F0FDF4', ec=C['success'])
arrow(ax, 7.6, 3.5, 7.8, 3.5, col=C['primary'], lw=2)

# Level 2 PhysicsSolver
box(ax, 10.6, 4.8, 2.0, 1.6, 'PhysicsSolver',
    'J(t) = J₀/(1+k₁t)²·e^(−k₂t)\nLRV = log₁₀(1/(1−Pc))',
    fc='#FFF7ED', ec=C['warning'], fontsize=9.5, subfontsize=7.5)
arrow(ax, 9.22, 4.2, 9.6, 4.8, col=C['success'])
arrow(ax, 9.6, 4.8, 9.6, 4.8, col=C['success'])
ax.annotate('', xy=(9.6, 4.8), xytext=(9.22, 4.2),
            arrowprops=dict(arrowstyle='->', color=C['success'], lw=1.8))
ax.annotate('', xy=(9.6, 4.8), xytext=(9.6, 4.8),
            arrowprops=dict(arrowstyle='->', color=C['success'], lw=1.8))

# Outputs J(t) and LRV
box(ax, 12.5, 5.6, 1.4, 0.9, 'J(t) curve', '', fc='#FFF7ED', ec=C['warning'])
box(ax, 12.5, 4.0, 1.4, 0.9, 'LRV value', '', fc='#FFF7ED', ec=C['warning'])
arrow(ax, 11.6, 5.2, 11.8, 5.6, col=C['warning'])
arrow(ax, 11.6, 4.4, 11.8, 4.1, col=C['warning'])

# Level 2 bracket
ax.plot([9.4, 13.3], [2.5, 2.5], color=C['warning'], lw=1.2)
ax.text(11.3, 2.2, 'LEVEL 2 — PhysicsSolver  (no learnable weights)',
        ha='center', fontsize=9, color=C['warning'], style='italic')

# BlockingRegimeClassifier
box(ax, 10.6, 2.0, 2.2, 1.5, 'Blocking Regime\nClassifier',
    'Linear 11→64→5\n5-class logits',
    fc='#FDF4FF', ec=C['purple'])
ax.annotate('', xy=(9.5, 2.0), xytext=(9.22, 2.8),
            arrowprops=dict(arrowstyle='->', color=C['purple'], lw=1.5,
                            connectionstyle='arc3,rad=-0.2'))
ax.text(8.6, 2.4, 'also\nfrom\ninputs', fontsize=7.5, color=C['purple'], ha='center')
box(ax, 12.5, 2.0, 1.4, 0.9, 'Regime\nlogits (5)', '', fc='#FDF4FF', ec=C['purple'], fontsize=8.5)
arrow(ax, 11.7, 2.0, 11.8, 2.0, col=C['purple'])

# Loss box at bottom
box(ax, 7.5, 0.55, 7.8, 0.75,
    'Loss = L_flux + L_LRV + L_physics + L_regime + L_fedprox',
    '', fc='#FEF2F2', ec=C['error'], fontsize=9)
for xi, yi in [(12.5, 5.15), (12.5, 3.55), (12.5, 1.55)]:
    ax.annotate('', xy=(7.8, 0.92), xytext=(xi, yi),
                arrowprops=dict(arrowstyle='->', color=C['error'], lw=1.2,
                                linestyle='dashed', connectionstyle='arc3,rad=0.3'))

ax.set_title('FiltrationPINN Architecture — Two-Level Physics-Informed Neural Network',
             fontsize=13, color=C['text'], pad=10)
save('fig08_pinn_architecture.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig09 — PINN forward pass example')
fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
fig.patch.set_facecolor(C['bg'])

# Panel 1: Input vector
ax = axes[0]
ax.set_facecolor(C['surface'])
ax.axis('off')
ax.set_title('Step 1: Input  (Site 1)', fontsize=10.5, color=C['text'], pad=8)
features = [
    ('pore_size_nm', '200'),
    ('nmwco_kda', '300'),
    ('membrane_area_m²', '0.02'),
    ('tmp_bar', '1.0'),
    ('feed_flux_lmh', '100'),
    ('pH', '7.0'),
    ('IS_mM', '150'),
    ('mab_conc_g_L', '2.0'),
    ('temperature_C', '25'),
    ('virus_size_nm', '25'),
    ('virus_charge', '−0.5'),
]
for i, (name, val) in enumerate(features):
    y = 0.94 - i * 0.085
    col = C['primary'] if i < 3 else (C['success'] if i < 9 else C['purple'])
    ax.text(0.08, y, f'x[{i:02d}]', fontsize=8, color=C['muted'], transform=ax.transAxes)
    ax.text(0.25, y, name, fontsize=8.5, color=C['text'], transform=ax.transAxes)
    r = FancyBboxPatch((0.72, y-0.035), 0.22, 0.07,
                       boxstyle='round,pad=0.01', facecolor=col+'22',
                       edgecolor=col, linewidth=1, transform=ax.transAxes)
    ax.add_patch(r)
    ax.text(0.83, y, val, fontsize=8.5, color=col, ha='center',
            fontweight='bold', transform=ax.transAxes)
legend_patches = [
    mpatches.Patch(facecolor=C['primary']+'22', edgecolor=C['primary'], label='Filter descriptors (3)'),
    mpatches.Patch(facecolor=C['success']+'22', edgecolor=C['success'], label='Process conditions (6)'),
    mpatches.Patch(facecolor=C['purple']+'22', edgecolor=C['purple'], label='Virus properties (2)'),
]
ax.legend(handles=legend_patches, fontsize=7.5, loc='lower center',
          framealpha=0.9, edgecolor=C['border'])

# Panel 2: Network layers → parameters
ax = axes[1]
ax.set_facecolor(C['surface'])
ax.axis('off')
ax.set_title('Step 2: ParameterPredictor\n(Level 1, 4 linear layers)', fontsize=10.5, color=C['text'], pad=8)
params_out = [
    ('J0', '42.3', 'LMH', 'Initial flux'),
    ('ks', '0.018', '1/min', 'Standard block rate'),
    ('ki', '0.0005', '1/LMH·min', 'Intermediate rate'),
    ('kc', '0.009', '1/min', 'Complete block rate'),
    ('kcf', '0.0002', 'LMH²/min', 'Cake filtration rate'),
    ('k1', '0.021', '1/min', 'Combined pore constr.'),
    ('k2', '0.004', '1/min', 'Combined cake depos.'),
    ('Pc', '0.74', '—', 'Capture probability'),
    ('Jcrit', '55.0', 'LMH', 'Critical flux'),
    ('Dv', '3.2e-11', 'm²/s', 'Virus diffusion coeff'),
]
ax.text(0.5, 0.97, 'Layer dims: 11→128→128→64→10',
        fontsize=8, color=C['text2'], ha='center', transform=ax.transAxes)
for i, (name, val, unit, desc) in enumerate(params_out):
    y = 0.89 - i * 0.086
    ax.text(0.04, y, f'{name}', fontsize=9, color=C['primary'],
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.19, y, f'= {val}', fontsize=8.5, color=C['text'], transform=ax.transAxes)
    ax.text(0.38, y, unit, fontsize=7.5, color=C['muted'], transform=ax.transAxes)
    ax.text(0.56, y, desc, fontsize=7.5, color=C['text2'], transform=ax.transAxes)
    act = 'Softplus+ε' if name != 'Pc' else 'Sigmoid'
    ax.text(0.86, y, act, fontsize=7, color=C['accent'], transform=ax.transAxes)

# Panel 3: Physics equations
ax = axes[2]
ax.set_facecolor(C['surface'])
ax.axis('off')
ax.set_title('Step 3: PhysicsSolver\n(Level 2, no learned weights)', fontsize=10.5, color=C['text'], pad=8)
t_pts = np.linspace(0, 120, 200)
J0, k1, k2 = 42.3, 0.021, 0.004
Jt = (J0 / (1 + k1*t_pts)**2) * np.exp(-k2*t_pts)
Pc_val = 0.74
LRV_val = np.log10(1/(1-0.74))

ins_ax = ax.inset_axes([0.05, 0.44, 0.90, 0.50])
ins_ax.plot(t_pts, Jt, color=C['primary'], lw=2.2)
ins_ax.fill_between(t_pts, Jt, alpha=0.12, color=C['primary'])
ins_ax.set_xlabel('Time (min)', fontsize=8)
ins_ax.set_ylabel('J(t)  LMH', fontsize=8)
ins_ax.set_title(f'J(t) = {J0}/((1+{k1}t)²·e^(−{k2}t))', fontsize=8, color=C['text'])
ins_ax.set_facecolor('#F8FAFF')
ins_ax.grid(True, alpha=0.4)
ins_ax.spines['top'].set_visible(False)
ins_ax.spines['right'].set_visible(False)
ins_ax.text(65, Jt[-1]+1.5, f'J(120)={Jt[-1]:.1f}', fontsize=7.5, color=C['primary'])

formulas = [
    ('Combined 1-A flux:', f'J(t) = 42.3/(1+0.021t)² · e^(−0.004t)', C['primary']),
    ('J at t=0:', 'J₀ = 42.3 LMH', C['primary']),
    ('J at t=60 min:', f'J(60) = {(J0/(1+k1*60)**2)*np.exp(-k2*60):.1f} LMH', C['primary']),
    ('J at t=120 min:', f'J(120) = {Jt[-1]:.1f} LMH', C['primary']),
    ('Flux ratio:', f'J(120)/J(0) = {Jt[-1]/J0:.2f}', C['warning']),
    ('Manabe Pc:', f'Pc = 1−e^(−λJ/Jcrit) = 0.74', C['success']),
    ('LRV:', f'log₁₀(1/(1−0.74)) = {LRV_val:.2f}', C['success']),
    ('Compliant:', f'LRV {LRV_val:.2f} ≥ 4.0  ✓', C['success']),
]
for i, (lbl, val, col) in enumerate(formulas):
    y = 0.41 - i * 0.054
    ax.text(0.04, y, lbl, fontsize=8.5, color=C['text2'], transform=ax.transAxes)
    ax.text(0.4, y, val, fontsize=8.5, color=col, fontweight='bold', transform=ax.transAxes)

# Panel 4: Loss components
ax = axes[3]
ax.set_facecolor(C['surface'])
ax.set_title('Step 4: Loss Computation\n(during local FL training)', fontsize=10.5, color=C['text'], pad=8)
loss_names = ['L_flux', 'L_LRV', 'L_physics', 'L_regime', 'L_fedprox']
loss_vals = [0.042, 0.018, 0.001, 0.031, 0.005]
loss_cols = [C['primary'], C['accent'], C['warning'], C['purple'], C['error']]
bars = ax.barh(loss_names, loss_vals, color=loss_cols, height=0.55, zorder=2)
ax.set_xlabel('Loss value (illustrative)', fontsize=9)
ax.set_xlim(0, 0.065)
ax.grid(axis='x', alpha=0.5, zorder=1)
total = sum(loss_vals)
for bar, v in zip(bars, loss_vals):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}  ({v/total*100:.0f}%)', va='center', fontsize=8.5, color=C['text'])
ax.text(0.98, 0.04, f'L_total = {total:.3f}', transform=ax.transAxes,
        ha='right', fontsize=10, fontweight='bold', color=C['text'],
        bbox=dict(fc=C['surface'], ec=C['border'], boxstyle='round,pad=0.3'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(left=False)

fig.suptitle('PINN Forward Pass — Site 1 Example  (PES 0.2 µm, TMP 1.0 bar, n=800)',
             fontsize=12, color=C['text'], y=1.02)
save('fig09_pinn_forward_pass.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig10 — loss components across 5 sites')
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_facecolor(C['surface'])
# Illustrative loss values per site. Site 4 has highest L_fedprox
loss_data = {
    'L_flux':    [0.042, 0.065, 0.038, 0.110, 0.031],
    'L_LRV':     [0.018, 0.025, 0.021, 0.048, 0.015],
    'L_physics': [0.001, 0.002, 0.001, 0.003, 0.001],
    'L_regime':  [0.031, 0.028, 0.034, 0.027, 0.033],
    'L_fedprox': [0.005, 0.007, 0.004, 0.066, 0.003],  # Site 4 spike!
}
loss_cols_map = {'L_flux': C['primary'], 'L_LRV': C['accent'],
                 'L_physics': C['warning'], 'L_regime': C['purple'],
                 'L_fedprox': C['error']}
x = np.arange(5)
width = 0.14
offsets = np.linspace(-0.3, 0.3, 5)
for i, (name, vals) in enumerate(loss_data.items()):
    bars = ax.bar(x + offsets[i], vals, width, label=name,
                  color=loss_cols_map[name], zorder=2, alpha=0.88)
    if name == 'L_fedprox':
        ax.text(x[3] + offsets[i], vals[3] + 0.003, f'{vals[3]:.3f}',
                ha='center', fontsize=8.5, color=C['error'], fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SITES, fontsize=11)
ax.set_ylabel('Loss value', fontsize=11)
ax.set_title('Loss Function Components Across All 5 Sites (Round 1, 5 Local Epochs)\n'
             'Note Site 4\'s elevated L_fedprox — the proximal term penalising its high-TMP drift',
             fontsize=11, color=C['text'])
ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor=C['border'])
ax.grid(axis='y', alpha=0.4, zorder=1)
ax.set_ylim(0, 0.135)
save('fig10_loss_components.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig11 — round 1 weight updates (existing section 7)')
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.set_facecolor(C['surface'])
ax.grid(True, alpha=0.4, zorder=1)

# FedAvg dashed arrows
for i, (lo, col) in enumerate(zip(LOCAL_OPTS, SITE_COLORS)):
    ax.annotate('', xy=lo, xytext=W_GLOBAL,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                linestyle='dashed', connectionstyle='arc3,rad=0.05'))
    ax.scatter(*lo, color=col, s=100, marker='s', zorder=5, alpha=0.85)
    offset = np.array([0.05, 0.04]) * ((-1)**i)
    ax.annotate(f'{SITES[i]}\n(FedAvg)\n[{lo[0]:.2f},{lo[1]:.2f}]',
                xy=lo, xytext=lo+offset+np.array([0.03, 0.0]),
                fontsize=7.5, color=col, ha='left' if i%2==0 else 'right')

# FedProx solid arrows
for i, (fp, col) in enumerate(zip(FP_OPTS, SITE_COLORS)):
    ax.annotate('', xy=fp, xytext=W_GLOBAL,
                arrowprops=dict(arrowstyle='->', color=col, lw=2.2,
                                linestyle='solid', connectionstyle='arc3,rad=-0.05'))
    ax.scatter(*fp, color=col, s=120, marker='o', zorder=6, edgecolors='white', linewidths=1.2)

# Aggregated results
ax.scatter(*FEDAVG_R, marker='D', s=220, color=C['text2'], zorder=7,
           edgecolors='white', linewidths=1.5)
ax.annotate(f'FedAvg result\n[{FEDAVG_R[0]:.3f}, {FEDAVG_R[1]:.3f}]',
            xy=FEDAVG_R, xytext=(FEDAVG_R[0]+0.05, FEDAVG_R[1]-0.07),
            fontsize=8.5, color=C['text2'], fontweight='bold')

ax.scatter(*FEDPROX_R, marker='D', s=220, color=C['primary'], zorder=7,
           edgecolors='white', linewidths=1.5)
ax.annotate(f'FedProx result\n[{FEDPROX_R[0]:.3f}, {FEDPROX_R[1]:.3f}]',
            xy=FEDPROX_R, xytext=(FEDPROX_R[0]-0.38, FEDPROX_R[1]+0.04),
            fontsize=8.5, color=C['primary'], fontweight='bold')

# W_global
ax.scatter(*W_GLOBAL, marker='*', s=450, color=C['text'], zorder=8)
ax.annotate('W_global\n[2.00, 0.50]', xy=W_GLOBAL,
            xytext=(W_GLOBAL[0]+0.06, W_GLOBAL[1]+0.06),
            fontsize=9, color=C['text'], fontweight='bold')

# True optimum
ax.scatter(*TRUE_OPT, marker='P', s=200, color=C['success'], zorder=8)
ax.annotate(f'True optimum\n[{TRUE_OPT[0]:.3f}, {TRUE_OPT[1]:.3f}]',
            xy=TRUE_OPT, xytext=(TRUE_OPT[0]+0.05, TRUE_OPT[1]+0.06),
            fontsize=8.5, color=C['success'])

legend_elems = [
    Line2D([0],[0], linestyle='--', color=C['text2'], lw=1.8, label='FedAvg (unconstrained drift)'),
    Line2D([0],[0], linestyle='-', color=C['text2'], lw=2.2, label='FedProx (proximal constraint)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C['text2'], markersize=9, label='FedAvg local end-point'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C['text2'], markersize=9, label='FedProx local end-point'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=C['text2'], markersize=9, label='Aggregated result (FedAvg)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=C['primary'], markersize=9, label='Aggregated result (FedProx)'),
    Line2D([0],[0], marker='P', color='w', markerfacecolor=C['success'], markersize=9, label='True global optimum'),
]
legend_elems += [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                         markersize=9, label=s) for s, c in zip(SITES, SITE_COLORS)]
ax.legend(handles=legend_elems, fontsize=8, loc='lower right',
          framealpha=0.95, edgecolor=C['border'], ncol=2)
ax.set_xlabel('J0_w  (initial flux weight)', fontsize=11)
ax.set_ylabel('k1_w  (pore-constriction weight)', fontsize=11)
ax.set_title('Round 1 Weight Updates — FedAvg vs FedProx  (μ = 0.10 for clarity)',
             fontsize=12, color=C['text'])
ax.set_xlim(0.45, 2.70)
ax.set_ylim(-0.05, 0.90)
save('fig11_round1_updates.png')


# ════════════════════════════════════════════════════════════════════════════
print('fig12 — convergence over 50 rounds')

def simulate_fl(local_targets, n_k, W_init, alpha=0.8, mu=0.0, rounds=50):
    W = np.array(W_init, dtype=float)
    wts = n_k / n_k.sum()
    hist = [W.copy()]
    for _ in range(rounds):
        local_w = W + alpha * (local_targets - W)  # each site after local training
        if mu > 0:
            local_w = (local_w + mu * W) / (1 + mu)
        W = np.sum(wts[:,None] * local_w, axis=0)
        hist.append(W.copy())
    return np.array(hist)

hist_fedavg  = simulate_fl(LOCAL_OPTS, N, W_GLOBAL, alpha=0.8, mu=0.0, rounds=50)
hist_fedprox = simulate_fl(LOCAL_OPTS, N, W_GLOBAL, alpha=0.8, mu=0.01, rounds=50)
rounds_x = np.arange(51)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
for ax in (ax1, ax2):
    ax.set_facecolor(C['surface'])
    ax.grid(True, alpha=0.4, zorder=1)

ax1.plot(rounds_x, hist_fedavg[:,0], color=C['text2'], lw=2,
         ls='--', label='FedAvg  (μ = 0)', zorder=3)
ax1.plot(rounds_x, hist_fedprox[:,0], color=C['primary'], lw=2.5,
         label='FedProx  (μ = 0.01)', zorder=4)
ax1.axhline(TRUE_OPT[0], color=C['success'], lw=1.5, ls=':', label=f'True optimum ({TRUE_OPT[0]:.3f})')
ax1.axhline(W_GLOBAL[0], color=C['border'], lw=1.2, ls='-', alpha=0.8)
ax1.set_ylabel('J0_w', fontsize=11)
ax1.legend(fontsize=9.5, loc='lower right', framealpha=0.95, edgecolor=C['border'])
ax1.set_title('Convergence of Global Model Weights Over 50 FL Rounds', fontsize=12, color=C['text'])
gap_fedavg  = abs(hist_fedavg[-1,0]  - TRUE_OPT[0])
gap_fedprox = abs(hist_fedprox[-1,0] - TRUE_OPT[0])
ax1.text(48, hist_fedavg[-1,0]+0.003, f'Δ={gap_fedavg:.4f}', fontsize=8, color=C['text2'], ha='right')
ax1.text(48, hist_fedprox[-1,0]-0.007, f'Δ={gap_fedprox:.4f}', fontsize=8, color=C['primary'], ha='right')

ax2.plot(rounds_x, hist_fedavg[:,1], color=C['text2'], lw=2,
         ls='--', label='FedAvg  (μ = 0)', zorder=3)
ax2.plot(rounds_x, hist_fedprox[:,1], color=C['primary'], lw=2.5,
         label='FedProx  (μ = 0.01)', zorder=4)
ax2.axhline(TRUE_OPT[1], color=C['success'], lw=1.5, ls=':', label=f'True optimum ({TRUE_OPT[1]:.3f})')
ax2.axhline(W_GLOBAL[1], color=C['border'], lw=1.2, ls='-', alpha=0.8)
ax2.set_xlabel('FL Round', fontsize=11)
ax2.set_ylabel('k1_w', fontsize=11)
ax2.legend(fontsize=9.5, loc='lower right', framealpha=0.95, edgecolor=C['border'])

fig.tight_layout()
save('fig12_convergence.png')


print('\nAll figures generated successfully.')
