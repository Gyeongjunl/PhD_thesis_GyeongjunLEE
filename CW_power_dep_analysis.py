"""
CW Power Dependence Analysis — X_D on chip 3-1
Lorentzian peak fitting + CW saturation model (I = Imax * P / (P + Psat))

Input:  "power dep chip 3_1 narrow emission.dat"
Output: CW_Lorentzian_fit_final.png

Usage:
    python CW_power_dep_analysis.py

Author: Gyeongjun Lee (analysis script generated with Claude)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE = "power dep chip 3_1 narrow emission.dat"   # same folder as this script
XD_CENTER = 1.9644   # eV — X_D peak center (chip 3-1)
MAX_POWER = 60       # µW — exclude points above this (fitting becomes unreliable)
FIT_RANGE_UW = 15    # µW — use only P <= this for Psat fit (rising part)

# ── Load data ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, DATA_FILE)

with open(data_path, 'r') as f:
    lines = f.readlines()

powers_uw = np.array([float(x) for x in lines[0].strip().split('\t')])
rows = []
for line in lines[2:]:
    vals = line.strip().split('\t')
    try:
        rows.append([float(v) for v in vals])
    except:
        pass

arr = np.array(rows)
energies = arr[:, 0]
spectra  = arr[:, 1:]

# Sort by energy (ascending)
idx = np.argsort(energies)
energies = energies[idx]
spectra  = spectra[idx, :]

# ── Lorentzian model ───────────────────────────────────────────────────────────
def lorentzian(E, A, E0, G, c0, c1):
    """Lorentzian peak + linear background.
    A  : amplitude
    E0 : center (eV)
    G  : FWHM (eV)
    c0, c1 : background offset and slope
    """
    E_mean = E.mean()
    return A * (G/2)**2 / ((E - E0)**2 + (G/2)**2) + c0 + c1*(E - E_mean)


def fit_lorentzian(E, S, E0_init=XD_CENTER, G_init=0.001):
    """Gauss-Newton nonlinear least-squares fit of Lorentzian + linear bg.

    Returns
    -------
    params : (A, E0, G, c0, c1)
    rss    : residual sum of squares
    """
    E_mean = E.mean()

    # Initial guess
    bg  = (S[:3].mean() + S[-3:].mean()) / 2
    S_bg = S - bg
    pk_i = np.argmax(S_bg)
    A   = max(float(S_bg[pk_i]), 1.0)
    E0  = float(E[pk_i]) if abs(float(E[pk_i]) - E0_init) < 0.005 else E0_init
    G   = G_init
    c0  = bg
    c1  = (S[-3:].mean() - S[:3].mean()) / (E[-1] - E[0])

    params = np.array([A, E0, G, c0, c1])

    for _ in range(600):
        A_, E0_, G_, c0_, c1_ = params
        half_G = G_ / 2
        denom  = (E - E0_)**2 + half_G**2

        I_pred    = A_ * half_G**2 / denom + c0_ + c1_*(E - E_mean)
        residuals = S - I_pred

        # Jacobian columns
        dI_dA  = half_G**2 / denom
        dI_dE0 = A_ * half_G**2 * 2*(E - E0_) / denom**2
        dI_dG  = A_ * half_G / denom - A_ * half_G**3 / denom**2
        dI_dc0 = np.ones(len(E))
        dI_dc1 = E - E_mean

        J   = np.column_stack([dI_dA, dI_dE0, dI_dG, dI_dc0, dI_dc1])
        JtJ = J.T @ J + 1e-8 * np.eye(5)
        Jtr = J.T @ residuals

        try:
            delta = np.linalg.solve(JtJ, Jtr)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search (keep A > 0, G > 0)
        step = 1.0
        for _ in range(25):
            p_new = params + step * delta
            if p_new[0] > 0 and p_new[2] > 5e-6:
                break
            step *= 0.5
        params = params + step * delta
        params[2] = abs(params[2])   # enforce G > 0

        if np.max(np.abs(step * delta)) < 1e-14:
            break

    rss = float(np.sum((S - lorentzian(E, *params))**2))
    return params, rss


# ── CW saturation model ────────────────────────────────────────────────────────
def cw_saturation(P, Imax, Psat):
    """Standard two-level CW saturation: I = Imax * P / (P + Psat)"""
    return Imax * P / (P + Psat)


# ── Fit all valid power points ─────────────────────────────────────────────────
valid_mask = powers_uw <= MAX_POWER
P_valid    = powers_uw[valid_mask]

E0_arr, fwhm_arr, area_arr, amp_arr = [], [], [], []
G_prev = 0.001   # warm-start FWHM from previous fit

for i, p in enumerate(powers_uw):
    if p > MAX_POWER:
        continue

    # Adaptive window: slightly wider at high power
    win  = min(0.012 + 0.0001 * p, 0.020)
    mask = (energies >= XD_CENTER - win) & (energies <= XD_CENTER + win)
    E_w  = energies[mask]
    S_w  = spectra[mask, i]

    params, _ = fit_lorentzian(E_w, S_w, E0_init=XD_CENTER, G_init=G_prev)
    A_, E0_, G_, c0_, c1_ = params
    G_prev = G_

    fwhm_meV = G_ * 1000                   # eV → meV
    area     = A_ * np.pi * (G_ / 2)       # Lorentzian analytical area

    E0_arr.append(E0_)
    fwhm_arr.append(fwhm_meV)
    area_arr.append(area)
    amp_arr.append(A_)

E0_arr   = np.array(E0_arr)
fwhm_arr = np.array(fwhm_arr)
area_arr = np.array(area_arr)
amp_arr  = np.array(amp_arr)

# ── Psat fit (grid search) ─────────────────────────────────────────────────────
rising = P_valid <= FIT_RANGE_UW
P_r    = P_valid[rising]
I_r    = area_arr[rising]

best_res, best_p = np.inf, None
for Imax in np.linspace(20, 80, 300):
    for Psat in np.linspace(0.5, 20, 400):
        res = float(np.sum((I_r - cw_saturation(P_r, Imax, Psat))**2))
        if res < best_res:
            best_res = res
            best_p   = (Imax, Psat)

Imax_fit, Psat_fit = best_p

# ── Print summary ──────────────────────────────────────────────────────────────
print("=" * 55)
print("Lorentzian fit results — X_D CW power dependence")
print("=" * 55)
print(f"{'P (µW)':>8}  {'E0 (eV)':>10}  {'FWHM (meV)':>11}  {'Area':>10}")
print("-" * 48)
for p, e0, fw, ar in zip(P_valid, E0_arr, fwhm_arr, area_arr):
    print(f"{p:8.2f}  {e0:10.6f}  {fw:11.3f}  {ar:10.4f}")
print()
print(f"CW saturation fit (P ≤ {FIT_RANGE_UW} µW):")
print(f"  Psat  = {Psat_fit:.2f} µW")
print(f"  Imax  = {Imax_fit:.3f} a.u.")
print(f"Pulsed Psat (chip 3-4, for reference): 11.3 µW")
print(f"FWHM : {fwhm_arr[0]:.3f} → {fwhm_arr[-1]:.3f} meV "
      f"({P_valid[0]:.2f} → {P_valid[-1]:.0f} µW)")
print(f"E0 shift: {(E0_arr[-1]-E0_arr[0])*1000:.2f} meV "
      f"({P_valid[0]:.2f} → {P_valid[-1]:.0f} µW)")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plt.rcParams.update({'font.size': 11})

# Panel 1 — Saturation curve
ax1 = axes[0]
P_plot = np.logspace(np.log10(0.04), np.log10(110), 400)
ax1.scatter(P_valid, area_arr, color='steelblue', s=60, zorder=5, label='Data')
ax1.plot(P_plot, cw_saturation(P_plot, Imax_fit, Psat_fit), 'r-', lw=2,
         label=f'CW fit: $P_{{\\mathrm{{sat}}}}$ = {Psat_fit:.1f} µW')
ax1.axvline(Psat_fit, color='gray', ls='--', alpha=0.5, lw=1)
ax1.text(Psat_fit * 1.15, area_arr.max() * 0.05,
         f'{Psat_fit:.1f} µW', color='gray', fontsize=9)
ax1.set_xscale('log')
ax1.set_xlabel('CW power (µW)')
ax1.set_ylabel('Lorentzian area (a.u.·eV)')
ax1.set_title('$X_D$ CW saturation — chip 3-1')
ax1.legend()

# Panel 2 — FWHM + peak shift
ax2 = axes[1]
ax2.scatter(P_valid, fwhm_arr, color='darkorange', s=60, zorder=5, label='FWHM')
ax2.axhline(fwhm_arr[0], color='gray', ls='--', alpha=0.5,
            label=f'Low-$P$ limit: {fwhm_arr[0]:.2f} meV')
ax2.set_xscale('log')
ax2.set_xlabel('CW power (µW)')
ax2.set_ylabel('Lorentzian FWHM (meV)', color='darkorange')
ax2.set_title('Linewidth vs. CW power')
ax2.legend(loc='upper left')
ax2b = ax2.twinx()
shift = (E0_arr - E0_arr[0]) * 1000
ax2b.scatter(P_valid, shift, color='teal', marker='s', s=40, alpha=0.8)
ax2b.set_ylabel('Peak shift (meV)', color='teal')
ax2b.axhline(0, color='teal', ls=':', alpha=0.3)

# Panel 3 — Spectra + Lorentzian fits
ax3 = axes[2]
cmap     = plt.cm.plasma
mask_plt = (energies >= 1.956) & (energies <= 1.975)
E_plt    = energies[mask_plt]
sel_idx  = [0, 4, 8, 11, 13, 15]

for k, j in enumerate(sel_idx):
    if j >= len(P_valid):
        continue
    orig_j = int(np.where(powers_uw == P_valid[j])[0][0])
    c      = cmap(k / len(sel_idx))
    S_plt  = spectra[mask_plt, orig_j]
    Smin, Smax = S_plt.min(), S_plt.max()
    if Smax == Smin:
        continue
    ax3.plot((E_plt - XD_CENTER) * 1000,
             (S_plt - Smin) / (Smax - Smin),
             color=c, lw=0.8, alpha=0.6)
    # Overlay fit
    win   = min(0.012 + 0.0001 * P_valid[j], 0.020)
    mask_w = (energies >= XD_CENTER - win) & (energies <= XD_CENTER + win)
    params, _ = fit_lorentzian(energies[mask_w], spectra[mask_w, orig_j])
    E_fine = np.linspace(E_plt[0], E_plt[-1], 400)
    I_fit  = lorentzian(E_fine, *params)
    ax3.plot((E_fine - XD_CENTER) * 1000,
             (I_fit - I_fit.min()) / (I_fit.max() - I_fit.min()),
             color=c, lw=2, ls='--', label=f'{P_valid[j]:.1f} µW')

ax3.set_xlabel('Energy rel. to $X_D$ (meV)')
ax3.set_ylabel('Norm. intensity')
ax3.set_title('Spectra + Lorentzian fits')
ax3.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join(script_dir, 'CW_Lorentzian_fit_final.png')
plt.savefig(out_path, dpi=200)
print(f"\nFigure saved → {out_path}")
