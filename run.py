import numpy as np
import matplotlib.pyplot as plt
import src.parameters as p
from src.courtemanche import step_courtemanche

# ============================================================================
# condizioni iniziali
# ============================================================================

Y = np.array([
    -81.2,      # V
     11.2,      # Nai
    139.0,      # Ki
    1.02e-4,    # Cai
      1.49,     # CaUp
      1.49,     # CaRel
    0.00291,    # m
    0.965,      # h
    0.978,      # j
    0.0304,     # oa
    0.999,      # oi
    0.00496,    # ua
    0.999,      # ui
    3.29e-5,    # xr
    0.0187,     # xs
    1.37e-4,    # d
    0.999,      # f
    0.775,      # fCa
    0.0,        # u
    1.0,        # v
    0.999       # w
], dtype=float)

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================

dt_ms = p.dt_ms
Tmax_ms = p.stim_start + (p.n_stim - 1) * p.stim_period + p.stim_period
nt      = int(Tmax_ms / dt_ms)

last_stim_time = p.stim_start + (p.n_stim - 1) * p.stim_period
t_start_AP     = last_stim_time - 100.0
t_end_AP       = last_stim_time + 600.0

first_stim_time = p.stim_start
t_start_first_AP = first_stim_time - 100.0
t_end_first_AP   = first_stim_time + 600.0

# ============================================================================
# RECORDING
# ============================================================================

record_V = np.zeros(nt)
time_array = np.zeros(nt)

ion_records = {k: [] for k in [
    'INa','IK1','Ito','IKur','IKr','IKs','ICaL',
    'INaK','INaCa','IbNa','IbCa','IpCa'
]}

sr_records = {k: [] for k in ['I_rel','I_up','I_tr']}

conc_records = {k: [] for k in ['Nai','Ki','Cai','CaUp','CaRel']}

# ============================================================================
# SIMULAZIONE
# ============================================================================

for n in range(nt):
    t_ms = n * dt_ms
    Y, ion_curr, sr_curr, conc = step_courtemanche(Y, t_ms, dt_ms)

    record_V[n] = Y[0]
    time_array[n] = t_ms

    for k in ion_records:
        ion_records[k].append(ion_curr[k])
    for k in sr_records:
        sr_records[k].append(sr_curr[k])
    for k in conc_records:
        conc_records[k].append(conc[k])

    if n % max(1, nt // 10) == 0:
        print(f"{100*n/nt:3.0f}% — t = {t_ms:8.1f} ms | V = {Y[0]:6.1f} mV")

print("Simulazione completata!")

# ============================================================================
# CONVERSIONE
# ============================================================================

for k in ion_records:
    ion_records[k] = np.array(ion_records[k])

for k in sr_records:
    sr_records[k] = np.array(sr_records[k])

for k in conc_records:
    conc_records[k] = np.array(conc_records[k])

# ============================================================================
# GRAFICI
# ============================================================================

# --- Grafico 1: segnale completo ---
plt.figure(figsize=(8, 5))
plt.plot(time_array, record_V, 'b-', linewidth=0.8)
plt.xlabel("Tempo (ms)", fontsize=12)
plt.ylabel("V (mV)", fontsize=12)
plt.title(f"Courtemanche 1998 — {p.n_stim} potenziali d'azione atriali", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, Tmax_ms)
plt.tight_layout()
plt.show()

# --- Grafico 2: dettaglio ultimo AP ---
mask_AP = (time_array >= t_start_AP) & (time_array <= t_end_AP)
plt.figure(figsize=(8, 5))
plt.plot(time_array[mask_AP] - t_start_AP, record_V[mask_AP], 'b-', linewidth=1.5)
plt.xlabel("Tempo dalla depolarizzazione (ms)", fontsize=14)
plt.ylabel("V (mV)", fontsize=14)
plt.title(f"Ultimo AP — stimolo n°{p.n_stim}  |  t₀ = {last_stim_time:.0f} ms", fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



