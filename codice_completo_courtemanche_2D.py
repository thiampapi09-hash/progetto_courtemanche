import numpy as np
import os
from datetime import datetime
from numba import njit, prange
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ============================================================================
# 1. PARAMETRI FISICI E CELLULARI (Courtemanche 1998 - modello atriale)
# ============================================================================
# Dominio 2D
Lx = 5.0                     # cm
Ly = 5.0                     # cm
Nx = 250                     # numero di intervalli in x
Ny = 250                     # numero di intervalli in y
dx = Lx / Nx                 # cm
dy = Ly / Ny                 # cm

dt = 0.01e-3                 # passo di integrazione (10 microsecondi) in secondi
Tmax = 10                    # durata totale: 10000 ms (10 stimoli)
nt = int(Tmax / dt)          # numero di passi

D = 0.05                     # coefficiente di diffusione (cm²/s)
# Condizione di stabilità (diffusione esplicita)
if (D * dt) / dx**2 > 1/3:
    raise RuntimeError("Condizione di stabilità violata: ridurre dt o aumentare dx")

# Costanti fisiche
R = 8.3143                   # J·K⁻¹·mol⁻¹
T = 310.0                    # K
F = 96.4867                  # C/mmol
RTF = R * T / F
FRT = 1 / RTF

# Geometria cellulare (Courtemanche)
Cm     = 100.0               # pF
V_cell = 20100.0             # µm³
V_i    = V_cell * 0.68       # µm³
V_up   = V_cell * 0.0552     # µm³
V_rel  = V_cell * 0.0048     # µm³

# Concentrazioni extracellulari (mM)
Ko  = 5.4
Nao = 140.0
Cao = 1.8

# Conduttanze massime (nS/pF)
gNa      = 7.8
gK1      = 0.09
gto      = 0.1652
gKur_base = 0.005
gKr      = 0.0294
gKs      = 0.129
gCaL     = 0.1238
gbNa     = 0.000674
gbCa     = 0.00113

# Correnti massime (pA/pF)
INaK_max = 0.60
INaCa_max = 1600.0
IpCa_max = 0.275
I_up_max = 0.005             # mM/ms

# Parametri INaK
KmNai = 10.0
KmKo  = 1.5

# Parametri INaCa
gamma_naca  = 0.35
KmNa_naca   = 87.5
KmCa_naca   = 1.38
ksat        = 0.1

# Parametri SR
K_rel     = 30.0
K_up      = 0.00092
Ca_up_max = 15.0
tau_tr    = 180.0

# Buffer del calcio (mM)
CMDN_max = 0.05
TRPN_max = 0.07
CSQN_max = 10.0
Km_CMDN  = 0.00238
Km_TRPN  = 0.0005
Km_CSQN  = 0.8

# Costanti per rilascio Ca dal JSR
c1_rel = 3.4175e-13
c2_rel = 13.67e-16

# Finestra di analisi dell’ultimo potenziale d’azione: invece di fissare manualmente un intervallo temporale (es. 9000–9600 ms),
# si utilizza l’ultimo istante di stimolazione generato dal modello. In questo modo la selezione dell’ultimo AP è coerente con la dinamica simulata
# e si adatta automaticamente a eventuali modifiche di stim_start, stim_period o Tmax.
# Stimolo elettrico (2 ms, -2000 pA = -20 pA/pF), periodicità 1000 ms (1 Hz)
stim_amplitude = -2000.0    # pA
stim_start     = 50.0       # ms
stim_duration  = 2.0        # ms
n_stim = 10                 # numero di stimoli
stim_period    = 1000.0     # ms

# ============================================================================
# FINESTRA ULTIMO POTENZIALE D'AZIONE (ultimo stimolo)
# ============================================================================

last_stim_time = stim_start + (n_stim - 1) * stim_period  # ultimo impulso
t_start_AP = last_stim_time - 100.0
t_end_AP   = last_stim_time + 600.0

# Fattore di temperatura per Ito e IKur
KQ10 = 3.0

# Larghezza della zona inizialmente depolarizzata (prime 'stim_width' righe)
stim_width = 5

# ============================================================================
# 2. FUNZIONI DI SUPPORTO
# ============================================================================
@njit(fastmath=True)
def E_K(Ki):   return RTF * np.log(Ko / Ki)
@njit(fastmath=True)
def E_Na(Nai): return RTF * np.log(Nao / Nai)
@njit(fastmath=True)
def E_Ca(Cai): return 0.5 * RTF * np.log(Cao / Cai)

@njit(fastmath=True)
def stimulus_current(t_ms, i, j):
    """10 stimoli applicati alle prime 'stim_width' righe, con periodicità 'stim_period' e durata 'stim_duration'"""
    if i < stim_width:
        for k in range(n_stim):
            t0 = stim_start + k * stim_period
            if t0 <= t_ms < t0 + stim_duration:
                return stim_amplitude / Cm
    return 0.0

@njit(fastmath=True)
def rush_larsen(x, x_inf, tau, dt_ms):
    return x_inf - (x_inf - x) * np.exp(-dt_ms / tau)

# ============================================================================
# 3. VARIABILI DI GATING (Courtemanche 1998) - identiche al codice 0D
# ============================================================================
@njit(fastmath=True)
def m_gate(V):
    if abs(V + 47.13) < 1e-6:
        alpha = 3.2
    else:
        alpha = 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))
    beta = 0.08 * np.exp(-V / 11.0)
    return alpha / (alpha + beta), 1.0 / (alpha + beta)

@njit(fastmath=True)
def h_gate(V):
    if V < -40.0:
        alpha = 0.135 * np.exp((V + 80.0) / -6.8)
        beta = 3.56 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.35 * V)
    else:
        alpha = 0.0
        beta = 1.0 / (0.13 * (1 + np.exp((V + 10.66) / -11.1)))
    if alpha + beta == 0:
        return 0.0, 1000.0
    return alpha / (alpha + beta), 1.0 / (alpha + beta)

@njit(fastmath=True)
def j_gate(V):
    if V < -40.0:
        alpha = ((-127140 * np.exp(0.2444 * V) - 3.474e-5 * np.exp(-0.04391 * V))
                 * (V + 37.78) / (1 + np.exp(0.311 * (V + 79.23))))
        beta = 0.1212 * np.exp(-0.01052 * V) / (1 + np.exp(-0.1378 * (V + 40.14)))
    else:
        alpha = 0.0
        beta = 0.3 * np.exp(-2.535e-7 * V) / (1 + np.exp(-0.1 * (V + 32.0)))
    if alpha + beta == 0:
        return 0.0, 1000.0
    return alpha / (alpha + beta), 1.0 / (alpha + beta)

@njit(fastmath=True)
def oa_gate(V):
    inf = 1.0 / (1 + np.exp((V + 20.47) / -17.54))
    alpha = 0.65 / (np.exp((V + 10.0) / -8.5) + np.exp((V - 30.0) / -59.0))
    beta = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
    tau = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau

@njit(fastmath=True)
def oi_gate(V):
    inf = 1.0 / (1 + np.exp((V + 43.1) / 5.3))
    alpha = 1.0 / (18.53 + np.exp((V + 113.7) / 10.95))
    beta = 1.0 / (35.56 + np.exp((V + 1.26) / -7.44))
    tau = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau

@njit(fastmath=True)
def ua_gate(V):
    inf = 1.0 / (1 + np.exp((V + 30.3) / -9.6))
    alpha = 0.65 / (np.exp((V + 10.0) / -8.5) + np.exp((V - 30.0) / -59.0))
    beta = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
    tau = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau

@njit(fastmath=True)
def ui_gate(V):
    inf = 1.0 / (1 + np.exp((V - 99.45) / 27.48))
    alpha = 1.0 / (21.0 + np.exp((V - 185.0) / -28.0))
    beta = 1.0 / np.exp((V - 158.0) / -16.0)
    tau = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau

@njit(fastmath=True)
def xr_gate(V):
    inf = 1.0 / (1 + np.exp((V + 14.1) / -6.5))
    if abs(V + 14.1) < 1e-6:
        alpha = 0.0003 * 5.0
    else:
        alpha = 0.0003 * (V + 14.1) / (1 - np.exp((V + 14.1) / -5.0))
    if abs(V - 3.3328) < 1e-7:
        beta = 7.3898e-5 * 5.1237
    else:
        beta = 7.3898e-5 * (V - 3.3328) / (np.exp((V - 3.3328) / 5.1237) - 1)
    tau = 1.0 / (alpha + beta)
    return inf, tau

@njit(fastmath=True)
def xs_gate(V):
    # Forma originale con radice quadrata (come da immagine fornita)
    inf = 1.0 / np.sqrt(1 + np.exp((V - 19.9) / -12.7))
    if abs(V - 19.9) < 1e-6:
        alpha = 4e-5 * 17.0
    else:
        alpha = 4e-5 * (V - 19.9) / (1 - np.exp((V - 19.9) / -17.0))
    if abs(V - 19.9) < 1e-6:
        beta = 3.5e-5 * 9.0
    else:
        beta = 3.5e-5 * (V - 19.9) / (np.exp((V - 19.9) / 9.0) - 1)
    tau = 0.5 / (alpha + beta)
    return inf, tau

@njit(fastmath=True)
def d_gate(V):
    inf = 1.0 / (1 + np.exp((V + 10.0) / -8.0))
    if abs(V + 10.0) < 1e-6:
        tau = 1.0 / (6.24 * 2 * 0.035)
    else:
        alpha = 0.035 * (V + 10.0) / (1 - np.exp(-(V + 10.0) / 6.24))
        beta = 0.035 * (V + 10.0) / (np.exp((V + 10.0) / 6.24) - 1)
        tau = 1.0 / (alpha + beta)
    return inf, tau

@njit(fastmath=True)
def f_gate(V):
    inf = 1.0 / (1 + np.exp((V + 28.0) / 6.9))
    tau = 9.0 / (0.0197 * np.exp(-0.03372 * (V + 10.0)**2) + 0.02)
    return inf, tau

@njit(fastmath=True)
def fCa_gate(Cai):
    inf = 1.0 / (1 + Cai / 0.00035)
    tau = 2.0
    return inf, tau

@njit(fastmath=True)
def u_gate(Fn):
    inf = 1.0 / (1 + np.exp(-(Fn - c1_rel) / c2_rel))
    tau = 8.0
    return inf, tau

@njit(fastmath=True)
def v_gate(Fn):
    inf = 1 - 1.0 / (1 + np.exp(-(Fn - 0.2 * c1_rel) / c2_rel))
    tau = 1.91 + 2.09 / (1 + np.exp(-(Fn - c1_rel) / c2_rel))
    return inf, tau

@njit(fastmath=True)
def w_gate(V):
    inf = 1 - 1.0 / (1 + np.exp(-(V - 40.0) / 17.0))
    if abs(V - 7.9) < 1e-6:
        tau = 6.0 * (2.0 / 13.0)
    else:
        tau = 6.0 * (1 - np.exp(-(V - 7.9) / 5.0)) / ((1 + 0.3 * np.exp(-(V - 7.9) / 5.0)) * (V - 7.9))
    return inf, tau

@njit(fastmath=True)
def gKur(V):
    return gKur_base * (1 + 10.0 / (1 + np.exp((V - 15.0) / -13.0)))

# ============================================================================
# 4. FUNZIONE PRINCIPALE DI REAZIONE-DIFFUSIONE (2D)
# ============================================================================
@njit(parallel=True, fastmath=True)
def step_courtemanche_2d(Y, t_ms, dt_ms, dx, D):
    """
    Aggiorna lo stato del modello 2D per un passo temporale.
    Y shape: (Nx, Ny, nvars) con nvars = 21
    """
    Nx, Ny, nvars = Y.shape
    Y_new = np.empty_like(Y)

    # Calcolo del laplaciano per V (diffusione)
    V = Y[:,:,0]
    laplacian = np.zeros_like(V)
    for i in prange(Nx):
        for j in range(Ny):
            i_prev = i - 1 if i > 0 else 0
            i_next = i + 1 if i < Nx - 1 else Nx - 1
            j_prev = j - 1 if j > 0 else 0
            j_next = j + 1 if j < Ny - 1 else Ny - 1
            laplacian[i, j] = (V[i_prev, j] + V[i_next, j] +
                               V[i, j_prev] + V[i, j_next] - 4 * V[i, j]) / (dx**2)

    for i in prange(Nx):
        for j in range(Ny):
            # Estrai variabili di stato
            V_val, Nai, Ki, Cai, CaUp, CaRel = Y[i, j, 0:6]
            m, h, j_g, oa, oi, ua, ui, xr, xs, d, f, fCa, u, v, w = Y[i, j, 6:]

            # Potenziali di inversione
            EK  = E_K(Ki)
            ENa = E_Na(Nai)
            ECa = E_Ca(Cai)

            # Correnti ioniche (identiche al modello 0D)
            INa  = gNa * m**3 * h * j_g * (V_val - ENa)
            IK1  = gK1 * (V_val - EK) / (1 + np.exp(0.07 * (V_val + 80.0)))
            Ito  = gto * oa**3 * oi * (V_val - EK)
            IKur = gKur(V_val) * ua**3 * ui * (V_val - EK)
            IKr  = gKr * xr * (V_val - EK) / (1 + np.exp((V_val + 15.0) / 22.4))
            IKs  = gKs * xs**2 * (V_val - EK)
            ICaL = gCaL * d * f * fCa * (V_val - ECa)  

            sigma = (np.exp(Nao / 67.3) - 1) / 7.0
            fNaK  = 1.0 / (1 + 0.1245 * np.exp(-0.1 * V_val * FRT) + 0.0365 * sigma * np.exp(-V_val * FRT))
            INaK  = INaK_max * fNaK * Ko / (Ko + KmKo) / (1 + (KmNai / Nai)**1.5)

            INaCa = INaCa_max * (
                np.exp(gamma_naca * V_val * FRT) * Nai**3 * Cao
                - np.exp((gamma_naca - 1) * V_val * FRT) * Nao**3 * Cai
            ) / (
                (KmNa_naca**3 + Nao**3) * (KmCa_naca + Cao)
                * (1 + ksat * np.exp((gamma_naca - 1) * V_val * FRT))
            )

            IbCa = gbCa * (V_val - ECa)
            IbNa = gbNa * (V_val - ENa)
            IpCa = IpCa_max * Cai / (0.0005 + Cai)

            I_stim = stimulus_current(t_ms, i, j)
            I_ion  = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IpCa + INaK + INaCa + IbNa + IbCa

            # Dinamica del calcio intracellulare e SR
            Fn = (1e-12 * V_rel * K_rel * u**2 * v * w * (CaRel - Cai)
                  - 5e-13 / F * (0.5 * ICaL - 0.2 * INaCa) * Cm)

            I_rel    = K_rel * u**2 * v * w * (CaRel - Cai)
            I_up     = I_up_max / (1 + K_up / Cai)
            I_up_leak = I_up_max * CaUp / Ca_up_max
            I_tr     = (CaUp - CaRel) / tau_tr

            # Aggiornamento variabili di gating (Rush-Larsen)
            m_inf,  tau_m   = m_gate(V_val);       m_new   = rush_larsen(m,   m_inf,   tau_m,   dt_ms)
            h_inf,  tau_h   = h_gate(V_val);       h_new   = rush_larsen(h,   h_inf,   tau_h,   dt_ms)
            j_inf,  tau_j   = j_gate(V_val);       j_new   = rush_larsen(j_g, j_inf,   tau_j,   dt_ms)
            oa_inf, tau_oa  = oa_gate(V_val);      oa_new  = rush_larsen(oa,  oa_inf,  tau_oa,  dt_ms)
            oi_inf, tau_oi  = oi_gate(V_val);      oi_new  = rush_larsen(oi,  oi_inf,  tau_oi,  dt_ms)
            ua_inf, tau_ua  = ua_gate(V_val);      ua_new  = rush_larsen(ua,  ua_inf,  tau_ua,  dt_ms)
            ui_inf, tau_ui  = ui_gate(V_val);      ui_new  = rush_larsen(ui,  ui_inf,  tau_ui,  dt_ms)
            xr_inf, tau_xr  = xr_gate(V_val);      xr_new  = rush_larsen(xr,  xr_inf,  tau_xr,  dt_ms)
            xs_inf, tau_xs  = xs_gate(V_val);      xs_new  = rush_larsen(xs,  xs_inf,  tau_xs,  dt_ms)
            d_inf,  tau_d   = d_gate(V_val);       d_new   = rush_larsen(d,   d_inf,   tau_d,   dt_ms)
            f_inf,  tau_f   = f_gate(V_val);       f_new   = rush_larsen(f,   f_inf,   tau_f,   dt_ms)
            fCa_inf,tau_fCa = fCa_gate(Cai);       fCa_new = rush_larsen(fCa, fCa_inf, tau_fCa, dt_ms)
            u_inf,  tau_u   = u_gate(Fn);          u_new   = rush_larsen(u,   u_inf,   tau_u,   dt_ms)
            v_inf,  tau_v   = v_gate(Fn);          v_new   = rush_larsen(v,   v_inf,   tau_v,   dt_ms)
            w_inf,  tau_w   = w_gate(V_val);       w_new   = rush_larsen(w,   w_inf,   tau_w,   dt_ms)

            # Termine diffusivo convertito in mV/ms
            diff_term_ms = (D * laplacian[i, j]) / 1000.0
            dVdt = diff_term_ms - (I_ion + I_stim)
            V_new = V_val + dt_ms * dVdt

            # Aggiornamento concentrazioni (Eulero esplicito)
            dNaidt  = (-3 * INaK - (3 * INaCa + IbNa + INa)) * Cm / (V_i * F)
            Nai_new = Nai + dt_ms * dNaidt

            
            dKidt   = (2 * INaK - (IK1 + Ito + IKur + IKr + IKs + I_stim)) * Cm / (V_i * F)
            Ki_new  = Ki + dt_ms * dKidt

            B1 = ((2 * INaCa - (IpCa + ICaL + IbCa)) * Cm / (2 * V_i * F)
                  + (V_up * (I_up_leak - I_up) + I_rel * V_rel) / V_i)
            B2 = (1 + TRPN_max * Km_TRPN / (Cai + Km_TRPN)**2
                  + CMDN_max * Km_CMDN / (Cai + Km_CMDN)**2)
            Cai_new = Cai + dt_ms * B1 / B2

            dCaUpdt  = I_up - (I_up_leak + I_tr * V_rel / V_up)
            CaUp_new = CaUp + dt_ms * dCaUpdt

            buffer_factor = 1 + CSQN_max * Km_CSQN / (CaRel + Km_CSQN)**2
            CaRel_new = CaRel + dt_ms * (I_tr - I_rel) / buffer_factor

            # Costruzione nuovo stato
            Y_new[i, j, 0] = V_new
            Y_new[i, j, 1] = Nai_new
            Y_new[i, j, 2] = Ki_new
            Y_new[i, j, 3] = Cai_new
            Y_new[i, j, 4] = CaUp_new
            Y_new[i, j, 5] = CaRel_new
            Y_new[i, j, 6] = m_new
            Y_new[i, j, 7] = h_new
            Y_new[i, j, 8] = j_new
            Y_new[i, j, 9] = oa_new
            Y_new[i, j,10] = oi_new
            Y_new[i, j,11] = ua_new
            Y_new[i, j,12] = ui_new
            Y_new[i, j,13] = xr_new
            Y_new[i, j,14] = xs_new
            Y_new[i, j,15] = d_new
            Y_new[i, j,16] = f_new
            Y_new[i, j,17] = fCa_new
            Y_new[i, j,18] = u_new
            Y_new[i, j,19] = v_new
            Y_new[i, j,20] = w_new

    return Y_new

# ============================================================================
# 5. CONDIZIONI INIZIALI
# ============================================================================
Y0_cell = np.array([
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

# Inizializzazione griglia 2D
Y = np.zeros((Nx, Ny, len(Y0_cell)), dtype=float)
for i in range(Nx):
    for j in range(Ny):
        Y[i, j, :] = Y0_cell.copy()

dt_ms = dt * 1000   # converte in millisecondi

# ============================================================================
# 6. LOOP PRINCIPALE (10 stimoli, durata 10000 ms)
# ============================================================================
save_every = 10      # salva ogni 10 passi (5 ms) per avere una GIF fluida
V_record = []
t_record = []

print("Inizio simulazione 2D Courtemanche 1998 - 10 stimoli (10000 ms)...")
print(f"Dominio: Lx={Lx} cm, Ly={Ly} cm, griglia {Nx}x{Ny} punti (dx={dx:.5f} cm)")
start_time = time.time()

for n in range(nt):
    t_ms_n = n * dt_ms
    Y = step_courtemanche_2d(Y, t_ms_n, dt_ms, dx, D)

    if n % save_every == 0:
        V_record.append(Y[:,:,0].copy())
        t_record.append(t_ms_n / 1000.0)   # tempo in secondi

    if n % (nt // 10) == 0 and n > 0:
        elapsed = time.time() - start_time
        print(f"{100*n/nt:.0f}% completato, t = {t_ms_n/1000:.2f} s, elapsed = {elapsed:.1f} s")

V_record = np.array(V_record)
t_record = np.array(t_record)

# ============================================================================
# 7. CALCOLO DELLA CONDUCTION VELOCITY (CV) PER INTERPOLAZIONE LINEARE
# ============================================================================
threshold = -20.0   # soglia in mV
col = Ny // 2       # colonna centrale lungo y

# Punti lungo x: uno dopo la zona stimolata, uno vicino al bordo opposto
i1 = stim_width + 15
i2 = Nx - 10        # punto prima del bordo per evitare effetti di confine

def find_activation_time(V_trace, threshold, t_record):
    """Restituisce il tempo (in secondi) di attraversamento della soglia per interpolazione lineare."""
    for k in range(1, len(V_trace)):
        if V_trace[k-1] < threshold and V_trace[k] >= threshold:
            V_prev = V_trace[k-1]
            V_now  = V_trace[k]
            t_prev = t_record[k-1]
            t_now  = t_record[k]
            return t_prev + (t_now - t_prev) * (threshold - V_prev) / (V_now - V_prev)
    return np.nan

trace1 = V_record[:, i1, col]
trace2 = V_record[:, i2, col]

t1 = find_activation_time(trace1, threshold, t_record)
t2 = find_activation_time(trace2, threshold, t_record)

print("\n" + "="*60)
if not np.isnan(t1) and not np.isnan(t2):
    distance = (i2 - i1) * dx   # cm
    delta_t = t2 - t1            # secondi
    CV = distance / delta_t      # cm/s
    print(f"Distanza tra i punti: {distance:.3f} cm")
    print(f"Tempo di attivazione a x = {i1*dx:.2f} cm: {t1*1000:.2f} ms")
    print(f"Tempo di attivazione a x = {i2*dx:.2f} cm: {t2*1000:.2f} ms")
    print(f"Conduction velocity (CV) = {CV:.2f} cm/s  ({CV/100:.2f} m/s)")
else:
    print("ERRORE: la soglia di depolarizzazione non è stata raggiunta in uno dei punti.")
    print("Verificare la propagazione dell'onda o ridurre la soglia.")
    CV = np.nan
print("="*60)

# ============================================================================
# 8. SALVATAGGIO, PLOT DEL POTENZALE D'AZIONE AL CENTRO E CREAZIONE GIF
# ============================================================================
outdir = "output_courtemanche_2D_10stim"
os.makedirs(outdir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

np.savez_compressed(
    f"{outdir}/courtemanche_2d_{ts}.npz",
    V=V_record,
    t=t_record,
    dx=dx, dy=dx, dt=dt, D=D,
    stim_width=stim_width,
    CV=CV
)

# --- Plot del potenziale d'azione al centro del dominio ---
center_x = Nx // 2
center_y = Ny // 2
V_center = V_record[:, center_x, center_y]

# =====================================================================
# FINESTRA ULTIMO POTENZIALE D'AZIONE
# =====================================================================
t_start_AP_s = t_start_AP * 1e-3
t_end_AP_s   = t_end_AP * 1e-3
mask_AP = (t_record >= t_start_AP_s) & (t_record <= t_end_AP_s)
# estrazione ultimo AP
t_AP = t_record[mask_AP]
V_center_AP = V_center[mask_AP]
# --- PLOT 1D (solo ultimo AP) ---
plt.figure(figsize=(10, 6))
plt.plot(t_record, V_center, 'b-', linewidth=1.2, alpha=0.4)  # contesto globale
plt.plot(t_AP, V_center_AP, 'r-', linewidth=2.0)              # ultimo AP
plt.xlabel('Tempo (s)')
plt.ylabel('Potenziale di membrana (mV)')
plt.title(
    f"Potenziale d'azione al centro del dominio "
    f"(x={center_x*dx:.2f} cm, y={center_y*dy:.2f} cm)\n"
    + (f"CV = {CV:.1f} cm/s" if not np.isnan(CV) else "")
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/PA_center_{ts}.png", dpi=150)
plt.close()

# --- ANIMAZIONE 2D (SOLO ULTIMO AP) ---
print("\nCreazione GIF (ultimo AP)...")
# estrazione frames ultimo AP
V_AP_frames = V_record[mask_AP]
t_AP_frames = t_record[mask_AP]
fig, ax = plt.subplots()
extent = [0, Lx, 0, Ly]
im = ax.imshow(
    V_AP_frames[0],
    cmap="jet",
    origin="lower",
    extent=extent,
    vmin=-85,
    vmax=40,
    aspect='auto'
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Membrane potential (mV)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("Ultimo potenziale d'azione")
def update(k):
    im.set_data(V_AP_frames[k])
    ax.set_title(f"t = {t_AP_frames[k]:.2f} s")
    return [im]
anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(V_AP_frames),
    interval=100
)
anim.save(
    f"{outdir}/Courtemanche_2D_lastAP_{ts}.gif",
    writer="pillow",
    fps=15
)
plt.close()
print(f"Simulazione completata. Dati salvati in {outdir}/")
print(f"Grafico PA centrale salvato: {outdir}/PA_center_{ts}.png")
print(f"GIF salvata: {outdir}/Courtemanche_2D_lastAP_{ts}.gif")im.save(
    f"{outdir}/Courtemanche_2D_lastAP_{ts}.gif",
    writer="pillow",
    fps=15
)
plt.close()
print(f"Simulazione completata. Dati salvati in {outdir}/")
print(f"Grafico PA centrale salvato: {outdir}/PA_center_{ts}.png")
print(f"GIF salvata: {outdir}/Courtemanche_2D_lastAP_{ts}.gif")