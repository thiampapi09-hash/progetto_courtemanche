import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SISTEMA DI UNITÀ ADOTTATO IN TUTTO IL CODICE (senza eccezioni):
#   Tempo          → millisecondi (ms)
#   Potenziale     → millivolt (mV)
#   Concentrazioni → millimolare (mM)
#   Correnti       → pA/pF  (densità di corrente di membrana)
#   Capacità       → picofarad (pF)
#   Volumi         → µm³
#   Costante di Faraday F = 96.4867 C/mmol, usata come µA·ms/mmol
#     (stesso valore numerico; il fattore di scala 1e6 è esplicito nelle ODE
#      di concentrazione — vedi commenti nelle equazioni differenziali)
# ============================================================================

# ============================================================================
# 1. PARAMETRI FISICI E CELLULARI
# ============================================================================

# --- Passo di integrazione (ms) ---
dt_ms = 0.05        # ms  (= 50 µs)

# --- Costanti fisiche ---
R   = 8.3143        # J/(mol·K)
T   = 310.0         # K  (37 °C → T_K = T_Celsius + 273.15)
F   = 96.4867       # C/mmol  (= µA·ms/mmol; fattore 1e6 esplicito nelle ODE)
RTF = R * T / F     # mV  — fattore termico per i potenziali di Nernst
FRT = 1.0 / RTF     # 1/mV

# --- Geometria cellulare (volumi in µm³) ---
Cm     = 100.0              # pF
V_cell = 20100.0            # µm³
V_i    = V_cell * 0.68      # µm³  volume intracellulare
V_up   = V_cell * 0.0552    # µm³  SR di uptake (NSR)
V_rel  = V_cell * 0.0048    # µm³  SR di release (JSR)

# --- Concentrazioni extracellulari (mM) ---
Ko  = 5.4
Nao = 140.0
Cao = 1.8

# --- Conduttanze massime (nS/pF = pA/pF per mV) ---
gNa       = 7.8
gK1       = 0.09
gto       = 0.1652
gKur_base = 0.005
gKr       = 0.0294
gKs       = 0.129
gCaL      = 0.1238
gbNa      = 0.000674
gbCa      = 0.00113

# --- Correnti massime di pompa/scambiatore (pA/pF) ---
INaK_max  = 0.6
INaCa_max = 1600.0
IpCa_max  = 0.275

# --- Uptake SR massimo (mM/ms) ---
I_up_max = 0.005

# --- Parametri pompa Na/K ---
KmNai = 10.0    # mM
KmKo  = 1.5     # mM

# --- Parametri scambiatore Na/Ca ---
gamma_naca = 0.35
KmNa_naca  = 87.5   # mM
KmCa_naca  = 1.38   # mM
ksat       = 0.1

# --- Parametri SR (ms e mM) ---
K_rel     = 30.0     # 1/ms
K_up      = 0.00092  # mM
Ca_up_max = 15.0     # mM
tau_tr    = 180.0    # ms

# --- Buffer del calcio (mM) ---
CMDN_max = 0.05
TRPN_max = 0.07
CSQN_max = 10.0
Km_CMDN  = 0.00238
Km_TRPN  = 0.0005
Km_CSQN  = 0.8

# --- Costanti segnale di rilascio Ca (Fn) ---
c1_rel = 3.4175e-13
c2_rel = 13.67e-16

# --- Fattore di temperatura (adimensionale) ---
KQ10 = 3.0

# ============================================================================
# PARAMETRI DELLO STIMOLO E FINESTRA TEMPORALE DINAMICA
# ============================================================================
stim_amplitude = -2000.0    # pA   (diviso Cm = pA/pF nel loop)
stim_start     =    50.0    # ms   inizio stimolo all'interno di ogni ciclo
stim_duration  =     2.0    # ms   durata di ogni impulso
stim_period    =  1000.0    # ms   periodo di pacing  (1 Hz)
n_stim         =    10      # —    numero totale di stimoli da erogare

Tmax_ms = stim_start + (n_stim - 1) * stim_period + stim_period
nt      = int(Tmax_ms / dt_ms)

last_stim_time = stim_start + (n_stim - 1) * stim_period
t_start_AP     = last_stim_time - 100.0
t_end_AP       = last_stim_time + 600.0

first_stim_time = stim_start
t_start_first_AP = first_stim_time - 100.0
t_end_first_AP   = first_stim_time + 600.0

# ============================================================================
# 2. FUNZIONI DI SUPPORTO
# ============================================================================

def E_K(Ki):
    return RTF * np.log(Ko / Ki)

def E_Na(Nai):
    return RTF * np.log(Nao / Nai)

def E_Ca(Cai):
    return 0.5 * RTF * np.log(Cao / Cai)

def stimulus_current(t_ms):
    t_in_cycle = t_ms % stim_period
    if stim_start <= t_in_cycle < stim_start + stim_duration:
        return stim_amplitude / Cm
    return 0.0

def rush_larsen(x, x_inf, tau_ms, dt_ms):
    return x_inf - (x_inf - x) * np.exp(-dt_ms / tau_ms)

# ============================================================================
# 3. VARIABILI DI GATING
# ============================================================================

def m_gate(V):
    if abs(V + 47.13) < 1e-6:
        alpha = 3.2
    else:
        alpha = 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))
    beta   = 0.08 * np.exp(-V / 11.0)
    tau_ms = 1.0 / (alpha + beta)
    return alpha / (alpha + beta), tau_ms

def h_gate(V):
    if V < -40.0:
        alpha = 0.135 * np.exp((V + 80.0) / -6.8)
        beta  = 3.56 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.35 * V)
    else:
        alpha = 0.0
        beta  = 1.0 / (0.13 * (1 + np.exp((V + 10.66) / -11.1)))
    if alpha + beta == 0:
        return 0.0, 1000.0
    tau_ms = 1.0 / (alpha + beta)
    return alpha / (alpha + beta), tau_ms

def j_gate(V):
    if V < -40.0:
        alpha = ((-127140 * np.exp(0.2444 * V) - 3.474e-5 * np.exp(-0.04391 * V))
                 * (V + 37.78) / (1 + np.exp(0.311 * (V + 79.23))))
        beta  = 0.1212 * np.exp(-0.01052 * V) / (1 + np.exp(-0.1378 * (V + 40.14)))
    else:
        alpha = 0.0
        beta  = 0.3 * np.exp(-2.535e-7 * V) / (1 + np.exp(-0.1 * (V + 32.0)))
    if alpha + beta == 0:
        return 0.0, 1000.0
    tau_ms = 1.0 / (alpha + beta)
    return alpha / (alpha + beta), tau_ms

def oa_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 20.47) / -17.54))
    alpha  = 0.65 / (np.exp((V + 10.0) / -8.5) + np.exp((V - 30.0) / -59.0))
    beta   = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
    tau_ms = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau_ms

def oi_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 43.1) / 5.3))
    alpha  = 1.0 / (18.53 + np.exp((V + 113.7) / 10.95))
    beta   = 1.0 / (35.56 + np.exp((V + 1.26) / -7.44))
    tau_ms = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau_ms

def ua_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 30.3) / -9.6))
    alpha  = 0.65 / (np.exp((V + 10.0) / -8.5) + np.exp((V - 30.0) / -59.0))
    beta   = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
    tau_ms = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau_ms

def ui_gate(V):
    inf    = 1.0 / (1 + np.exp((V - 99.45) / 27.48))
    alpha  = 1.0 / (21.0 + np.exp((V - 185.0) / -28.0))
    beta   = 1.0 / np.exp((V - 158.0) / -16.0)
    tau_ms = 1.0 / ((alpha + beta) * KQ10)
    return inf, tau_ms

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
    tau_ms = 1.0 / (alpha + beta)
    return inf, tau_ms

def xs_gate(V):
    inf = 1.0 / np.sqrt(1 + np.exp((V - 19.9) / -12.7))
    if abs(V - 19.9) < 1e-6:
        alpha = 4e-5 * 17.0
    else:
        alpha = 4e-5 * (V - 19.9) / (1 - np.exp((V - 19.9) / -17.0))
    if abs(V - 19.9) < 1e-6:
        beta = 3.5e-5 * 9.0
    else:
        beta = 3.5e-5 * (V - 19.9) / (np.exp((V - 19.9) / 9.0) - 1)
    tau_ms = 0.5 / (alpha + beta)
    return inf, tau_ms

def d_gate(V):
    inf = 1.0 / (1 + np.exp((V + 10.0) / -8.0))
    if abs(V + 10.0) < 1e-6:
        tau_ms = 1.0 / (6.24 * 2 * 0.035)
    else:
        tau_ms = ((1 - np.exp((V + 10.0) / -6.24))
                  / (0.035 * (V + 10.0) * (1 + np.exp((V + 10.0) / -6.24))))
    return inf, tau_ms

def f_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 28.0) / 6.9))
    tau_ms = 9.0 / (0.0197 * np.exp(-0.0337**2 * (V + 10.0)**2) + 0.02)
    return inf, tau_ms

def fCa_gate(Cai):
    inf    = 1.0 / (1 + Cai / 0.00035)
    tau_ms = 2.0
    return inf, tau_ms

def u_gate(Fn):
    inf    = 1.0 / (1 + np.exp(-(Fn - c1_rel) / c2_rel))
    tau_ms = 8.0
    return inf, tau_ms

def v_gate(Fn):
    inf    = 1 - 1.0 / (1 + np.exp(-(Fn - 0.2 * c1_rel) / c2_rel))
    tau_ms = 1.91 + 2.09 / (1 + np.exp(-(Fn - c1_rel) / c2_rel))
    return inf, tau_ms

def w_gate(V):
    inf = 1 - 1.0 / (1 + np.exp(-(V - 40.0) / 17.0))
    if abs(V - 7.9) < 1e-6:
        tau_ms = 6.0 * (2.0 / 13.0)
    else:
        tau_ms = (6.0 * (1 - np.exp(-(V - 7.9) / 5.0))
                  / ((1 + 0.3 * np.exp(-(V - 7.9) / 5.0)) * (V - 7.9)))
    return inf, tau_ms

def gKur(V):
    return gKur_base * (1 + 10.0 / (1 + np.exp((V - 15.0) / -13.0)))

# ============================================================================
# 4. PASSO DI INTEGRAZIONE (modificato per restituire anche correnti e concentrazioni)
# ============================================================================

def step_courtemanche(Y, t_ms, dt_ms):
    """
    Esegue un passo di integrazione.
    Restituisce:
        Y_new : array nuovo stato
        ion_currents : dict con correnti ioniche (pA/pF)
        sr_currents  : dict con correnti del reticolo (mM/ms)
        conc : dict con concentrazioni (mM)
    """
    V, Nai, Ki, Cai, CaUp, CaRel = Y[0:6]
    m, h, j, oa, oi, ua, ui, xr, xs, d, f, fCa, u, v, w = Y[6:]

    # Potenziali di inversione
    EK  = E_K(Ki)
    ENa = E_Na(Nai)
    ECa = E_Ca(Cai)

    # Correnti ioniche
    INa   = gNa * m**3 * h * j * (V - ENa)
    IK1   = gK1 * (V - EK) / (1 + np.exp(0.07 * (V + 80.0)))
    Ito   = gto * oa**3 * oi * (V - EK)
    IKur  = gKur(V) * ua**3 * ui * (V - EK)
    IKr   = gKr * xr * (V - EK) / (1 + np.exp((V + 15.0) / 22.4))
    IKs   = gKs * xs**2 * (V - EK)
    ICaL  = gCaL * d * f * fCa * (V - ECa)

    sigma = (np.exp(Nao / 67.3) - 1) / 7.0
    fNaK  = 1.0 / (1 + 0.1245 * np.exp(-0.1 * V * FRT) + 0.0365 * sigma * np.exp(-V * FRT))
    INaK  = INaK_max * fNaK * Ko / (Ko + KmKo) / (1 + (KmNai / Nai)**1.5)

    INaCa = INaCa_max * (
        np.exp(gamma_naca * V * FRT) * Nai**3 * Cao
        - np.exp((gamma_naca - 1) * V * FRT) * Nao**3 * Cai
    ) / (
        (KmNa_naca**3 + Nao**3) * (KmCa_naca + Cao)
        * (1 + ksat * np.exp((gamma_naca - 1) * V * FRT))
    )

    IbNa  = gbNa * (V - ENa)
    IbCa  = gbCa * (V - ECa)
    IpCa  = IpCa_max * Cai / (0.0005 + Cai)

    I_stim = stimulus_current(t_ms)

    I_ion = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IpCa + INaK + INaCa + IbNa + IbCa

    # Dinamica del calcio e SR
    Fn = (1e-12 * V_rel * K_rel * u**2 * v * w * (CaRel - Cai)
          - 5e-13 / F * (0.5 * ICaL - 0.2 * INaCa) * Cm)

    I_rel     = K_rel * u**2 * v * w * (CaRel - Cai)    # mM/ms
    I_up      = I_up_max / (1 + K_up / Cai)             # mM/ms
    I_up_leak = I_up_max * CaUp / Ca_up_max             # mM/ms
    I_tr      = (CaUp - CaRel) / tau_tr                 # mM/ms

    # Aggiornamento gating con Rush-Larsen
    m_inf,   tau_m   = m_gate(V);    m   = rush_larsen(m,   m_inf,   tau_m,   dt_ms)
    h_inf,   tau_h   = h_gate(V);    h   = rush_larsen(h,   h_inf,   tau_h,   dt_ms)
    j_inf,   tau_j   = j_gate(V);    j   = rush_larsen(j,   j_inf,   tau_j,   dt_ms)
    oa_inf,  tau_oa  = oa_gate(V);   oa  = rush_larsen(oa,  oa_inf,  tau_oa,  dt_ms)
    oi_inf,  tau_oi  = oi_gate(V);   oi  = rush_larsen(oi,  oi_inf,  tau_oi,  dt_ms)
    ua_inf,  tau_ua  = ua_gate(V);   ua  = rush_larsen(ua,  ua_inf,  tau_ua,  dt_ms)
    ui_inf,  tau_ui  = ui_gate(V);   ui  = rush_larsen(ui,  ui_inf,  tau_ui,  dt_ms)
    xr_inf,  tau_xr  = xr_gate(V);   xr  = rush_larsen(xr,  xr_inf,  tau_xr,  dt_ms)
    xs_inf,  tau_xs  = xs_gate(V);   xs  = rush_larsen(xs,  xs_inf,  tau_xs,  dt_ms)
    d_inf,   tau_d   = d_gate(V);    d   = rush_larsen(d,   d_inf,   tau_d,   dt_ms)
    f_inf,   tau_f   = f_gate(V);    f   = rush_larsen(f,   f_inf,   tau_f,   dt_ms)
    fCa_inf, tau_fCa = fCa_gate(Cai);fCa = rush_larsen(fCa, fCa_inf, tau_fCa, dt_ms)
    u_inf,   tau_u   = u_gate(Fn);   u   = rush_larsen(u,   u_inf,   tau_u,   dt_ms)
    v_inf,   tau_v   = v_gate(Fn);   v   = rush_larsen(v,   v_inf,   tau_v,   dt_ms)
    w_inf,   tau_w   = w_gate(V);    w   = rush_larsen(w,   w_inf,   tau_w,   dt_ms)

    # Equazioni differenziali (Eulero)
    dVdt  = -(I_ion + I_stim)
    V_new = V + dt_ms * dVdt

    dNaidt  = (-3*INaK - (3*INaCa + IbNa + INa)) * Cm * 1e-6 / (V_i * F)
    Nai_new = Nai + dt_ms * dNaidt

    dKidt  = (2*INaK - (IK1 + Ito + IKur + IKr + IKs + I_stim)) * Cm * 1e-6 / (V_i * F)
    Ki_new = Ki + dt_ms * dKidt

    B1 = ((2*INaCa - (IpCa + ICaL + IbCa)) * Cm * 1e-6 / (2 * V_i * F)
          + (V_up * (I_up_leak - I_up) + I_rel * V_rel) / V_i)
    B2 = (1 + TRPN_max * Km_TRPN / (Cai + Km_TRPN)**2
          + CMDN_max * Km_CMDN / (Cai + Km_CMDN)**2)
    dCaidt  = B1 / B2
    Cai_new = Cai + dt_ms * dCaidt

    dCaUpdt  = I_up - (I_up_leak + I_tr * V_rel / V_up)
    CaUp_new = CaUp + dt_ms * dCaUpdt

    buffer_csqn = 1 + CSQN_max * Km_CSQN / (CaRel + Km_CSQN)**2
    dCaReldt    = (I_tr - I_rel) / buffer_csqn
    CaRel_new   = CaRel + dt_ms * dCaReldt

    Y_new = np.array([V_new, Nai_new, Ki_new, Cai_new, CaUp_new, CaRel_new,
                      m, h, j, oa, oi, ua, ui, xr, xs, d, f, fCa, u, v, w],
                     dtype=float)

    # Dizionari per salvare le grandezze di interesse
    ion_currents = {
        'INa': INa, 'IK1': IK1, 'Ito': Ito, 'IKur': IKur,
        'IKr': IKr, 'IKs': IKs, 'ICaL': ICaL,
        'INaK': INaK, 'INaCa': INaCa, 'IbNa': IbNa, 'IbCa': IbCa, 'IpCa': IpCa
    }
    sr_currents = {'I_rel': I_rel, 'I_up': I_up, 'I_tr': I_tr}
    conc = {'Nai': Nai_new, 'Ki': Ki_new, 'Cai': Cai_new,
            'CaUp': CaUp_new, 'CaRel': CaRel_new}

    return Y_new, ion_currents, sr_currents, conc

# ============================================================================
# 5. CONDIZIONI INIZIALI
# ============================================================================
Y0 = np.array([
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
# 6. SIMULAZIONE CON REGISTRAZIONE DI TUTTE LE GRANDEZZE
# ============================================================================
print("Simulazione Courtemanche 1998 — unità uniformi ms / mV / mM")
print(f"  n_stim      = {n_stim}")
print(f"  stim_period = {stim_period:.0f} ms")
print(f"  stim_start  = {stim_start:.0f} ms")
print(f"  Tmax_ms     = {Tmax_ms:.0f} ms")
print(f"  dt_ms       = {dt_ms} ms  →  {nt} passi")

Y = Y0.copy()
record_V   = np.empty(nt)
time_array = np.empty(nt)

# Inizializza i dizionari per la registrazione (liste)
ion_records = {key: [] for key in ['INa','IK1','Ito','IKur','IKr','IKs','ICaL','INaK','INaCa','IbNa','IbCa','IpCa']}
sr_records  = {key: [] for key in ['I_rel','I_up','I_tr']}
conc_records = {key: [] for key in ['Nai','Ki','Cai','CaUp','CaRel']}

for n in range(nt):
    t_ms = n * dt_ms
    Y, ion_curr, sr_curr, conc = step_courtemanche(Y, t_ms, dt_ms)

    record_V[n]   = Y[0]
    time_array[n] = t_ms

    for k in ion_records:
        ion_records[k].append(ion_curr[k])
    for k in sr_records:
        sr_records[k].append(sr_curr[k])
    for k in conc_records:
        conc_records[k].append(conc[k])

    if n % (nt // 10) == 0:
        print(f"  {100*n/nt:3.0f}% — t = {t_ms:8.1f} ms | V = {Y[0]:6.1f} mV")

print("Simulazione completata!")

# Converti le liste in array numpy per i plot
for k in ion_records:
    ion_records[k] = np.array(ion_records[k])
for k in sr_records:
    sr_records[k] = np.array(sr_records[k])
for k in conc_records:
    conc_records[k] = np.array(conc_records[k])

# ============================================================================
# 7. GRAFICI
# ============================================================================

# --- Grafico 1: segnale completo ---
plt.figure(figsize=(8, 5))
plt.plot(time_array, record_V, 'b-', linewidth=0.8)
plt.xlabel("Tempo (ms)", fontsize=12)
plt.ylabel("V (mV)", fontsize=12)
plt.title(f"Courtemanche 1998 — {n_stim} potenziali d'azione atriali", fontsize=12)
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
plt.title(f"Ultimo AP — stimolo n°{n_stim}  |  t₀ = {last_stim_time:.0f} ms", fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()