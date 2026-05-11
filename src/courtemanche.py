from . import parameters as p
import numpy as np

# ============================================================================
# 2. FUNZIONI DI SUPPORTO
# ============================================================================

def E_K(Ki):
    return p.RTF * np.log(p.Ko / Ki)

def E_Na(Nai):
    return p.RTF * np.log(p.Nao / Nai)

def E_Ca(Cai):
    return 0.5 * p.RTF * np.log(p.Cao / Cai)

def stimulus_current(t_ms):
    t_in_cycle = t_ms % p.stim_period
    if p.stim_start <= t_in_cycle < p.stim_start + p.stim_duration:
        return p.stim_amplitude / p.Cm
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
    tau_ms = 1.0 / ((alpha + beta) * p.KQ10)
    return inf, tau_ms

def oi_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 43.1) / 5.3))
    alpha  = 1.0 / (18.53 + np.exp((V + 113.7) / 10.95))
    beta   = 1.0 / (35.56 + np.exp((V + 1.26) / -7.44))
    tau_ms = 1.0 / ((alpha + beta) * p.KQ10)
    return inf, tau_ms

def ua_gate(V):
    inf    = 1.0 / (1 + np.exp((V + 30.3) / -9.6))
    alpha  = 0.65 / (np.exp((V + 10.0) / -8.5) + np.exp((V - 30.0) / -59.0))
    beta   = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
    tau_ms = 1.0 / ((alpha + beta) * p.KQ10)
    return inf, tau_ms

def ui_gate(V):
    inf    = 1.0 / (1 + np.exp((V - 99.45) / 27.48))
    alpha  = 1.0 / (21.0 + np.exp((V - 185.0) / -28.0))
    beta   = 1.0 / np.exp((V - 158.0) / -16.0)
    tau_ms = 1.0 / ((alpha + beta) * p.KQ10)
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
    inf    = 1.0 / (1 + np.exp(-(Fn - p.c1_rel) / p.c2_rel))
    tau_ms = 8.0
    return inf, tau_ms

def v_gate(Fn):
    inf    = 1 - 1.0 / (1 + np.exp(-(Fn - 0.2 * p.c1_rel) / p.c2_rel))
    tau_ms = 1.91 + 2.09 / (1 + np.exp(-(Fn - p.c1_rel) / p.c2_rel))
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
    return p.gKur_base * (1 + 10.0 / (1 + np.exp((V - 15.0) / -13.0)))

# ============================================================================
# 4. PASSO DI INTEGRAZIONE
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
    INa   = p.gNa * m**3 * h * j * (V - ENa)
    IK1   = p.gK1 * (V - EK) / (1 + np.exp(0.07 * (V + 80.0)))
    Ito   = p.gto * oa**3 * oi * (V - EK)
    IKur  = gKur(V) * ua**3 * ui * (V - EK)
    IKr   = p.gKr * xr * (V - EK) / (1 + np.exp((V + 15.0) / 22.4))
    IKs   = p.gKs * xs**2 * (V - EK)
    ICaL  = p.gCaL * d * f * fCa * (V - ECa)

    sigma = (np.exp(p.Nao / 67.3) - 1) / 7.0
    fNaK  = 1.0 / (1 + 0.1245 * np.exp(-0.1 * V * p.FRT) + 0.0365 * sigma * np.exp(-V * p.FRT))
    INaK  = p.INaK_max * fNaK * p.Ko / (p.Ko + p.KmKo) / (1 + (p.KmNai / Nai)**1.5)

    INaCa = p.INaCa_max * (
        np.exp(p.gamma_naca * V * p.FRT) * Nai**3 * p.Cao
        - np.exp((p.gamma_naca - 1) * V * p.FRT) * p.Nao**3 * Cai
    ) / (
        (p.KmNa_naca**3 + p.Nao**3) * (p.KmCa_naca + p.Cao)
        * (1 + p.ksat * np.exp((p.gamma_naca - 1) * V * p.FRT))
    )

    IbNa  = p.gbNa * (V - ENa)
    IbCa  = p.gbCa * (V - ECa)
    IpCa  = p.IpCa_max * Cai / (0.0005 + Cai)

    I_stim = stimulus_current(t_ms)

    I_ion = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IpCa + INaK + INaCa + IbNa + IbCa

    # Dinamica del calcio e SR
    Fn = (1e-12 * p.V_rel * p.K_rel * u**2 * v * w * (CaRel - Cai)
          - 5e-13 / p.F * (0.5 * ICaL - 0.2 * INaCa) * p.Cm)

    I_rel     = p.K_rel * u**2 * v * w * (CaRel - Cai)    # mM/ms
    I_up      = p.I_up_max / (1 + p.K_up / Cai)             # mM/ms
    I_up_leak = p.I_up_max * CaUp / p.Ca_up_max             # mM/ms
    I_tr      = (CaUp - CaRel) / p.tau_tr                 # mM/ms

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

    dNaidt  = (-3*INaK - (3*INaCa + IbNa + INa)) * p.Cm * 1e-6 / (p.V_i * p.F)
    Nai_new = Nai + dt_ms * dNaidt

    dKidt  = (2*INaK - (IK1 + Ito + IKur + IKr + IKs + I_stim)) * p.Cm * 1e-6 / (p.V_i * p.F)
    Ki_new = Ki + dt_ms * dKidt

    B1 = ((2*INaCa - (IpCa + ICaL + IbCa)) * p.Cm * 1e-6 / (2 * p.V_i * p.F)
          + (p.V_up * (I_up_leak - I_up) + I_rel * p.V_rel) / p.V_i)
    B2 = (1 + p.TRPN_max * p.Km_TRPN / (Cai + p.Km_TRPN)**2
          + p.CMDN_max * p.Km_CMDN / (Cai + p.Km_CMDN)**2)
    dCaidt  = B1 / B2
    Cai_new = Cai + dt_ms * dCaidt

    dCaUpdt  = I_up - (I_up_leak + I_tr * p.V_rel / p.V_up)
    CaUp_new = CaUp + dt_ms * dCaUpdt

    buffer_csqn = 1 + p.CSQN_max * p.Km_CSQN / (CaRel + p.Km_CSQN)**2
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