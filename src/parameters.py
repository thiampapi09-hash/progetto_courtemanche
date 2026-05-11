# ============================================================================
# PARAMETRI FISICI E NUMERICI
# ============================================================================

dt_ms = 0.05   # oppure 0.01 per simulazione più fine

R   = 8.3143
T   = 310.0
F   = 96.4867

RTF = R * T / F
FRT = 1.0 / RTF

Cm     = 100.0
V_cell = 20100.0
V_i    = V_cell * 0.68
V_up   = V_cell * 0.0552
V_rel  = V_cell * 0.0048

Ko  = 5.4
Nao = 140.0
Cao = 1.8

# Conduttanze
gNa = 7.8
gK1 = 0.09
gto = 0.1652
gKur_base = 0.005
gKr = 0.0294
gKs = 0.129
gCaL = 0.1238
gbNa = 0.000674
gbCa = 0.00113

INaK_max  = 0.6
INaCa_max = 1600.0
IpCa_max  = 0.275
I_up_max  = 0.005

KmNai = 10.0
KmKo  = 1.5

gamma_naca = 0.35
KmNa_naca  = 87.5
KmCa_naca  = 1.38
ksat       = 0.1

K_rel = 30.0
K_up  = 0.00092
Ca_up_max = 15.0
tau_tr    = 180.0

CMDN_max = 0.05
TRPN_max = 0.07
CSQN_max = 10.0
Km_CMDN  = 0.00238
Km_TRPN  = 0.0005
Km_CSQN  = 0.8

c1_rel = 3.4175e-13
c2_rel = 13.67e-16

KQ10 = 3.0

# STIMOLO
stim_amplitude = -2000.0
stim_start     = 50.0
stim_duration  = 2.0
stim_period    = 1000.0
n_stim         = 10