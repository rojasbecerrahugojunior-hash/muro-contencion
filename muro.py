# Estructuras de Contención — Streamlit (dashboard con tabs + CSV + modo libro + chequeos)
# Jaky / Rankine / Coulomb; c-φ y grieta; NF; q uniforme y franja; Sísmico (MO).
# Diagramas, métricas, desglose, tabla z–σh y chequeos de estabilidad (FS y presiones).

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
theme = st.get_option("theme.base") or "light"

img_name = (
    "retaining_wall_ref_dark_nooverlap.png"
    if theme == "dark"
    else "retaining_wall_ref_light_nooverlap.png"
)

IMG = ROOT / "assets" / img_name   # <- OJO: ahora apunta a la carpeta assets/

st.image(
    str(IMG),
    use_container_width=True,
    caption="Esquema de parámetros: H, B, b_t, b_h, t_b, t_st, t_sb, β, q, a, b, NF (hw), Ea, Es, μ·N, Pasivo."
)


# ----------------------------- Constantes & helpers -----------------------------
GAMMA_W = 9.81  # kN/m³
DEC2 = dict(step=0.01, format="%.2f")  # inputs con 2 decimales


# Trig en GRADOS
def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


def tand(x):
    return np.tan(np.deg2rad(x))


def atan2d(y, x):
    return np.rad2deg(np.arctan2(y, x))


# Coeficientes estáticos
def K0_jaky(phi_deg: float) -> float:
    return float(1.0 - sind(phi_deg))


def Ka_rankine(phi_deg: float) -> float:
    return float(tand(45.0 - phi_deg / 2.0) ** 2)


def Kp_rankine(phi_deg: float) -> float:
    return float(tand(45.0 + phi_deg / 2.0) ** 2)


# Coulomb (pared vertical, batter=0) con talud β y fricción δ
def Ka_coulomb_vertical(
    phi_deg: float, delta_deg: float, beta_deg: float = 0.0
) -> float:
    num = cosd(phi_deg - beta_deg) ** 2
    den = (
        cosd(beta_deg)
        * cosd(beta_deg + delta_deg)
        * (
            1.0
            + np.sqrt(
                (sind(phi_deg + delta_deg) * sind(phi_deg - beta_deg))
                / (cosd(beta_deg + delta_deg) * cosd(beta_deg))
            )
        )
        ** 2
    )
    return float(num / den)


def Kp_coulomb_vertical(
    phi_deg: float, delta_deg: float, beta_deg: float = 0.0
) -> float:
    num = cosd(phi_deg + beta_deg) ** 2
    den = (
        cosd(beta_deg)
        * cosd(beta_deg - delta_deg)
        * (
            1.0
            - np.sqrt(
                (sind(phi_deg + delta_deg) * sind(phi_deg + beta_deg))
                / (cosd(beta_deg - delta_deg) * cosd(beta_deg))
            )
        )
        ** 2
    )
    return float(num / den)


# Mononobe–Okabe (activo)
def Kae_MO(
    phi_deg: float, beta_deg: float, delta_deg: float, kh: float, kv: float
) -> float:
    theta_deg = atan2d(kh, (1.0 - kv))
    if (phi_deg + beta_deg + theta_deg) >= 90.0:
        return np.nan
    num = cosd(phi_deg - beta_deg - theta_deg) ** 2
    den = (
        cosd(theta_deg)
        * cosd(beta_deg)
        * cosd(beta_deg + delta_deg + theta_deg)
        * (
            1.0
            + np.sqrt(
                (sind(phi_deg + delta_deg) * sind(phi_deg - beta_deg - theta_deg))
                / (cosd(beta_deg + delta_deg + theta_deg) * cosd(beta_deg))
            )
        )
        ** 2
    )
    return float(num / den)


# Sobrecarga de franja (2D)
def vertical_stress_strip_at_wall(z, a: float, b: float, q: float):
    """Δσv = (q/π)[atan((a+b)/z) - atan(a/z)]"""
    z = np.maximum(np.array(z, dtype=float), 1e-6)  # evita z=0
    return (q / np.pi) * (np.arctan((a + b) / z) - np.arctan(a / z))


# ----------------------------- UI — Sidebar -----------------------------
with st.sidebar:
    st.header("Geometría & Método")
    H = st.number_input(
        "Altura del muro H [m]",
        value=6.00,
        min_value=0.50,
        max_value=60.00,
        key="H_key",
        **DEC2,
    )
    condicion = st.selectbox(
        "Condición", ["Reposo (K₀)", "Activa (Kₐ)", "Pasiva (Kₚ)"], key="cond_key"
    )
    metodo = st.selectbox("Método", ["Rankine", "Coulomb"], key="meth_key")
    beta = st.number_input(
        "β (talud del relleno) [°] (0 = horizontal)",
        value=0.00,
        min_value=0.00,
        max_value=30.00,
        key="beta_key",
        **DEC2,
    )

    st.divider()
    st.header("Controles de caso")
    use_cphi = st.checkbox(
        "Usar cohesión (c–φ) en Rankine Activa", value=True, key="use_cphi_key"
    )
    apply_tc = st.checkbox(
        "Aplicar grieta de tensión (truncar σh<0)", value=True, key="apply_tc_key"
    )
    crack_water = st.checkbox(
        "Agua llenando la grieta (si existe)", value=False, key="crack_water_key"
    )
    show_before = st.checkbox("Mostrar métricas ANTES (con tracción)", value=True)
    modo_libro = st.checkbox(
        "Modo libro (√Ka=0.625 en Rankine Activa)", value=False, key="modo_libro_key"
    )

    st.divider()
    st.header("Estratos (de arriba hacia abajo)")
    st.caption("φ y δ en grados. c en kN/m² (efectiva). γ y γ_sat en kN/m³.")
    default_layers = [
        {
            "z_inf": 6.00,
            "gamma": 17.40,
            "gamma_sat": 19.00,
            "phi": 26.00,
            "delta": 0.00,
            "c": 14.36,
        },
    ]
    layers = st.data_editor(
        default_layers,
        num_rows="dynamic",
        use_container_width=True,
        key="layers_key",
        column_config={
            "z_inf": st.column_config.NumberColumn(
                "Profundidad z_inferior [m]", step=0.01, min_value=0.00, format="%.2f"
            ),
            "gamma": st.column_config.NumberColumn(
                "γ (seco) [kN/m³]", step=0.01, min_value=5.00, format="%.2f"
            ),
            "gamma_sat": st.column_config.NumberColumn(
                "γ_sat [kN/m³]", step=0.01, min_value=5.00, format="%.2f"
            ),
            "phi": st.column_config.NumberColumn(
                "φ [°]", step=0.01, min_value=0.00, max_value=55.00, format="%.2f"
            ),
            "delta": st.column_config.NumberColumn(
                "δ [°]", step=0.01, min_value=0.00, max_value=40.00, format="%.2f"
            ),
            "c": st.column_config.NumberColumn(
                "c' [kN/m²]", step=0.01, min_value=0.00, format="%.2f"
            ),
        },
    )

    st.header("Nivel freático")
    nf_debajo = st.checkbox("NF por debajo de la base", value=True, key="nf_debajo_key")
    if nf_debajo:
        hw = H + 1.00  # evita hidrostática
    else:
        hw = st.number_input(
            "Profundidad del NF desde la coronación [m]",
            value=min(H, 3.00),
            min_value=0.00,
            max_value=1000.00,
            key="hw_key",
            **DEC2,
        )
    drenaje_ok = st.checkbox(
        "Drenaje eficaz (sin presión hidrostática contra el muro)",
        value=True,
        key="drenaje_ok_key",
    )

    st.header("Sobrecargas")
    use_q_uni = st.checkbox("Incluir q uniforme", value=False, key="use_q_uni_key")
    q_uniforme = st.number_input(
        "q uniforme [kN/m²]",
        value=0.00,
        min_value=0.00,
        max_value=5000.00,
        key="q_uniforme_key",
        **DEC2,
    )
    use_strip = st.checkbox("Incluir franja finita", value=False, key="use_strip_key")
    a = st.number_input(
        "Distancia a la pared a [m]",
        value=0.00,
        min_value=0.00,
        max_value=60.00,
        key="a_key",
        **DEC2,
    )
    b = st.number_input(
        "Ancho de franja b [m]",
        value=0.00,
        min_value=0.00,
        max_value=60.00,
        key="b_key",
        **DEC2,
    )
    q_strip = st.number_input(
        "q de franja [kN/m²]",
        value=0.00,
        min_value=0.00,
        max_value=5000.00,
        key="q_strip_key",
        **DEC2,
    )

    st.header("Sísmico (Mononobe–Okabe)")
    sismo = st.checkbox(
        "Activar análisis pseudo-estático", value=False, key="sismo_key"
    )
    kh = st.number_input(
        "k_h", value=0.20, min_value=0.00, max_value=0.50, key="kh_key", **DEC2
    )
    kv = st.number_input(
        "k_v", value=0.00, min_value=-0.30, max_value=0.30, key="kv_key", **DEC2
    )

    st.divider()
    st.header("Precisión & tamaño")
    dec_out = st.slider(
        "Decimales en resultados",
        2,
        6,
        st.session_state.get("dec_out_key", 4),
        1,
        key="dec_out_key",
    )
    nz = st.slider(
        "Resolución vertical (n puntos)",
        800,
        6000,
        st.session_state.get("nz_key", 2800),
        200,
        key="nz_key",
    )
    w_px = st.slider(
        "Ancho gráfico [px]",
        360,
        700,
        st.session_state.get("w_px_key", 420),
        10,
        key="w_px_key",
    )
    h_px = st.slider(
        "Alto gráfico [px]",
        320,
        600,
        st.session_state.get("h_px_key", 360),
        10,
        key="h_px_key",
    )
    fmt = f"{{:.{dec_out}f}}"

# ----------------------------- Preprocesado de capas -----------------------------
layers_sorted = []
z_top = 0.0
for row in layers:
    z_inf = float(row["z_inf"])
    if z_inf <= z_top:
        continue
    z_inf = min(z_inf, H)
    layers_sorted.append(
        {
            "z_sup": z_top,
            "z_inf": z_inf,
            "gamma": float(row["gamma"]),
            "gamma_sat": float(row["gamma_sat"]),
            "phi": float(row["phi"]),
            "delta": float(row["delta"]),
            "c": float(row["c"]),
        }
    )
    z_top = z_inf
if len(layers_sorted) == 0 or layers_sorted[-1]["z_inf"] < H:
    layers_sorted.append(
        {
            "z_sup": z_top,
            "z_inf": H,
            "gamma": 18.0,
            "gamma_sat": 20.0,
            "phi": 30.0,
            "delta": 0.0,
            "c": 0.0,
        }
    )


def K_from(cond, meth, phi, delta, beta):
    if cond == "Reposo (K₀)":
        return K0_jaky(phi)
    if meth == "Rankine":
        return Ka_rankine(phi) if cond == "Activa (Kₐ)" else Kp_rankine(phi)
    Kc = (
        Ka_coulomb_vertical(phi, delta, beta)
        if cond == "Activa (Kₐ)"
        else Kp_coulomb_vertical(phi, delta, beta)
    )
    if not np.isfinite(Kc) or Kc <= 0:
        return Ka_rankine(phi) if cond == "Activa (Kₐ)" else Kp_rankine(phi)
    return Kc


# ----------------------------- Perfiles por profundidad -----------------------------
def build_profiles(
    H,
    layers,
    hw,
    condicion,
    metodo,
    beta,
    use_cphi,
    apply_tc,
    nz=1200,
    modo_libro=False,
):
    """Devuelve z, K(z), σ'v(z), σh_before, σh_after, z_tc."""
    z = np.linspace(0.0, H, nz)
    dz = z[1] - z[0]
    gamma_eff = np.zeros_like(z)
    phi_z = np.zeros_like(z)
    delta_z = np.zeros_like(z)
    c_z = np.zeros_like(z)

    for L in layers:
        mask = (z >= L["z_sup"]) & (z <= L["z_inf"])
        mask_dry = mask & (z <= hw)
        mask_sub = mask & (z > hw)
        gamma_eff[mask_dry] = L["gamma"]
        gamma_eff[mask_sub] = max(0.0, L["gamma_sat"] - GAMMA_W)
        phi_z[mask] = L["phi"]
        delta_z[mask] = L["delta"]
        c_z[mask] = L["c"]

    sigma_v = np.cumsum(gamma_eff) * dz
    vec_K = np.vectorize(
        lambda ph, de: K_from(condicion, metodo, ph, de, beta), otypes=[float]
    )
    Kz = vec_K(phi_z, delta_z)
    Kz = np.nan_to_num(Kz, nan=0.0, posinf=0.0, neginf=0.0)

    # MODO LIBRO (solo Rankine Activa)
    if modo_libro and (condicion == "Activa (Kₐ)") and (metodo == "Rankine"):
        Kz[:] = (0.625) ** 2

    sigma_h = Kz * sigma_v  # base sin cohesión
    if use_cphi and condicion == "Activa (Kₐ)" and metodo == "Rankine":
        sqrtK = np.sqrt(np.clip(Kz, 0.0, None))
        sigma_h = sigma_h - 2.0 * c_z * sqrtK

    sigma_h_before = np.nan_to_num(sigma_h, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_h_after = (
        np.maximum(sigma_h_before, 0.0) if apply_tc else sigma_h_before.copy()
    )

    z_tc = np.nan
    if apply_tc:
        nonneg = np.where(sigma_h_before >= 0.0)[0]
        if nonneg.size > 0:
            z_tc = float(z[nonneg[0]])

    return z, Kz, sigma_v, sigma_h_before, sigma_h_after, z_tc


z, Kz, sigma_v, sigma_h_soil_before, sigma_h_soil_after, z_tc = build_profiles(
    H,
    layers_sorted,
    hw,
    condicion,
    metodo,
    beta,
    use_cphi,
    apply_tc,
    nz=nz,
    modo_libro=modo_libro,
)

# ----------------------------- Cargas adicionales y fuerzas -----------------------------
# K para cargas superficiales coherente con "modo libro"
if modo_libro and (condicion == "Activa (Kₐ)") and (metodo == "Rankine"):
    K_surf = (0.625) ** 2
else:
    K_surf = K_from(
        condicion, metodo, layers_sorted[0]["phi"], layers_sorted[0]["delta"], beta
    )

sigma_h_q = np.zeros_like(z)
if use_q_uni and q_uniforme > 0:
    sigma_h_q = np.ones_like(z) * (K_surf * q_uniforme)

sigma_h_strip = np.zeros_like(z)
if use_strip and (b > 0.0) and (q_strip > 0.0):
    sigma_h_strip = K_surf * vertical_stress_strip_at_wall(z, a, b, q_strip)

# Hidrostática (si NO hay drenaje) — actúa directo
sigma_h_w = np.zeros_like(z)
if (not drenaje_ok) and (hw < H):
    mask_w = z >= hw
    sigma_h_w[mask_w] = GAMMA_W * (z[mask_w] - hw)

# Agua llenando la grieta (si la hay)
sigma_h_crack = np.zeros_like(z)
if apply_tc and (not np.isnan(z_tc)) and crack_water and z_tc > 0.0:
    mask_c = z <= z_tc
    sigma_h_crack[mask_c] = GAMMA_W * z[mask_c]


def trapz_force_and_ybar(z, sigma, H):
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    F = float(np.trapz(sigma, z))
    M = float(np.trapz(sigma * (H - z), z))
    ybar = (M / F) if F > 0 else np.nan
    return F, ybar


def _force_only(z, sigma):
    return float(np.trapz(np.nan_to_num(sigma, nan=0.0), z))


# Componentes estáticos
P_soil_before = _force_only(z, sigma_h_soil_before)
P_soil_after = _force_only(z, sigma_h_soil_after)
P_q = _force_only(z, sigma_h_q)
P_strip = _force_only(z, sigma_h_strip)
P_w = _force_only(z, sigma_h_w)
P_crack = _force_only(z, sigma_h_crack)

# Totales estáticos
sigma_total_before = sigma_h_soil_before + sigma_h_q + sigma_h_strip + sigma_h_w
sigma_total_after = (
    sigma_h_soil_after + sigma_h_q + sigma_h_strip + sigma_h_w + sigma_h_crack
)
P_before, yb_before = trapz_force_and_ybar(z, sigma_total_before, H)
P_after, yb_after = trapz_force_and_ybar(z, sigma_total_after, H)

# Sísmico (MO)
P_seis = 0.0
yb_seis = np.nan
if sismo:
    phi_eq = layers_sorted[0]["phi"]
    delta_eq = layers_sorted[0]["delta"]
    bulk_sum = 0.0
    for L in layers_sorted:
        z1, z2 = L["z_sup"], L["z_inf"]
        zs1, zs2 = z1, min(z2, hw)
        if zs2 > zs1:
            bulk_sum += L["gamma"] * (zs2 - zs1)
        zu1, zu2 = max(z1, hw), z2
        if zu2 > zu1:
            bulk_sum += L["gamma_sat"] * (zu2 - zu1)
    gamma_avg_bulk = bulk_sum / H if H > 0 else 0.0

    Kae = Kae_MO(phi_eq, beta, delta_eq, kh, kv)
    if np.isnan(Kae):
        st.warning(
            "⚠️ Combinación sísmica inválida: φ + β + θ ≥ 90°. No se suma componente sísmica."
        )
    else:
        P_soil_MO = 0.5 * (1.0 - kv) * gamma_avg_bulk * (H**2) * Kae
        P_q_MO = Kae * (q_uniforme if (use_q_uni and q_uniforme > 0) else 0.0) * H
        P_seis = float(P_soil_MO + P_q_MO)
        yb_seis = 0.6 * H

# Totales finales
P_total = P_after + P_seis
M_total = P_after * (yb_after if np.isfinite(yb_after) else 0.0) + (
    P_seis * (yb_seis if np.isfinite(yb_seis) else 0.0)
)
yb_total = (M_total / P_total) if P_total > 0 else np.nan

# ----------------------------- SALIDA — Tabs (empujes) -----------------------------
st.subheader(f"{metodo} — {condicion}")

chip_cols = st.columns(3)
chip_cols[0].markdown(
    f"✅ **Cohesión c–φ:** {'ON' if (use_cphi and metodo=='Rankine' and condicion=='Activa (Kₐ)') else 'OFF'}"
)
chip_cols[1].markdown(
    f"🧪 **Modo libro:** {'ON (√Ka=0.625)' if modo_libro and metodo=='Rankine' and condicion=='Activa (Kₐ)' else 'OFF'}"
)
chip_cols[2].markdown(
    f"💧 **Drenaje eficaz:** {'ON (sin agua)' if drenaje_ok else 'OFF (con agua)'}"
)

hay_grieta = apply_tc and np.isfinite(z_tc) and (0.0 < z_tc < H)
if hay_grieta:
    st.success(f"Grieta de tensión: z_tc ≈ {fmt.format(z_tc)} m desde la coronación.")
else:
    st.info("No hay grieta de tensión en las condiciones actuales.")

tab_res, tab_diag, tab_tabla = st.tabs(["📊 Resumen", "📈 Diagramas", "🧮 Tabla z–σh"])

with tab_res:
    if show_before:
        st.markdown("#### Caso ANTES (incluye tracción)")
        cba1, cba2 = st.columns(2)
        cba1.metric("P ANTES [kN/m]", fmt.format(P_before))
        cba2.metric(
            "ȳ ANTES [m desde base]",
            fmt.format(0.0 if not np.isfinite(yb_before) else yb_before),
        )
        st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("P ESTÁTICO (DESPUÉS) [kN/m]", fmt.format(P_after))
    c2.metric("P SÍSMICO (MO) [kN/m]", fmt.format(P_seis))
    c3.metric("P TOTAL [kN/m]", fmt.format(P_total))

    c4, c5 = st.columns(2)
    c4.metric(
        "ȳ ESTÁTICO (DESPUÉS) [m desde base]",
        fmt.format(0.0 if not np.isfinite(yb_after) else yb_after),
    )
    c5.metric(
        "ȳ TOTAL [m desde base]",
        fmt.format(0.0 if not np.isfinite(yb_total) else yb_total),
    )

    st.caption(
        "“DESPUÉS” = suelo sin tracción (si aplica) + q + franja + NF (+ agua si Drenaje=OFF)."
    )

    fig0, ax0 = plt.subplots(figsize=(w_px / 96.0, (h_px - 40) / 96.0), dpi=120)
    ax0.bar(["Estático (DESPUÉS)", "Sísmico", "TOTAL"], [P_after, P_seis, P_total])
    ax0.set_ylabel("kN/m")
    ax0.set_title("Resultantes")
    fig0.tight_layout()
    st.pyplot(fig0, use_container_width=False)

    figc, axc = plt.subplots(figsize=(w_px / 96.0, h_px / 96.0), dpi=120)
    comp_vals = [P_soil_after, P_q, P_strip, P_w, P_crack]
    comp_lbls = ["Suelo", "q uniforme", "Franja", "Agua (NF)", "Agua en grieta"]
    axc.barh(comp_lbls, comp_vals)
    axc.set_xlabel("kN/m")
    axc.set_title("Desglose (estático DESPUÉS)")
    figc.tight_layout()
    st.pyplot(figc, use_container_width=False)

with tab_diag:
    xmax = float(
        np.nanmax([np.nanmax(sigma_total_before), np.nanmax(sigma_total_after)])
    )
    xmax = 1.05 * xmax if xmax > 0 else 10.0

    fig, ax = plt.subplots(figsize=(w_px / 96.0, h_px / 96.0), dpi=120)
    ax.plot(sigma_total_before, z, linewidth=2, label="ANTES")
    ax.plot(sigma_total_after, z, linewidth=2, linestyle="--", label="DESPUÉS")
    if hay_grieta:
        ax.axhline(z_tc, linestyle=":", linewidth=1)
        ax.text(0.02 * xmax, z_tc + 0.03 * H, f"z_tc ≈ {z_tc:.2f} m", va="bottom")
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("σh [kPa ≈ kN/m²]")
    ax.set_ylabel("Profundidad z [m]")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()

    colL, colR = st.columns([2, 1])
    with colL:
        st.pyplot(fig, use_container_width=False)

    z_metros = np.arange(0.0, H + 0.0001, 1.0)

    def _interp(arr):
        return np.interp(z_metros, z, np.nan_to_num(arr, nan=0.0))

    df_side = pd.DataFrame(
        {
            "z [m] (corona)": z_metros,
            "y [m] (base)": np.round(H - z_metros, dec_out),
            "K": np.interp(z_metros, z, Kz),
            "σ'v [kPa]": np.interp(z_metros, z, sigma_v),
            "σh ANTES [kPa]": _interp(sigma_total_before),
            "σh DESPUÉS [kPa]": _interp(sigma_total_after),
        }
    ).round(dec_out)
    with colR:
        st.dataframe(df_side, use_container_width=True, height=int(h_px))

with tab_tabla:
    z_metros = np.arange(0.0, H + 0.0001, 1.0)

    def _interp(arr):
        return np.interp(z_metros, z, np.nan_to_num(arr, nan=0.0))

    df = pd.DataFrame(
        {
            "z [m] (corona)": z_metros,
            "y [m] (base)": H - z_metros,
            "K": np.interp(z_metros, z, Kz),
            "σ'v [kPa]": np.interp(z_metros, z, sigma_v),
            "σh suelo (ANTES) [kPa]": _interp(sigma_h_soil_before),
            "σh suelo (DESPUÉS) [kPa]": _interp(sigma_h_soil_after),
            "σh agua NF [kPa]": _interp(sigma_h_w),
            "σh agua grieta [kPa]": _interp(sigma_h_crack),
            "σh q uniforme [kPa]": _interp(sigma_h_q),
            "σh franja [kPa]": _interp(sigma_h_strip),
            "σh TOTAL ANTES [kPa]": _interp(sigma_total_before),
            "σh TOTAL DESPUÉS [kPa]": _interp(sigma_total_after),
        }
    ).round(dec_out)
    st.dataframe(df, use_container_width=True, height=560)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar tabla z–σh (CSV)",
        data=csv_bytes,
        file_name="tabla_z_sigmah.csv",
        mime="text/csv",
    )

# ============================= CHEQUEOS GEOTÉCNICOS / ESTRUCTURALES =============================
st.divider()
st.subheader("🧱 Estabilidad y Presiones en Cimentación")

with st.sidebar:
    st.header("Geometría del muro (para estabilidad)")
    st.caption("Unidades en m y kN/m³. Concreto 24 kN/m³ por defecto.")
    B = st.number_input(
        "Ancho total de base B", value=4.60, min_value=0.50, step=0.01, key="geom_B"
    )
    bt = st.number_input(
        "Ancho de puntera (toe) b_t",
        value=1.30,
        min_value=0.00,
        step=0.01,
        key="geom_bt",
    )
    t_b = st.number_input(
        "Espesor de losa de base t_b",
        value=0.70,
        min_value=0.10,
        step=0.01,
        key="geom_tb",
    )

    st.caption("Alma (muro-cantilever) — espesor variable lineal")
    t_st = st.number_input(
        "Espesor del alma en coronación t_st",
        value=0.40,
        min_value=0.10,
        step=0.01,
        key="geom_tst",
    )
    t_sb = st.number_input(
        "Espesor del alma en la base t_sb",
        value=0.80,
        min_value=0.20,
        step=0.01,
        key="geom_tsb",
    )

    bh = st.number_input(
        "Ancho de talón (heel) b_h",
        value=max(0.10, 2.50),
        min_value=0.00,
        step=0.01,
        key="geom_bh",
    )
    gamma_c = st.number_input(
        "γ del concreto [kN/m³]",
        value=24.00,
        min_value=20.0,
        max_value=27.0,
        step=0.1,
        key="gamma_c_key",
    )

    st.header("Fricción y capacidad de apoyo")
    mu_base = st.number_input(
        "μ base (≈ tan φ_b)",
        value=0.50,
        min_value=0.00,
        max_value=1.20,
        step=0.01,
        key="mu_base_key",
    )
    include_passive = st.checkbox(
        "Incluir pasivo en frente de puntera", value=False, key="include_passive_key"
    )
    Kp_front = st.number_input(
        "Kp frente (si aplica)",
        value=3.00,
        min_value=0.00,
        step=0.1,
        key="Kp_front_key",
    )
    z_front = st.number_input(
        "Altura de suelo confinado frente a puntera [m]",
        value=0.50,
        min_value=0.00,
        step=0.05,
        key="z_front_key",
    )
    red_passive = st.number_input(
        "Reducción pasivo (%)",
        value=50,
        min_value=0,
        max_value=100,
        step=5,
        key="red_passive_key",
    )
    qadm = st.number_input(
        "q admisible del suelo [kPa]",
        value=300.0,
        min_value=50.0,
        step=5.0,
        key="qadm_key",
    )

# --- pesos (por metro) ---
# Losa de base (rectángulo B x t_b)
W_base = gamma_c * B * t_b
xW_base = B / 2.0  # centroide desde la puntera

# Alma (trapecio vertical: espesor t_st en cima y t_sb en base; altura H)
W_stem = gamma_c * 0.5 * (t_st + t_sb) * H
# Centroide del alma desde la cara de PUNTERA (toe). El alma se apoya sobre la losa, bordeado por la puntera bt:
xW_stem = bt + t_sb / 2.0

# Peso del suelo sobre el talón (usa σ'v(H) integrado por capas)
W_soil_heel = float(sigma_v[-1]) * bh
xW_soil_heel = bt + t_sb + bh / 2.0

# Peso del suelo sobre la puntera (si hay relleno delante; aquí 0 por defecto)
W_soil_toe = 0.0
xW_soil_toe = bt / 2.0

# Suma de verticales
N_static = W_base + W_stem + W_soil_heel + W_soil_toe

# --- fuerzas horizontales ---
P_H = P_total
a_P = yb_total if np.isfinite(yb_total) else (H / 3.0)  # brazo desde la base

# --- momentos respecto al borde de la PUNTERA (toe) ---
# Convención: estabilizadores (pesos) positivos; volcamiento por P_H negativo
M_res = (
    W_base * xW_base
    + W_stem * xW_stem
    + W_soil_heel * xW_soil_heel
    + W_soil_toe * xW_soil_toe
)
M_ot = P_H * a_P
FS_volteo = (M_res / M_ot) if M_ot > 0 else np.nan

# --- deslizamiento ---
P_passive = 0.0
if include_passive and (Kp_front > 0.0) and (z_front > 0.0):
    gamma_front = layers_sorted[0]["gamma"]  # capa superior (sin NF)
    P_passive = 0.5 * Kp_front * gamma_front * (z_front**2)
    P_passive *= 1.0 - red_passive / 100.0

R_fric = mu_base * N_static
FS_desl = (R_fric + P_passive) / P_H if P_H > 0 else np.nan

# --- resultante y presiones en la base ---
# Posición de la resultante desde la puntera:
x_R = (M_res - M_ot) / N_static if N_static > 0 else np.nan
# Excentricidad respecto al centro de la base (positiva hacia el talón):
e = (x_R - B / 2.0) if np.isfinite(x_R) else np.nan

# Presiones bajo la base (lineales). q_toe/heel en kPa.
q_med = N_static / B if B > 0 else np.nan
if np.isfinite(e):
    q_toe = q_med * (1.0 + 6.0 * (e / B))
    q_heel = q_med * (1.0 - 6.0 * (e / B))
    q_max = max(q_toe, q_heel)
    q_min = min(q_toe, q_heel)
else:
    q_toe = q_heel = q_max = q_min = np.nan

no_traccion = (q_min >= -1e-6) if np.isfinite(q_min) else False
dentro_qadm = (q_max <= qadm + 1e-6) if np.isfinite(q_max) else False
tercio_medio = (abs(e) <= B / 6.0) if np.isfinite(e) else False

# --- OUTPUT (chequeos) ---
tab_res2, tab_press = st.tabs(["✅ Chequeos", "📐 Detalle & brazos"])

with tab_res2:
    c1, c2, c3 = st.columns(3)
    c1.metric("FS deslizamiento", f"{FS_desl:.3f}" if np.isfinite(FS_desl) else "—")
    c2.metric("FS volteo", f"{FS_volteo:.3f}" if np.isfinite(FS_volteo) else "—")
    c3.metric(
        "N (vertical total) [kN/m]", f"{N_static:.2f}" if np.isfinite(N_static) else "—"
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("q_min [kPa]", f"{q_min:.1f}" if np.isfinite(q_min) else "—")
    c5.metric("q_max [kPa]", f"{q_max:.1f}" if np.isfinite(q_max) else "—")
    c6.metric("q_adm [kPa]", f"{qadm:.1f}")

    st.write(
        f"**Tensiones**: {'✔️ sin tracción (q_min ≥ 0)' if no_traccion else '⚠️ hay tracción (q_min < 0)'} | "
        f"{'✔️ q_max ≤ q_adm' if dentro_qadm else '⚠️ q_max > q_adm'} | "
        f"{'✔️ tercio medio' if tercio_medio else '⚠️ fuera del tercio medio'}"
    )

    st.caption(
        "Tip: Si FS_desl es bajo, aumenta μ / añade llave / considera pasivo con reducción.\n"
        "Si FS_volteo es bajo o hay tracción, aumenta B o b_h, baja t_b (↑peso), o reduce P_H."
    )

with tab_press:
    st.markdown(
        "**Brazos y centroides (desde el borde de puntera a lo largo de la base):**"
    )
    dfM = pd.DataFrame(
        {
            "Componente": [
                "Losa base",
                "Alma",
                "Suelo sobre talón",
                "Suelo frente puntera",
                "Empuje total",
            ],
            "F [kN/m]": [W_base, W_stem, W_soil_heel, W_soil_toe, P_H],
            "Brazo [m]": [xW_base, xW_stem, xW_soil_heel, xW_soil_toe, a_P],
            "Momento (+res/-volc) [kN·m/m]": [
                W_base * xW_base,
                W_stem * xW_stem,
                W_soil_heel * xW_soil_heel,
                W_soil_toe * xW_soil_toe,
                -P_H * a_P,
            ],
        }
    ).round(3)
    st.dataframe(dfM, use_container_width=True, height=260)

    st.markdown(
        f"- **x_R** (posición de la resultante desde la puntera): {x_R:.3f} m  \n"
        f"- **e** (excentricidad respecto al centro): {e:.3f} m  \n"
        f"- **Regla del tercio medio**: {'✔️ Cumple (|e| ≤ B/6)' if tercio_medio else '⚠️ No cumple'}"
    )
