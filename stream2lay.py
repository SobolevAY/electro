"""
Streamlit-приложение: двухслойная кривая ВЭЗ. Для курса электроразведки ГГФ НГУ, 2026
Идеальный зонд Шлюмберже (MN→0). Для ΔU — фиктивные I, MN.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Двухслойная кривая ВЭЗ",
                   page_icon="🌍", layout="wide")

# ---------------- Константы ----------------
EPS    = 1e-3      # порог сходимости ряда
I_APP  = 1.0       # A — только для отображения ΔU
MN_APP = 0.1       # m — только для отображения ΔU
N_PTS  = 25


# ---------------- Физика ----------------
@st.cache_data(show_spinner=False)
def rhoa_ideal(r_tuple, rho1, rho2, h, eps=EPS):
    """
    Идеальный зонд Шлюмберже (MN → 0):
        ρk(r) = ρ1 · [ 1 + 2 Σ_j k^j · r³ / (r² + (2 j h)²)^{3/2} ],
        k = (ρ2 − ρ1)/(ρ2 + ρ1).
    """
    r = np.asarray(r_tuple, dtype=float)
    k = (rho2 - rho1) / (rho2 + rho1)
    if abs(k) < 1e-12:
        return np.full_like(r, rho1), 0

    J  = 256
    r2 = r ** 2
    r3 = r ** 3
    while True:
        j  = np.arange(1, J + 1)[:, None]
        ds = r3[None, :] * (k ** j) / (
             r2[None, :] + (2.0 * j * h) ** 2) ** 1.5
        tot = ds.sum(axis=0)
        tail  = 2.0 * np.abs(ds[-1]) * abs(k) / max(1.0 - abs(k), 1e-12)
        S_abs = np.abs(1.0 + 2.0 * tot)
        if np.all(tail <= eps * np.maximum(S_abs, 1e-12)) or J >= 2 ** 18:
            break
        J *= 2
    return rho1 * (1.0 + 2.0 * tot), J


def delta_u_from_rhoa(r, rhoa, I=I_APP, mn=MN_APP):
    """ΔU, соответствующий ρk при конечных I и MN:
        ΔU = ρk · I / K,   K = π (r² − (MN/2)²) / MN.
    """
    K = np.pi * (r ** 2 - (mn / 2.0) ** 2) / mn
    return rhoa * I / K


# ---------------- Синхронизированные ползунок + число ----------------
def linked_input(label, key, vmin, vmax, default, step):
    """Линейный слайдер + текстовое поле, синхронизированные в обе стороны."""
    sk, ik = f"{key}_slider", f"{key}_input"
    if sk not in st.session_state:
        st.session_state[sk] = float(default)
    if ik not in st.session_state:
        st.session_state[ik] = float(default)

    def from_input():
        st.session_state[sk] = st.session_state[ik]

    def from_slider():
        st.session_state[ik] = st.session_state[sk]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(label, float(vmin), float(vmax), step=float(step),
                  key=sk, on_change=from_slider)
    with c2:
        st.number_input(label, min_value=float(vmin), max_value=float(vmax),
                        step=float(step), key=ik, on_change=from_input,
                        label_visibility="collapsed")
    return float(st.session_state[sk])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Параметры модели")

    rho1 = linked_input(r"$\rho_1$, Ом·м", "rho1", 0.1, 1000.0, 1.0, 0.1)
    rho2 = linked_input(r"$\rho_2$, Ом·м", "rho2", 0.1, 1000.0, 10.0, 0.1)
    h    = linked_input(r"$h$, м",         "h",    0.1,  200.0,  1.0, 0.1)

    st.divider()
    st.subheader("Диапазон разносов, м")
    ab2_min = st.number_input("AB/2 min", 0.2, 1000.0, 1.0, 0.1)
    ab2_max = st.number_input("AB/2 max", 0.5, 5000.0, 200.0, 1.0)
    if ab2_max <= ab2_min:
        st.error("AB/2 max должен быть больше min; беру max = min·10.")
        ab2_max = ab2_min * 10.0

    st.divider()
    mode = st.radio("Что показывать",
                    ("Кажущееся сопротивление ρk",
                     "Разность потенциалов ΔU"))


# ---------------- Расчёт ----------------
AB2 = np.geomspace(ab2_min, ab2_max, N_PTS)
rhoa, J_used = rhoa_ideal(tuple(AB2.tolist()), rho1, rho2, h)
dU = delta_u_from_rhoa(AB2, rhoa)
k12 = (rho2 - rho1) / (rho2 + rho1)


# ---------------- Заголовок ----------------
st.subheader("Двухслойная кривая ВЭЗ (ГГФ НГУ)")
#st.caption(
#    fr"Идеальный зонд Шлюмберже (MN→0), "
#    fr"для $\Delta U$ показаны $I={I_APP:g}$ А и $MN={MN_APP:g}$ м · "
#    fr"$k_{{12}}={k12:+.3f}$ · в ряду {J_used} членов"
#)


# ---------------- График ----------------
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=110)

if mode.startswith("Кажущееся"):
    y, ylabel = rhoa, r"$\rho_k$, Ом·м"

    # ±5 % полосы вокруг ρ1 и ρ2
    ax.axhspan(rho1 * 0.95, rho1 * 1.05, color="C2", alpha=0.18, zorder=0)
    ax.axhspan(rho2 * 0.95, rho2 * 1.05, color="C3", alpha=0.18, zorder=0)

    ax.axhline(rho1, color="C2", lw=2.4, alpha=0.95,
               label=fr"$\rho_1 = {rho1:.4g}$")
    ax.axhline(rho2, color="C3", lw=2.4, alpha=0.95,
               label=fr"$\rho_2 = {rho2:.4g}$")

    gm = float(np.sqrt(rho1 * rho2))
    ax.axhline(gm, color="C1", lw=1.8, ls="--", alpha=0.95,
               label=fr"$\sqrt{{\rho_1\rho_2}} = {gm:.4g}$")
else:
    y, ylabel = dU, r"$\Delta U$, В"

# вертикальные реперы: h, 2h, 3h, 5h, 10h, 30h
xlim = (AB2[0], AB2[-1])
for mult in (1, 2, 3, 5, 10, 30):
    xv = mult * h
    if xlim[0] <= xv <= xlim[1]:
        ax.axvline(xv, color="0.55", lw=0.9, ls=":", zorder=0)
        ax.text(xv, 0.98, f"{mult}h",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8,
                color="0.35", rotation=90)

ax.loglog(AB2, y, "o-", color="C0", lw=2.2, ms=6, zorder=3)

ax.set_xlabel(r"$AB/2$, м", fontsize=13)
ax.set_ylabel(ylabel, fontsize=13)
ax.set_title(
    fr"$\rho_1={rho1:.4g}$ Ом·м,  $\rho_2={rho2:.4g}$ Ом·м,  "
    fr"$h={h:.4g}$ м,   $k={k12:+.3f}$",
    fontsize=13,
)
ax.grid(True, which="both", ls="--", alpha=0.45)
if mode.startswith("Кажущееся"):
    ax.legend(fontsize=10, loc="best")

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ---------------- Таблица ----------------
with st.expander("Таблица: разнос — ρk — ΔU", expanded=False):
    df = pd.DataFrame({
        "AB/2, м": AB2,
        "ρk, Ом·м": rhoa,
        "ΔU, В": dU,
    })
    st.dataframe(
        df.style.format({
            "AB/2, м": "{:.4g}",
            "ρk, Ом·м": "{:.5g}",
            "ΔU, В": "{:.5g}",
        }),
        use_container_width=True,
        hide_index=True,
    )
