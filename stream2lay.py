"""
Streamlit-приложение: двухслойная кривая ВЭЗ.
Идеальный зонд Шлюмберже, MN = 0.1 м, I = 1 А.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Параметры по умолчанию ----------------
EPS  = 1e-4          # порог сходимости ряда (было 1e-5, стало грубее → быстрее)
MN   = 0.1           # м
I    = 1.0           # А

AB2 = np.array([
    1.5, 2, 3, 4, 5, 7, 9, 12, 15, 20,
    25, 32, 40, 50, 65, 80, 100, 123, 150, 180,
    220, 275, 340, 410, 480,
], dtype=float)


# ---------------- Физика (векторизовано по j) ----------------
@st.cache_data(show_spinner=False)
def compute_curve(rho1, rho2, h, ab2_tuple, mn=MN, I=I, eps=EPS):
    """
    Возвращает (dU, rhoa) для зонда Шлюмберже с фиксированным MN.

    Ряд:  U(s) = I*rho1/(2π) * [ 1/s + 2 Σ_{j≥1} k^j / sqrt(s² + (2jh)²) ]
    Сумма по j считается сразу для всех точек s как матрица (J, N).
    Число членов J подбирается удвоением, пока хвост ряда не станет мал.
    """
    ab2 = np.asarray(ab2_tuple, dtype=float)
    k   = (rho2 - rho1) / (rho2 + rho1)

    def U(s):
        s = np.asarray(s, dtype=float)
        if abs(k) < 1e-12:                       # однородная среда
            return I * rho1 / (2.0 * np.pi) / s

        J = 256
        while True:
            j   = np.arange(1, J + 1)[:, None]                 # (J, 1)
            ds  = (k ** j) / np.sqrt(s[None, :] ** 2
                                     + (2.0 * j * h) ** 2)     # (J, N)
            tot = ds.sum(axis=0)
            # консервативная оценка хвоста: Σ_{m>J} |k|^m ≈ |k|^{J+1}/(1-|k|)
            tail = np.abs(ds[-1]) * abs(k) / max(1.0 - abs(k), 1e-12)
            if np.all(tail <= eps * np.abs(tot)) or J >= 2**18:
                break
            J *= 2
        return I * rho1 / (2.0 * np.pi) * (1.0 / s + 2.0 * tot)

    am = ab2 - mn / 2.0
    an = ab2 + mn / 2.0
    dU = 2.0 * (U(am) - U(an))                   # суперпозиция A и B
    K  = np.pi * (ab2 ** 2 - (mn / 2.0) ** 2) / mn
    rhoa = K * dU / I
    return dU, rhoa


# ---------------- UI ----------------
st.set_page_config(page_title="Двухслойная кривая ВЭЗ",
                   page_icon="🌍", layout="wide")

st.title("Двухслойная кривая ВЭЗ")
st.caption(
    f"Идеальный зонд Шлюмберже · MN = {MN:g} м · I = {I:g} А · "
    "расчёт методом зеркальных отражений (ряд по кратностям)"
)

with st.sidebar:
    st.header("Параметры модели")
    st.caption("Ползунки задают log₁₀ величины; значение — под ползунком.")

    log_rho1 = st.slider(r"$\log_{10}\rho_1$", -1.0, 4.0, 0.0, 0.05)
    rho1 = float(10.0 ** log_rho1)
    st.caption(fr"$\rho_1$ = **{rho1:.3g}** Ом·м")

    log_rho2 = st.slider(r"$\log_{10}\rho_2$", -1.0, 4.0, 1.0, 0.05)
    rho2 = float(10.0 ** log_rho2)
    st.caption(fr"$\rho_2$ = **{rho2:.3g}** Ом·м")

    log_h = st.slider(r"$\log_{10}h$", -1.0, 2.0, 0.0, 0.05)
    h = float(10.0 ** log_h)
    st.caption(fr"$h$ = **{h:.3g}** м")

    st.divider()
    mode = st.radio(
        "Что показывать",
        ("Кажущееся сопротивление ρk", "Разность потенциалов ΔU"),
    )

# ---------------- Расчёт ----------------
dU, rhoa = compute_curve(rho1, rho2, h, tuple(AB2))
k12 = (rho2 - rho1) / (rho2 + rho1)

# ---------------- График ----------------
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)

if mode.startswith("Кажущееся"):
    y, ylabel = rhoa, r"$\rho_k$, Ом·м"
    ax.axhline(rho1, color="0.65", ls=":",  lw=1.2, label=fr"$\rho_1={rho1:.3g}$")
    ax.axhline(rho2, color="0.65", ls="--", lw=1.2, label=fr"$\rho_2={rho2:.3g}$")
else:
    y, ylabel = dU, r"$\Delta U$, В"

ax.loglog(AB2, y, "o-", color="C0", lw=2.0, ms=6)

ax.set_xlabel(r"$AB/2$, м", fontsize=13)
ax.set_ylabel(ylabel, fontsize=13)
ax.set_title(
    fr"$\rho_1={rho1:.3g}$ Ом·м,  $\rho_2={rho2:.3g}$ Ом·м,  "
    fr"$h={h:.3g}$ м,  $k_{{12}}={k12:+.3f}$",
    fontsize=13,
)
ax.grid(True, which="both", ls="--", alpha=0.5)
if mode.startswith("Кажущееся"):
    ax.legend(fontsize=11, loc="best")

st.pyplot(fig, use_container_width=True)

# ---------------- Пояснения ----------------
with st.expander("Формулы и как это считается"):
    st.markdown(
        r"""
**Метод зеркальных отражений.** Потенциал точечного источника
на поверхности двухслойной среды:

$$
U(s) = \frac{I\rho_1}{2\pi}
       \left[\frac{1}{s} + 2\sum_{j=1}^{\infty}
       \frac{k_{12}^{j}}{\sqrt{s^2 + (2jh)^2}}\right],
\qquad
k_{12} = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}.
$$

**Зонд Шлюмберже** (A–M–N–B, симметричный, $r = AB/2$):

$$
\Delta U = 2\bigl[\,U(r - MN/2) - U(r + MN/2)\,\bigr],
\qquad
\rho_k = \frac{\pi\bigl(r^2 - (MN/2)^2\bigr)}{MN}\,\frac{\Delta U}{I}.
$$

Ряд обрывается, когда консервативная оценка хвоста
$\sum_{m>J}|k|^{m}\big/\sqrt{\dots}$ становится меньше
$\varepsilon\cdot|\sum_{m\le J}|$; по умолчанию $\varepsilon = 10^{-4}$.
        """
    )
