# -*- coding: utf-8 -*-
"""
Streamlit app para plotar gráficos de linhas em:
- Curvas sobrepostas (um eixo comum)
- Painéis (small multiples) empilhados verticalmente, com eixos X/Y compartilhados

Suavização **avançada**:
- Nenhuma, Média móvel (com modo de borda), Savitzky–Golay, Gaussiana 1D, Mediana, LOWESS (robusta), Butterworth (passa-baixas)
- Controles específicos por método (janela, ordem do polinômio, sigma, fração LOWESS, ordem do filtro, corte como fração do Nyquist)

Outros recursos:
- Ajuste de **espessura** e **cor da linha** (uma cor para todas ou automáticas)
- Ajuste do **tamanho dos textos** dos eixos (títulos) e **números** (ticks)
- Rótulos mestres dos eixos (A para X e D para Y, configuráveis)
- Letras de identificação em cada painel (ex.: B, D, F, H)
- Borda preta em cada painel (mirror nos eixos)
- Download do gráfico via botão nativo do Plotly (não requer Kaleido/Chrome)

Requirements mínimos:
streamlit>=1.36
plotly>=5.22.0
pandas>=2.2
numpy>=1.26
scipy>=1.11            # Savitzky–Golay, Gaussiana, Mediana, Butterworth
statsmodels>=0.14      # apenas se usar LOWESS (opcional)
openpyxl>=3.1          # somente se for ler .xlsx
"""

import io
import csv
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SciPy filtros (opcionais mas recomendados)
try:
    from scipy.signal import savgol_filter, medfilt, butter, filtfilt
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# LOWESS via statsmodels (opcional)
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# -------------------------
# Utilidades
# -------------------------

def to_numeric_series(arr_like):
    """Converte qualquer array/Series para float (com coerção). Retorna np.ndarray[float]."""
    return pd.to_numeric(pd.Series(arr_like), errors="coerce").to_numpy(dtype=float)


def read_table_auto(file, sep_opt: str = "auto", sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Lê CSV/TXT/XLSX com separador detectado ou informado.
    - sep_opt em {"auto", ",", ";", "	"}
    - Para XLSX, sheet_name pode ser None (primeira) ou nome/índice.
    """
    name = file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(file, sheet_name=sheet_name)

    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    # Detecta separador
    if sep_opt == "auto":
        sample = raw[:4096].decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "	", " "])
            sep = dialect.delimiter
        except Exception:
            # Heurística simples
            if sample.count(";") > sample.count(",") and sample.count(";") >= sample.count("	"):
                sep = ";"
            elif sample.count("	") >= sample.count(",") and sample.count("	") >= sample.count(";"):
                sep = "	"
            else:
                sep = ","
    else:
        sep = sep_opt

    # Lê CSV/TXT
    return pd.read_csv(io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else io.StringIO(str(raw)),
                       sep=sep, engine="python")


def ensure_labels(n: int, raw_labels: str) -> List[str]:
    base = [s.strip() for s in (raw_labels or "").split(",") if s.strip()]
    if not base:
        base = ["B", "D", "F", "H"]
    if len(base) < n:
        alphabet = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        extra = [ch for ch in alphabet if ch not in base]
        base += extra[: max(0, n - len(base))]
    return base[:n]


def _interp_nans(y: np.ndarray) -> np.ndarray:
    y2 = np.asarray(y, dtype=float).copy()
    if y2.size == 0:
        return y2
    nans = np.isnan(y2)
    if nans.any() and (~nans).any():
        idx = np.arange(y2.size)
        y2[nans] = np.interp(idx[nans], idx[~nans], y2[~nans])
    return y2


def _is_uniform(x: np.ndarray, tol: float = 0.05) -> bool:
    if x.size < 3:
        return True
    dx = np.diff(x)
    m = np.nanmean(dx)
    s = np.nanstd(dx)
    return (m != 0) and (s / abs(m) < tol)


def smooth_series(x: np.ndarray, y: np.ndarray, method: str, params: dict) -> np.ndarray:
    """Aplica suavização escolhida. Retorna série suavizada.
    Métodos: none, moving, savgol, gaussian, median, lowess, butterworth
    """
    y0 = _interp_nans(y)

    if method == "none":
        return y0

    if method == "moving":
        w = int(max(1, params.get("window", 21)))
        if w <= 1:
            return y0
        if w % 2 == 0:
            w += 1
        mode = params.get("pad_mode", "reflect")  # reflect/nearest/edge/constant
        pad = w // 2
        ypad = np.pad(y0, pad, mode=mode)
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(ypad, kernel, mode="valid")

    if method == "savgol" and _HAS_SCIPY:
        w = int(max(3, params.get("window", 21)))
        if w % 2 == 0:
            w += 1
        poly = int(max(1, min(params.get("polyorder", 3), w - 2)))
        return savgol_filter(y0, window_length=w, polyorder=poly, mode="interp")

    if method == "gaussian" and _HAS_SCIPY:
        sigma = float(max(0.1, params.get("sigma", 2.0)))
        return gaussian_filter1d(y0, sigma=sigma, mode="reflect", truncate=3.0)

    if method == "median" and _HAS_SCIPY:
        w = int(max(3, params.get("window", 11)))
        if w % 2 == 0:
            w += 1
        return medfilt(y0, kernel_size=w)

    if method == "lowess":
        if not _HAS_SM:
            st.warning("LOWESS requer statsmodels>=0.14. Voltando à média móvel.")
            return smooth_series(x, y0, "moving", {"window": params.get("fallback_window", 21)})
        frac = float(min(max(params.get("frac", 0.05), 0.01), 0.99))
        iters = int(max(0, params.get("iters", 1)))
        # statsmodels retorna na ordem de x; usamos return_sorted=False p/ manter a ordem original
        try:
            return sm_lowess(y0, x, frac=frac, it=iters, return_sorted=False)
        except Exception:
            st.warning("Falha no LOWESS; usando média móvel como fallback.")
            return smooth_series(x, y0, "moving", {"window": params.get("fallback_window", 21)})

    if method == "butterworth" and _HAS_SCIPY:
        # Requer amostragem aproximadamente uniforme
        if not _is_uniform(x):
            st.info("Eixo X não uniforme; filtragem Butterworth pode distorcer. Usando Savitzky–Golay.")
            return smooth_series(x, y0, "savgol", {"window": params.get("window", 21), "polyorder": params.get("polyorder", 3)})
        order = int(min(max(params.get("order", 3), 1), 8))
        cutoff_frac = float(min(max(params.get("cutoff_frac", 0.1), 0.01), 0.49))  # 0–0.5
        dx = np.median(np.diff(x))
        fs = 1.0 / dx  # Hz equivalente
        wn = cutoff_frac * (fs / 2.0)  # Hz
        b, a = butter(order, wn, btype="low", fs=fs)
        return filtfilt(b, a, y0)

    # Se método não disponível (ex.: SciPy não instalado)
    if method in {"savgol", "gaussian", "median", "butterworth"} and not _HAS_SCIPY:
        st.warning("Este método requer SciPy. Voltando à média móvel.")
        return smooth_series(x, y0, "moving", {"window": params.get("fallback_window", 21)})

    return y0


# -------------------------
# App
# -------------------------

st.set_page_config(page_title="Stack Graph - Linhas", layout="wide")
st.title("Gráfico de Linhas • Curvas e Painéis (Plotly)")

with st.sidebar:
    st.header("Dados")
    up = st.file_uploader("Carregar arquivo", type=["csv", "txt", "xlsx"], accept_multiple_files=False)
    sep_opt = st.selectbox("Separador (CSV/TXT)", ["auto", ",", ";", "	"], index=0)

    sheet_name = None
    if up is not None and up.name.lower().endswith(".xlsx"):
        sheet_name = st.text_input("Sheet (deixe vazio para primeira)", value="") or None

    st.header("Colunas e modo")
    mode = st.radio("Modo de exibição", ["Painéis (small multiples)", "Curvas sobrepostas"], index=0)

    st.header("Suavização (avançado)")
    do_smooth = st.checkbox("Ativar suavização", value=False)
    method = st.selectbox(
        "Método",
        ["Média móvel", "Savitzky–Golay", "Gaussiana 1D", "Mediana", "LOWESS (robusta)", "Butterworth (passa-baixas)"]
    )
    # Controles específicos
    smooth_params = {}
    if method == "Média móvel":
        smooth_params["window"] = st.slider("Janela (pontos)", 3, 1001, 21, step=2)
        smooth_params["pad_mode"] = st.selectbox("Borda (padding)", ["reflect", "nearest", "edge", "constant"], index=0)
        chosen_method = "moving"
    elif method == "Savitzky–Golay":
        smooth_params["window"] = st.slider("Janela (pontos)", 5, 501, 31, step=2)
        smooth_params["polyorder"] = st.slider("Ordem do polinômio", 1, 7, 3)
        chosen_method = "savgol"
    elif method == "Gaussiana 1D":
        smooth_params["sigma"] = st.slider("Sigma (desvio padrão)", 0.2, 20.0, 2.0)
        chosen_method = "gaussian"
    elif method == "Mediana":
        smooth_params["window"] = st.slider("Janela (pontos)", 3, 501, 11, step=2)
        chosen_method = "median"
    elif method == "LOWESS (robusta)":
        smooth_params["frac"] = st.slider("Fração de suavização (0–1)", 0.01, 0.5, 0.08)
        smooth_params["iters"] = st.slider("Iterações robustas", 0, 5, 1)
        chosen_method = "lowess"
    else:  # Butterworth
        smooth_params["order"] = st.slider("Ordem do filtro", 1, 8, 3)
        smooth_params["cutoff_frac"] = st.slider("Corte (fração do Nyquist)", 0.01, 0.49, 0.10)
        # fallback para Savitzky se X não uniforme
        smooth_params["window"] = 31
        smooth_params["polyorder"] = 3
        chosen_method = "butterworth"

    st.header("Estilo de linhas")
    line_width = st.slider("Espessura das linhas", 0.5, 6.0, 1.0, step=0.5)
    use_single_color = st.checkbox("Usar a mesma cor para todas as curvas", value=True)
    line_color = st.color_picker("Cor da(s) linha(s)", value="#000000")

    st.header("Textos e bordas")
    x_label = st.text_input("Rótulo do eixo X", value="A")
    y_label = st.text_input("Rótulo do eixo Y", value="D")
    axis_title_size = st.slider("Tamanho do texto dos títulos dos eixos", 8, 36, 16)
    tick_font_size = st.slider("Tamanho dos números (ticks)", 6, 28, 12)
    panel_label_size = st.slider("Tamanho das letras dos painéis", 8, 30, 14)
    show_grid = st.checkbox("Mostrar grid", value=False)
    show_border = st.checkbox("Borda preta nos painéis", value=True)

    same_y = False
    panel_labels_raw = "B, D, F, H"
    if mode.startswith("Painéis"):
        same_y = st.checkbox("Mesma escala Y em todos os painéis", value=True)
        panel_labels_raw = st.text_input("Rótulos dos painéis (separados por vírgula)", value="B, D, F, H")

# Carrega/gera DataFrame
if up is not None:
    try:
        df = read_table_auto(up, sep_opt=sep_opt, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()
else:
    st.info("Nenhum arquivo carregado. Exibindo dados de exemplo.")
    x = np.linspace(0, 90, 2000)
    df = pd.DataFrame({
        "X": x,
        "B": 2e3*np.exp(-(x-8)**2/8) + 3e3*np.exp(-(x-30)**2/2),
        "D": np.linspace(80, 10, x.size),
        "F": 1e3*np.sin(x/2) + 2e3*np.exp(-(x-53)**2/3) + 1e3*np.exp(-(x-60)**2/5),
        "H": np.zeros_like(x)
    })

# Seleção de colunas
all_cols = df.columns.tolist()
col1, col2 = st.columns([1, 2])
with col1:
    x_col = st.selectbox("Coluna X", options=all_cols, index=0 if "X" not in all_cols else all_cols.index("X"))
with col2:
    default_y_candidates = [c for c in all_cols if c != x_col][:4]
    y_cols = st.multiselect("Colunas Y (uma por painel, na ordem)", options=[c for c in all_cols if c != x_col],
                            default=default_y_candidates if default_y_candidates else [])

if not x_col or not y_cols:
    st.warning("Selecione a coluna X e pelo menos uma coluna Y.")
    st.stop()

# Converte dados selecionados para float
x_vals = to_numeric_series(df[x_col].values)
valid_mask = ~np.isnan(x_vals)
x_vals = x_vals[valid_mask]

ys = []
for yc in y_cols:
    yv = to_numeric_series(df[yc].values)[valid_mask]
    if do_smooth:
        yv = smooth_series(x_vals, yv, chosen_method, smooth_params)
    ys.append(yv)

# -------------------------
# Construção da figura
# -------------------------

common_axis_style = dict(showgrid=show_grid, zeroline=False)
if show_border:
    common_axis_style.update(dict(showline=True, linewidth=1, linecolor="black", mirror=True))

if mode.startswith("Painéis"):
    nrows = len(ys)
    labels = ensure_labels(nrows, panel_labels_raw)

    fig = make_subplots(
        rows=nrows, cols=1,
        shared_xaxes=True,
        shared_yaxes=True if same_y else False,
        vertical_spacing=0.02,
        x_title=x_label,
        y_title=y_label,
    )

    # Adiciona uma curva por painel
    for i, (yv, yc) in enumerate(zip(ys, y_cols), start=1):
        color = line_color if use_single_color else None
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=yv,
                mode="lines",
                line=dict(color=color, width=line_width),
                name=str(yc),
                showlegend=False if use_single_color else True,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        height=max(350, 220 * nrows),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=30, b=50),
        showlegend=False if use_single_color else True,
        xaxis_title_font=dict(size=axis_title_size),
        yaxis_title_font=dict(size=axis_title_size),
    )

    fig.update_xaxes(**common_axis_style, tickfont=dict(size=tick_font_size))
    fig.update_yaxes(**common_axis_style, tickfont=dict(size=tick_font_size))

    # Rótulos dos painéis
    for i in range(1, nrows + 1):
        yax = fig.layout["yaxis" if i == 1 else f"yaxis{i}"]
        y0, y1 = yax.domain
        fig.add_annotation(
            text=labels[i - 1],
            xref="paper", yref="paper",
            x=0.02,
            y=y1 - 0.01,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=panel_label_size, color="#000"),
        )

else:
    # Curvas sobrepostas
    fig = go.Figure()
    for yv, yc in zip(ys, y_cols):
        color = line_color if use_single_color else None
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=yv,
                mode="lines",
                line=dict(color=color, width=line_width),
                name=str(yc),
            )
        )

    fig.update_layout(
        height=550,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=40, b=60),
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False if use_single_color else True,
        xaxis_title_font=dict(size=axis_title_size),
        yaxis_title_font=dict(size=axis_title_size),
    )

    fig.update_xaxes(**common_axis_style, tickfont=dict(size=tick_font_size))
    fig.update_yaxes(**common_axis_style, tickfont=dict(size=tick_font_size))

# -------------------------
# Renderização + Download (cliente)
# -------------------------

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "grafico",
            "height": 900,
            "width": 1200,
            "scale": 2,
        },
    },
)

with st.expander("ℹ️ Dicas e observações"):
    st.markdown(
        """
        **Suavização**
        - *Média móvel*: boa para ruído branco; experimente janelas ímpares (p.ex. 21, 31, 51) e padding *reflect*.
        - *Savitzky–Golay*: preserva picos/derivadas; ajuste janela (ímpar) e ordem do polinômio.
        - *Gaussiana*: suavização contínua controlada por **sigma** (mais alto = mais suave).
        - *Mediana*: remove *spikes* sem reduzir picos largos; use janela ímpar.
        - *LOWESS*: regressão local robusta; **frac** controla o quanto suaviza.
        - *Butterworth*: corte em fração do Nyquist; útil para séries bem amostradas (X ~ uniforme).
        
        **Estilo**
        - Ajuste **cor**, **espessura**, **títulos** e **ticks** na barra lateral.
        - Botão de **download (câmera)** do Plotly funciona no Safari sem Kaleido/Chrome.
        """
    )


























