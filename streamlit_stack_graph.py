# -*- coding: utf-8 -*-
"""
Streamlit app para plotar gráficos de linhas em:
- Curvas sobrepostas (um eixo comum)
- Painéis (small multiples) empilhados verticalmente, com eixos X/Y compartilhados

Novidades:
- Botão/controle para **suavizar curvas** (média móvel), com ajuste de janela
- Ajuste de **espessura** e **cor da linha**
- Ajuste de **tamanho dos textos** dos eixos (títulos) e dos **números** (ticks)
- Linhas pretas finas por padrão (estilo minimalista)
- Rótulos mestres dos eixos (A para X e D para Y, configuráveis)
- Letras de identificação em cada painel (ex.: B, D, F, H)
- Borda preta em cada painel (mirror nos eixos)
- Download do gráfico via botão nativo do Plotly (não requer Kaleido/Chrome)

Requisitos mínimos (requirements.txt):
streamlit>=1.36
plotly>=5.22.0
pandas>=2.2
numpy>=1.26
openpyxl>=3.1   # somente se for ler .xlsx
"""

import io
import csv
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Gera lista de rótulos de painéis com base em uma string separada por vírgulas.
    - Se a lista for menor que n, completa com letras sequenciais (B, C, D, ...)
    - Se for maior, trunca.
    """
    base = [s.strip() for s in (raw_labels or "").split(",") if s.strip()]
    if not base:
        base = ["B", "D", "F", "H"]
    # Completa se necessário
    if len(base) < n:
        alphabet = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        extra = [ch for ch in alphabet if ch not in base]
        base += extra[: max(0, n - len(base))]
    return base[:n]


def smooth_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Suaviza a série com média móvel (janela em pontos). Lida com NaNs por interpolação.
    Se window <= 1 ou série vazia, retorna y sem alterações.
    """
    if y.size == 0 or window <= 1:
        return y
    w = int(max(1, round(window)))
    # Interpola NaNs para evitar propagação
    y2 = y.astype(float).copy()
    nans = np.isnan(y2)
    if nans.any():
        idx = np.arange(y2.size)
        y2[nans] = np.interp(idx[nans], idx[~nans], y2[~nans])
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y2, kernel, mode="same")


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

    st.header("Suavização")
    do_smooth = st.checkbox("Aplicar suavização (média móvel)", value=False)
    smooth_window = st.slider("Janela da média móvel (pontos)", min_value=3, max_value=501, value=21, step=2,
                              help="Use número ímpar para resultados mais estáveis; aumenta = mais suave")

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

# Remove NaNs de X mantendo alinhamento
valid_mask = ~np.isnan(x_vals)
x_vals = x_vals[valid_mask]

ys = []
for yc in y_cols:
    yv = to_numeric_series(df[yc].values)
    yv = yv[valid_mask]
    if do_smooth:
        yv = smooth_moving_average(yv, smooth_window)
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

    # shared_yaxes: compartilhado ou não
    shared_y = True if same_y else False

    fig = make_subplots(
        rows=nrows, cols=1,
        shared_xaxes=True,
        shared_yaxes=shared_y,
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

    # Estilo geral
    fig.update_layout(
        height=max(350, 220 * nrows),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=30, b=50),
        showlegend=False if use_single_color else True,
    )

    # Grid/bordas + fontes dos ticks
    fig.update_xaxes(**common_axis_style, tickfont=dict(size=tick_font_size))
    fig.update_yaxes(**common_axis_style, tickfont=dict(size=tick_font_size))

    # Tamanho dos títulos mestres dos eixos
    fig.update_layout(
        xaxis_title_font=dict(size=axis_title_size),
        yaxis_title_font=dict(size=axis_title_size),
    )

    # === Rótulos de cada painel (coordenadas do "paper") ===
    for i in range(1, nrows + 1):
        yax = fig.layout["yaxis" if i == 1 else f"yaxis{i}"]
        y0, y1 = yax.domain  # topo do domínio do subplot i
        fig.add_annotation(
            text=labels[i - 1],
            xref="paper", yref="paper",
            x=0.02,           # ~2% a partir da borda esquerda
            y=y1 - 0.01,      # um pouco abaixo do topo do painel
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=panel_label_size, color="#000" if use_single_color else line_color),
        )

else:
    # Curvas sobrepostas em um único eixo
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
    )

    fig.update_xaxes(**common_axis_style, tickfont=dict(size=tick_font_size))
    fig.update_yaxes(**common_axis_style, tickfont=dict(size=tick_font_size))
    # Tamanhos dos títulos
    fig.update_layout(
        xaxis_title_font=dict(size=axis_title_size),
        yaxis_title_font=dict(size=axis_title_size),
    )

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

# Dica de uso
with st.expander("ℹ️ Dicas e observações"):
    st.markdown(
        """
        - No modo **Painéis**, selecione **uma coluna Y por painel** (na ordem desejada).
        - Use **Aplicar suavização** para reduzir ruído (método média móvel). Ajuste a **janela** conforme a densidade de pontos.
        - Ajuste **cor da linha**, **espessura**, **tamanho dos títulos** e **ticks** na barra lateral.
        - O botão de **download (câmera)** do Plotly funciona no Safari sem precisar de Kaleido/Chrome.
        """
    )

























