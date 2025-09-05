# Streamlit – Stack Graph (Origin-style) for Multiple Series
# Author: ChatGPT (GPT-5 Thinking)
# Description:
#   Build stacked multi-panel (layers) or offset-overlaid plots from a single table
#   of X plus multiple Y columns, similar to Origin's "Stack Graph". Supports
#   vertical or horizontal stacking, optional axis exchange for horizontal stacks,
#   adjustable spacing, smoothing, normalization, and one-axis-title mode.

import io
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import plotly.io as pio

st.set_page_config(page_title="Stack Graph – Multi-Layer/Offset", page_icon="🧮", layout="wide")

st.title("🧮 Stack Graph – Multi-Layer/Offset")
st.caption("Empilhe curvas em painéis (layers) verticais ou horizontais, ou sobreponha com deslocamento (offset), no estilo do Stack Graph do Origin.")

# ----------------------------- Sidebar Inputs ---------------------------- #
st.sidebar.header("Entrada de Dados")
files = st.sidebar.file_uploader("CSV com dados (1 arquivo)", type=["csv", "txt"], accept_multiple_files=False)

if not files:
    st.info("Envie um CSV contendo uma coluna X (opcional) e várias colunas Y.")
    st.stop()

# Robust read (comma or semicolon; dot/comma decimals)
@st.cache_data
def robust_read_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio)
        return df
    except Exception:
        bio.seek(0)
        try:
            df = pd.read_csv(bio, sep=';')
            return df
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, decimal=',', sep=';')
            return df

raw = robust_read_csv(files.getvalue())
all_cols = list(raw.columns)

st.sidebar.subheader("Mapeamento")
col_x = st.sidebar.selectbox("Coluna X (opcional)", ["<None>"] + all_cols, index=0)
ys = st.sidebar.multiselect("Colunas Y (uma ou mais)", all_cols, default=[c for c in all_cols if c != col_x][:4])

if not ys:
    st.warning("Selecione pelo menos uma coluna Y.")
    st.stop()

# Pre-processing options
st.sidebar.subheader("Pré-processamento")
normalize = st.sidebar.selectbox("Normalização Y", ["Nenhuma", "Min–Max (0–1)", "Dividir pelo máximo", "Z-score"], index=0)
smooth_on = st.sidebar.checkbox("Suavizar (Savitzky–Golay)", value=False)
sg_window = st.sidebar.slider("Janela SG (pontos, ímpar)", 5, 201, 21, step=2)
sg_poly = st.sidebar.slider("Ordem SG", 2, 5, 3)

# Layout options
st.sidebar.header("Layout & Estilo")
mode = st.sidebar.radio("Modo de plotagem", ["Multi-painel (empilhado)", "Offset sobreposto"], index=0)
stack_dir = st.sidebar.radio("Direção do empilhamento", ["Vertical", "Horizontal"], index=0)
exchange_axes = st.sidebar.checkbox("Horizontal com eixos trocados (X↔Y)", value=False, help="Análogo ao modo Horizontal (X–Y Axes Exchanged)")
spacing = st.sidebar.slider("Espaçamento entre painéis (0=justo)", 0.0, 0.25, 0.05, step=0.01)
show_one_axis_title = st.sidebar.checkbox("Mostrar um único título de eixo (global)", value=True)
share_x = st.sidebar.checkbox("Compartilhar eixo X entre painéis", value=True)
share_y = st.sidebar.checkbox("Compartilhar eixo Y entre painéis", value=False)

# Offset options (for overlaid mode)
offset_val = st.sidebar.number_input("Offset entre curvas (unidades Y)", value=0.0, step=0.1)
offset_as_frac = st.sidebar.checkbox("Offset relativo ao range de cada série (\%)", value=True)
offset_pct = st.sidebar.slider("Se relativo: % do range", 1, 200, 25)

palette = st.sidebar.selectbox("Paleta de cores", ["Plotly", "Viridis", "Cividis", "Plasma", "Turbo"], index=0)

# ------------------------------ Data Prep -------------------------------- #
# Build X and Y matrix
df = raw.copy()
if col_x != "<None>":
    x = pd.to_numeric(df[col_x], errors='coerce').to_numpy()
else:
    x = np.arange(len(df))

Y = []
labels = []
for c in ys:
    y = pd.to_numeric(df[c], errors='coerce').to_numpy()
    if normalize == "Min–Max (0–1)":
        mn, mx = np.nanmin(y), np.nanmax(y)
        rng = mx - mn if mx > mn else 1.0
        y = (y - mn) / rng
    elif normalize == "Dividir pelo máximo":
        mx = np.nanmax(np.abs(y))
        y = y / (mx if mx else 1.0)
    elif normalize == "Z-score":
        mu, sd = np.nanmean(y), np.nanstd(y)
        y = (y - mu) / (sd if sd else 1.0)
    if smooth_on:
        w = max(5, int(sg_window)); w += (w % 2 == 0)
        try:
            y = savgol_filter(y, window_length=w, polyorder=int(sg_poly))
        except Exception:
            pass
    Y.append(y)
    labels.append(str(c))

n = len(Y)

# Color sequence
if palette == "Plotly":
    colors = px_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
elif palette == "Viridis":
    colors = ["#440154","#482878","#3E4989","#31688E","#26828E","#1F9E89","#35B779","#6DCD59","#B4DE2C","#FDE725"]
elif palette == "Cividis":
    colors = ["#00224E","#25366F","#3F4A89","#5A5D9C","#7370A3","#8B84A2","#A29A98","#B9B08D","#D0C781","#E8DF74"]
elif palette == "Plasma":
    colors = ["#0d0887","#6a00a8","#b12a90","#e16462","#fca636","#fcffa4"]
else:
    colors = ["#30123B","#4145AB","#2CA6D8","#2AD4A5","#7CE080","#F9F871","#F6C64F","#F08E3E","#E84F3D","#D61E3C"]

# --------------------------- Plot Construction --------------------------- #
if mode == "Multi-painel (empilhado)":
    if stack_dir == "Vertical":
        fig = make_subplots(rows=n, cols=1, shared_xaxes=share_x, shared_yaxes=share_y, vertical_spacing=spacing)
        for i, (y, lab) in enumerate(zip(Y, labels), start=1):
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=lab, line=dict(width=2), showlegend=False), row=i, col=1)
            # Titles per layer as y-axis titles (or annotations)
            fig.update_yaxes(title_text=lab if not show_one_axis_title else None, row=i, col=1)
        if show_one_axis_title:
            fig.update_yaxes(title_text="Y", row=int(np.ceil(n/2.0)), col=1)
        fig.update_xaxes(title_text="X", row=n, col=1)
    else:
        # Horizontal stacking: either exchange axes or not
        fig = make_subplots(rows=1, cols=n, shared_xaxes=share_x, shared_yaxes=share_y, horizontal_spacing=spacing)
        for j, (y, lab) in enumerate(zip(Y, labels), start=1):
            xx, yy = (y, x) if exchange_axes else (x, y)
            fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name=lab, line=dict(width=2), showlegend=False), row=1, col=j)
            fig.update_yaxes(title_text=lab if not show_one_axis_title else None, row=1, col=j)
        if show_one_axis_title:
            fig.update_yaxes(title_text="Y", row=1, col=int(np.ceil(n/2.0)))
        fig.update_xaxes(title_text="X", row=1, col=int(np.ceil(n/2.0)))
else:
    # Offset overlay in a single axes
    fig = go.Figure()
    # Compute offsets
    offsets = []
    if offset_as_frac:
        for y in Y:
            rng = float(np.nanmax(y) - np.nanmin(y)) or 1.0
            offsets.append(rng * (offset_pct/100.0))
    else:
        offsets = [offset_val] * n
    current_shift = 0.0
    for i, (y, lab) in enumerate(zip(Y, labels)):
        fig.add_trace(go.Scatter(x=x, y=y + current_shift, mode='lines', name=lab, line=dict(width=2)))
        current_shift += offsets[i] if i < len(offsets) else 0.0
    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y (offset)")

# Apply colorway
fig.update_layout(template='plotly_dark', height=max(400, 220*n if mode.startswith("Multi") and stack_dir=="Vertical" else 600))
fig.update_layout(colorway=colors)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Export ---------------------------------- #
st.subheader("Exportar figura")
html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
st.download_button("⬇️ Baixar HTML interativo", data=html, file_name="stack_graph.html")

png_ok = st.checkbox("Exportar PNG (requer 'kaleido' instalado)", value=False)
if png_ok:
    try:
        import kaleido  # noqa: F401
        png_bytes = pio.to_image(fig, format='png', scale=2)
        st.download_button("⬇️ Baixar PNG", data=png_bytes, file_name="stack_graph.png")
    except Exception as e:
        st.warning(f"PNG indisponível: instale 'kaleido'. Erro: {e}")

# ------------------------------- Help ------------------------------------ #
with st.expander("Dicas & Notas"):
    st.markdown(
        """
        - **Multi-painel (empilhado):** cada coluna Y vira um *layer*. Você escolhe empilhar **Vertical** (linhas) ou **Horizontal** (colunas). 
        - **Horizontal com eixos trocados (X↔Y):** troca os eixos para reproduzir o comportamento "Horizontal (X–Y Axes Exchanged)".
        - **Compartilhar eixos:** compartilhe X e/ou Y para escalas comuns; títulos por layer podem ser ocultos com "Mostrar um único título".
        - **Offset sobreposto:** plota tudo num único painel, aplicando deslocamento cumulativo (fixo ou relativo ao range de cada série).
        - **Suavização SG:** útil para sinais ruidosos; use uma janela ímpar (5, 7, 9, ...).
        - **Exportação:** baixe HTML interativo (funciona sem dependências) e, opcionalmente, PNG via *kaleido*.
        """
    )
