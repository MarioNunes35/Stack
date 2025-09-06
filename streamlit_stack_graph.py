# -*- coding: utf-8 -*-
"""
Streamlit app para plotar gráficos de linhas em:
- Curvas sobrepostas (um eixo comum)
- Painéis (small multiples) empilhados verticalmente, com eixos X/Y compartilhados

Características:
- Linhas pretas finas (estilo minimalista)
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
import math
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
    - sep_opt em {"auto", ",", ";", "\t"}
    - Para XLSX, sheet_name pode ser None (primeira) ou nome/índice.
    """
    name = file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(file, sheet_name=sheet_name)

    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    # Detecta separador
    if sep_opt == "auto":
        sample = raw[:4096].decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        dialect = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", " "])
            sep = dialect.delimiter
        except Exception:
            # Heurística simples
            if sample.count(";") > sample.count(",") and sample.count(";") >= sample.count("\t"):
                sep = ";"
            elif sample.count("\t") >= sample.count(",") and sample.count("\t") >= sample.count(";"):
                sep = "\t"
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
        # Evita duplicar o que já existe
        extra = [ch for ch in alphabet if ch not in base]
        base += extra[: max(0, n - len(base))]
    # Trunca
    return base[:n]


# -------------------------
# App
# -------------------------

st.set_page_config(page_title="Stack Graph - Linhas", layout="wide")
st.title("Gráfico de Linhas • Curvas e Painéis (Plotly)")

with st.sidebar:
    st.header("Dados")
    up = st.file_uploader("Carregar arquivo", type=["csv", "txt", "xlsx"], accept_multiple_files=False)
    sep_opt = st.selectbox("Separador (CSV/TXT)", ["auto", ",", ";", "\t"], index=0)

    sheet_name = None
    if up is not None and up.name.lower().endswith(".xlsx"):
        sheet_name = st.text_input("Sheet (deixe vazio para primeira)", value="") or None

    st.header("Colunas")
    mode = st.radio("Modo de exibição", ["Painéis (small multiples)", "Curvas sobrepostas"], index=0)

    st.header("Estilo")
    x_label = st.text_input("Rótulo do eixo X", value="A")
    y_label = st.text_input("Rótulo do eixo Y", value="D")
    line_width = st.slider("Espessura das linhas", 0.5, 4.0, 1.0, step=0.5)
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
    x = np.linspace(0, 10, 400)
    df = pd.DataFrame({
        "X": x,
        "Y1": np.sin(x),
        "Y2": np.cos(x),
        "Y3": np.sin(2*x),
        "Y4": np.cos(2*x),
    })

# Seleção de colunas
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
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
    ys.append(yv)

# -------------------------
# Construção da figura
# -------------------------

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
    for i, yv in enumerate(ys, start=1):
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=yv,
                mode="lines",
                line=dict(color="black", width=line_width),
                showlegend=False,
            ),
            row=i, col=1,
        )

    # Estilo geral
    fig.update_layout(
        height=max(350, 220 * nrows),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=30, b=50),
        showlegend=False,
    )

    # Grid e borda
    fig.update_xaxes(showgrid=show_grid, zeroline=False)
    fig.update_yaxes(showgrid=show_grid, zeroline=False)

    if show_border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    # === Rótulos de cada painel (CORREÇÃO do erro do xref) ===
    # Usamos coordenadas relativas ao "paper" e a posição do domínio de cada eixo Y
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
            font=dict(size=14, color="black"),
        )

else:
    # Curvas sobrepostas em um único eixo
    fig = go.Figure()
    for yv, yc in zip(ys, y_cols):
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=yv,
                mode="lines",
                line=dict(width=line_width),  # cores automáticas do Plotly
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
        showlegend=True,
    )

    fig.update_xaxes(showgrid=show_grid, zeroline=False)
    fig.update_yaxes(showgrid=show_grid, zeroline=False)

    if show_border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

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
        # Você pode habilitar/desabilitar botões da modebar aqui se quiser
        # "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    },
)

# Dica de uso
with st.expander("ℹ️ Dicas e observações"):
    st.markdown(
        """
        - No modo **Painéis**, selecione **uma coluna Y por painel** (na ordem desejada).
        - Para replicar o estilo da figura de referência, mantenha **linhas pretas finas**, **grid desligado**, **borda preta** e rótulos **X = A** e **Y = D**.
        - O botão de **download (câmera)** do Plotly funciona no Safari sem precisar de Kaleido/Chrome.
        """
    )
























