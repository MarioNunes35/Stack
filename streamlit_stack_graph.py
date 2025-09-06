

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stack Graph Plotter", layout="wide")

st.title("📈 Stack Graph Plotter")
st.caption("Carregue CSV/TXT/Excel, selecione eixos e colunas, e gere gráfico em **overlay** ou **painéis (small multiples)**.")

# ============== I/O helpers ==============
@st.cache_data(show_spinner=False)
def robust_read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    data = bytes(file_bytes)
    bio = io.BytesIO(data)
    sig = data[:8]

    # Excel moderno (.xlsx)
    if sig[:2] == b'PK' or filename.lower().endswith(".xlsx"):
        bio.seek(0)
        return pd.read_excel(bio, engine="openpyxl")

    # Excel antigo (.xls)
    if sig.startswith(b"\xD0\xCF\x11\xE0") or filename.lower().endswith(".xls"):
        bio.seek(0)
        return pd.read_excel(bio)

    # CSV/TXT tentativas de codificação/sep/decimal
    def try_read(enc, sep, dec, engine=None, enc_errors=None):
        bio.seek(0)
        return pd.read_csv(
            bio,
            encoding=enc,
            sep=sep,
            decimal=dec,
            engine=engine,
            encoding_errors=enc_errors,
        )

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16-le", "utf-16-be"]
    seps = [";", ",", "\t"]
    decimals = [",", "."]

    for enc in encodings:
        for sep in seps:
            for dec in decimals:
                try:
                    df = try_read(enc, sep, dec)
                    if df.shape[1] == 1:
                        df2 = try_read(enc, sep, dec, engine="python")
                        if df2.shape[1] > 1:
                            return df2
                    return df
                except Exception:
                    pass

    # Última tentativa permissiva
    bio.seek(0)
    try:
        return pd.read_csv(bio, engine="python", sep=None, encoding="latin1", encoding_errors="replace")
    except Exception:
        bio.seek(0)
        return pd.read_table(bio, encoding="latin1", encoding_errors="replace")


def detect_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])]


def guess_x_column(df: pd.DataFrame):
    candidates = [
        "x","X","time","Time","tempo","Tempo","t","T","2theta","2-theta","two_theta",
        "wavenumber","Wavenumber","frequency","Frequency","A","Ângulo","angle","Angle",
        "m/z","mz","MZ","temperature","Temperature","Temperatura","index","Index"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    nums = detect_numeric_columns(df)
    if nums:
        return nums[0]
    return df.columns[0]


def nice_first_valid(arr):
    for v in arr:
        if pd.notna(v):
            try:
                return float(v)
            except Exception:
                continue
    return 0.0


def to_numeric_series(s, allow_datetime=False):
    """Converte Series/ndarray/list para ndarray float (ou datetime64 se allow_datetime=True)."""
    import pandas as pd
    import numpy as np
    # Se vier DataFrame, pega a primeira coluna
    if hasattr(s, 'ndim') and getattr(s, 'ndim', 1) == 2:
        try:
            s = s.iloc[:, 0]
        except Exception:
            pass
    # Se for pandas Series/Index, checa datetime
    if hasattr(s, 'dtype'):
        try:
            if allow_datetime and (pd.api.types.is_datetime64_any_dtype(s.dtype) or pd.api.types.is_timedelta64_dtype(s.dtype)):
                return s.to_numpy()
        except Exception:
            pass
        try:
            return pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
        except Exception:
            return pd.to_numeric(pd.Series(np.asarray(s)), errors='coerce').to_numpy(dtype=float)
    # Caso geral: array/list
    try:
        arr = pd.to_numeric(s, errors='coerce')
        return np.asarray(arr, dtype=float)
    except Exception:
        return pd.to_numeric(pd.Series(np.asarray(s)), errors='coerce').to_numpy(dtype=float)


def apply_axis_preset(values, preset, pad_pct=2.0, vmin_in=None, vmax_in=None):
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmin, vmax = float(np.min(vals)), float(np.max(vals))

    if preset == "Auto (Plotly)":
        return None
    elif preset == "Dados (min → max)":
        pass
    elif preset == "0 → max":
        vmin = 0.0
    elif preset == "P1 → P99":
        vmin, vmax = np.percentile(vals, [1, 99])
    elif preset == "Custom":
        if vmin_in is not None:
            vmin = float(vmin_in)
        if vmax_in is not None:
            vmax = float(vmax_in)

    span = vmax - vmin if vmax > vmin else (abs(vmax) if vmax != 0 else 1.0)
    pad = (pad_pct / 100.0) * span
    vmin -= pad
    vmax += pad
    return [vmin, vmax]


def detect_xy_pairs(df: pd.DataFrame):
    cols = list(df.columns)
    m_x = []
    for c in cols:
        m = re.fullmatch(r'(x|X)(?:\.(\d+))?', str(c))
        if m:
            idx = m.group(2) or "0"
            m_x.append((c, idx))
    if not m_x:
        return None
    pairs = []
    for x_name, idx in m_x:
        for cand in ["y", "Y", "intensity", "Intensity", "I"]:
            y_name = f"{cand}.{idx}" if idx != "0" else cand
            if y_name in cols:
                pairs.append((x_name, y_name))
                break
    return pairs if pairs else None


# ============== Sidebar Controls ==============
with st.sidebar:
    st.header("⚙️ Opções do Gráfico")
    layout_mode = st.selectbox("Layout", ["Overlay (um gráfico)", "Painéis (small multiples)"], index=1)
    chart_type = st.selectbox("Traço", ["Linha", "Área empilhada (overlay)"], index=0)

    st.subheader("Ajuste Rápido dos Eixos (globais)")
    x_preset = st.selectbox("X", ["Auto (Plotly)", "Dados (min → max)", "0 → max", "P1 → P99", "Custom"], index=1)
    y_preset = st.selectbox("Y", ["Auto (Plotly)", "Dados (min → max)", "0 → max", "P1 → P99", "Custom"], index=1)

    x_min = st.number_input("X min (se Custom)", value=None, placeholder="auto", step=1.0, format="%.6f") if x_preset == "Custom" else None
    x_max = st.number_input("X max (se Custom)", value=None, placeholder="auto", step=1.0, format="%.6f") if x_preset == "Custom" else None
    y_min = st.number_input("Y min (se Custom)", value=None, placeholder="auto", step=1.0, format="%.6f") if y_preset == "Custom" else None
    y_max = st.number_input("Y max (se Custom)", value=None, placeholder="auto", step=1.0, format="%.6f") if y_preset == "Custom" else None
    pad_pct = st.slider("Padding (%)", 0.0, 20.0, 2.0, 0.5)

    st.subheader("Transformações Rápidas")
    x_zero_min = st.checkbox("Fazer X iniciar em 0 (subtrai min de X)", value=False)
    y_zero_first = st.checkbox("Fazer Y iniciar no 0 (subtrai o primeiro valor de cada série)", value=False)
    y_norm_0_100 = st.checkbox("Normalizar Y para 0–100%", value=False)

    st.subheader("Estilo")
    title = st.text_input("Título", value="")
    x_label = st.text_input("Rótulo X", value="A")
    y_label = st.text_input("Rótulo Y", value="D")
    show_grid = st.checkbox("Mostrar grid", value=True)
    show_range_slider = st.checkbox("Mostrar range slider do X", value=True)
    font_size = st.slider("Tamanho da fonte", 8, 28, 14)
    line_width = st.slider("Espessura das linhas", 1, 8, 2)
    panel_labels_str = st.text_input("Rótulos dos painéis (separados por vírgula)", value="")

    same_y_panels = st.checkbox("Mesma escala Y em todos os painéis", value=True)
    frame_panels = st.checkbox("Borda preta nos painéis", value=True)

    st.subheader("Exportar")
    filebase = st.text_input("Nome do arquivo", value="grafico")
    export_scale = st.slider("Escala (resolução)", 1, 6, 3)
    use_server_export = st.checkbox("Usar exportação no servidor (Kaleido)", value=False, help="Se desligado, use o botão da câmera no gráfico (cliente).")
    export_png = st.checkbox("Exportar PNG (servidor)", value=True, disabled=not use_server_export)
    export_svg = st.checkbox("Exportar SVG (servidor)", value=False, disabled=not use_server_export)
    export_html = st.checkbox("Exportar HTML interativo", value=True)

# ============== Main Area ==============
files = st.file_uploader(
    "Envie 1 ou mais arquivos (CSV, TXT, XLSX, XLS)",
    type=["csv", "txt", "tsv", "dat", "xlsx", "xls"],
    accept_multiple_files=True,
)

all_traces = []
x_domains_global = []

if files:
    for f in files:
        try:
            df = robust_read_any(f.getvalue(), f.name)
        except Exception as e:
            st.error(f"Falha ao ler **{f.name}**: {e}")
            continue

        with st.expander(f"Prévia — {f.name}"):
            st.dataframe(df.head(50), use_container_width=True)

        pairs = detect_xy_pairs(df)

        if pairs:
            st.markdown(f"#### Pares X,Y detectados — `{f.name}`")
            sel_pairs = []
            for i, (xc, yc) in enumerate(pairs, start=1):
                use = st.checkbox(f"Usar par {i}: X = `{xc}` , Y = `{yc}`", value=True, key=f"usepair_{f.name}_{i}")
                if use:
                    sel_pairs.append((xc, yc))
            if not sel_pairs:
                st.warning("Selecione ao menos um par X,Y.")
                continue

            for (x_col, y_col) in sel_pairs:
                x_vals = to_numeric_series(df[x_col])
                if x_zero_min and np.isfinite(np.nanmin(x_vals)):
                    x_vals = x_vals - np.nanmin(x_vals)
                y_vals = to_numeric_series(df[y_col])
                if y_zero_first:
                    y_vals = y_vals - nice_first_valid(y_vals)
                if y_norm_0_100:
                    ymin = np.nanmin(y_vals); ymax = np.nanmax(y_vals)
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax != ymin:
                        y_vals = (y_vals - ymin) * 100.0 / (ymax - ymin)
                all_traces.append({"x": x_vals, "y": y_vals, "name": f"{f.name} — {y_col}"})
                x_domains_global.extend(list(x_vals))

        else:
            st.markdown(f"#### Seleção de colunas — `{f.name}`")
            default_x = guess_x_column(df)
            x_col = st.selectbox(f"Coluna X ({f.name})", options=list(df.columns), index=list(df.columns).index(default_x) if default_x in df.columns else 0, key=f"x_{f.name}")

            y_options = [c for c in df.columns if c != x_col]
            def is_convertible(col):
                s = pd.to_numeric(df[col], errors="coerce")
                return s.notna().sum() >= max(3, int(0.5 * len(s)))
            default_ys = [c for c in y_options if is_convertible(c)]
            y_cols = st.multiselect(f"Colunas Y ({f.name})", options=y_options, default=default_ys if default_ys else y_options[:1], key=f"ys_{f.name}")

            x_vals_raw = to_numeric_series(df[x_col])
            if x_zero_min and np.isfinite(np.nanmin(x_vals_raw)):
                x_vals_raw = x_vals_raw - np.nanmin(x_vals_raw)

            for ycol in y_cols:
                y_vals = to_numeric_series(df[ycol])
                if y_zero_first:
                    y_vals = y_vals - nice_first_valid(y_vals)
                if y_norm_0_100:
                    ymin = np.nanmin(y_vals); ymax = np.nanmax(y_vals)
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax != ymin:
                        y_vals = (y_vals - ymin) * 100.0 / (ymax - ymin)
                all_traces.append({"x": x_vals_raw, "y": y_vals, "name": f"{f.name} — {ycol}"})
            x_domains_global.extend(list(x_vals_raw))

    if not all_traces:
        st.warning("Nenhum traço selecionado.")
        st.stop()

    xr = apply_axis_preset(x_domains_global, x_preset, pad_pct, x_min, x_max)

    if layout_mode.startswith("Overlay"):
        fig = go.Figure()
        for tr in all_traces:
            if chart_type.startswith("Área"):
                fig.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines", name=tr["name"],
                                         line=dict(width=line_width), stackgroup="one"))
            else:
                fig.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines", name=tr["name"],
                                         line=dict(width=line_width)))
        if xr:
            fig.update_xaxes(range=xr)
        yr = apply_axis_preset(np.concatenate([t["y"] for t in all_traces]), y_preset, pad_pct, y_min, y_max)
        if yr:
            fig.update_yaxes(range=yr)

        fig.update_layout(
            template="plotly_white",
            title=dict(text=title or None, x=0.02, xanchor="left"),
            legend=dict(font=dict(size=font_size), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title=x_label or None, showgrid=show_grid, rangeslider=dict(visible=show_range_slider)),
            yaxis=dict(title=y_label or None, showgrid=show_grid),
            font=dict(size=font_size),
            margin=dict(l=60, r=20, b=60, t=60),
            hovermode="x unified",
        )

    else:
        n = len(all_traces)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        labels = [s.strip() for s in panel_labels_str.split(",")] if panel_labels_str else []
        for i, tr in enumerate(all_traces, start=1):
            fig.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode="lines", name=tr["name"],
                                     line=dict(width=line_width)), row=i, col=1)
            # rótulo do painel (canto superior esquerdo)
            if i <= len(labels) and labels[i-1]:
                fig.add_annotation(text=labels[i-1], xref=f"x{i} domain", yref=f"y{i}",
                                   x=0.01, y=1.02, showarrow=False, font=dict(size=font_size, color="black"))
            # estilo por painel
            fig.update_yaxes(showgrid=show_grid, row=i, col=1)
            if frame_panels:
                fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True, row=i, col=1)
                fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True, row=i, col=1)

        if xr:
            fig.update_xaxes(range=xr)

        # mesma escala Y em todos painéis (global)
        if same_y_panels:
            yy = np.concatenate([np.array(t["y"], dtype=float) for t in all_traces])
            yy = yy[np.isfinite(yy)]
            if yy.size:
                yr = apply_axis_preset(yy, y_preset, pad_pct, y_min, y_max)
                if yr:
                    for i in range(1, n+1):
                        fig.update_yaxes(range=yr, row=i, col=1)

        # X label apenas no último
        for i in range(1, n):
            fig.update_xaxes(showticklabels=False, row=i, col=1)

        fig.update_layout(
            template="plotly_white",
            title=dict(text=title or None, x=0.02, xanchor="left"),
            showlegend=False,
            xaxis=dict(title=x_label or None, showgrid=show_grid, rangeslider=dict(visible=show_range_slider)),
            font=dict(size=font_size),
            margin=dict(l=60, r=20, b=60, t=60),
            hovermode="x unified",
            height=max(300, 220 * n),
        )
        if y_label:
            fig.add_annotation(text=y_label, xref="paper", yref="paper", x=-0.06, y=0.5,
                               textangle=-90, showarrow=False, font=dict(size=font_size))

    # ---- Export ----
    plot_config = {
        "displaylogo": False,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["toImage"],
        "toImageButtonOptions": {"format": "png", "filename": filebase, "scale": export_scale}
    }
    st.plotly_chart(fig, theme=None, config=plot_config, width="stretch")
    st.markdown("Dica: use o **ícone da câmera** no canto do gráfico para baixar PNG em alta (cliente).")

    if use_server_export:
        try:
            png_bytes = pio.to_image(fig, format="png", scale=export_scale)
            st.download_button("💾 Baixar PNG (servidor)", data=png_bytes, file_name=f"{filebase}.png", mime="image/png")
            if export_svg:
                svg_bytes = pio.to_image(fig, format="svg", scale=export_scale)
                st.download_button("💾 Baixar SVG (servidor)", data=svg_bytes, file_name=f"{filebase}.svg", mime="image/svg+xml")
        except Exception as e:
            st.warning("Exportação no servidor falhou (Kaleido/Chrome ausente). Use o botão da câmera no gráfico.")
            st.exception(e)

    if export_html:
        try:
            html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config=plot_config)
            st.download_button("🌐 Baixar HTML interativo", data=html_str.encode("utf-8"), file_name=f"{filebase}.html", mime="text/html")
        except Exception as e:
            st.warning("Falha ao gerar HTML.")
            st.exception(e)

else:
    st.info("Envie os arquivos para começar. Dica: **duplo-clique** no gráfico faz auto-zoom, e o **range slider** no X agiliza a navegação.")






