
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(page_title="Stack Graph Plotter", layout="wide")

st.title("📈 Stack Graph Plotter")
st.caption("Carregue CSV/TXT/Excel, selecione eixos e colunas, e gere gráfico de linhas ou área empilhada.")

# ============== I/O helpers ==============
@st.cache_data(show_spinner=False)
def robust_read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Lê CSV/TXT/Excel de forma robusta (codificações/separadores) e retorna um DataFrame.
    """
    data = bytes(file_bytes)
    bio = io.BytesIO(data)
    sig = data[:8]

    # Excel moderno (.xlsx) - ZIP magic
    if sig[:2] == b'PK' or filename.lower().endswith(".xlsx"):
        bio.seek(0)
        return pd.read_excel(bio, engine="openpyxl")

    # Excel antigo (.xls) - OLE magic
    if sig.startswith(b"\xD0\xCF\x11\xE0") or filename.lower().endswith(".xls"):
        bio.seek(0)
        return pd.read_excel(bio)  # engine auto

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
                        # Tenta detecção mais flexível
                        df2 = try_read(enc, sep, dec, engine="python")
                        if df2.shape[1] > 1:
                            return df2
                    return df
                except Exception:
                    pass

    # Última tentativa (bem permissiva)
    bio.seek(0)
    try:
        return pd.read_csv(bio, engine="python", sep=None, encoding="latin1", encoding_errors="replace")
    except Exception:
        bio.seek(0)
        return pd.read_table(bio, encoding="latin1", encoding_errors="replace")


def detect_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])]


def guess_x_column(df: pd.DataFrame):
    # Heurística por nomes comuns e pela primeira coluna numérica
    candidates = [
        "x","X","time","Time","tempo","Tempo","t","T",
        "wavenumber","Wavenumber","frequency","Frequency",
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
    # Primeiro valor não-NaN
    for v in arr:
        if pd.notna(v):
            return float(v)
    return 0.0


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


# ============== Sidebar Controls ==============
with st.sidebar:
    st.header("⚙️ Opções do Gráfico")
    chart_type = st.selectbox("Tipo", ["Linha", "Área empilhada"], index=0)

    st.subheader("Ajuste Rápido dos Eixos")
    x_preset = st.selectbox("X", ["Auto (Plotly)", "Dados (min → max)", "0 → max", "P1 → P99", "Custom"], index=0)
    y_preset = st.selectbox("Y", ["Auto (Plotly)", "Dados (min → max)", "0 → max", "P1 → P99", "Custom"], index=0)

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
    x_label = st.text_input("Rótulo X", value="")
    y_label = st.text_input("Rótulo Y", value="")
    show_grid = st.checkbox("Mostrar grid", value=True)
    show_range_slider = st.checkbox("Mostrar range slider do X", value=True)
    font_size = st.slider("Tamanho da fonte", 8, 28, 14)
    line_width = st.slider("Espessura das linhas", 1, 8, 2)

    st.subheader("Exportar")
    filebase = st.text_input("Nome do arquivo", value="grafico")
    export_scale = st.slider("Escala (resolução)", 1, 6, 3)
    use_server_export = st.checkbox("Usar exportação no servidor (Kaleido)", value=False, help="Requer Kaleido funcional no servidor. Se desligado, use o botão da câmera no gráfico (cliente).")
    export_png = st.checkbox("Exportar PNG (servidor)", value=True, disabled=not use_server_export)
    export_svg = st.checkbox("Exportar SVG (servidor)", value=False, disabled=not use_server_export)
    export_html = st.checkbox("Exportar HTML interativo", value=True)

# ============== Main Area ==============
files = st.file_uploader(
    "Envie 1 ou mais arquivos (CSV, TXT, XLSX, XLS)",
    type=["csv", "txt", "tsv", "dat", "xlsx", "xls"],
    accept_multiple_files=True,
)

datasets = []

if files:
    for f in files:
        try:
            df = robust_read_any(f.getvalue(), f.name)
        except Exception as e:
            st.error(f"Falha ao ler **{f.name}**: {e}")
            continue

        # Mostra uma prévia
        with st.expander(f"Prévia — {f.name}"):
            st.dataframe(df.head(50), use_container_width=True)

        # Seleções de colunas
        num_cols = detect_numeric_columns(df)
        default_x = guess_x_column(df)

        st.markdown(f"#### Seleção de colunas — `{f.name}`")
        c1, c2 = st.columns([1, 2])
        with c1:
            x_col = st.selectbox(f"Coluna X ({f.name})", options=list(df.columns), index=list(df.columns).index(default_x) if default_x in df.columns else 0, key=f"x_{f.name}")
        with c2:
            y_options = [c for c in df.columns if c != x_col and (c in num_cols)]
            if not y_options:
                y_options = [c for c in df.columns if c != x_col]
            y_cols = st.multiselect(f"Colunas Y ({f.name})", options=y_options, default=y_options[:1], key=f"ys_{f.name}")

        datasets.append({
            "df": df,
            "x_col": x_col,
            "y_cols": y_cols,
            "label_prefix": f"{f.name} — "
        })

    # Construção do gráfico
    if any(len(ds["y_cols"]) > 0 for ds in datasets):
        fig = go.Figure()

        all_x_vals = []
        all_y_vals = []

        for ds in datasets:
            df = ds["df"]
            if ds["x_col"] not in df.columns or len(ds["y_cols"]) == 0:
                continue

            # Extrai X e aplica transformações
            x_raw = df[ds["x_col"]].values
            # Tenta converter datas para numérico se forem datetimes -> plotly aceita datetimes também
            if np.issubdtype(df[ds["x_col"]].dtype, np.datetime64):
                x_vals = x_raw
            else:
                x_vals = pd.to_numeric(x_raw, errors="coerce")
                if x_zero_min and np.isfinite(np.nanmin(x_vals)):
                    x_vals = x_vals - np.nanmin(x_vals)

            for ycol in ds["y_cols"]:
                if ycol not in df.columns:
                    continue
                y_vals = pd.to_numeric(df[ycol].values, errors="coerce")
                if y_zero_first:
                    y_vals = y_vals - nice_first_valid(y_vals)
                if y_norm_0_100:
                    ymin = np.nanmin(y_vals)
                    ymax = np.nanmax(y_vals)
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax != ymin:
                        y_vals = (y_vals - ymin) * 100.0 / (ymax - ymin)

                name = f"{ds['label_prefix']}{ycol}"
                if chart_type == "Linha":
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=name, line=dict(width=line_width)))
                else:
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=name, stackgroup="one", line=dict(width=line_width)))

                # agrega valores para presets
                all_x_vals.extend(pd.to_datetime(x_vals) if np.issubdtype(df[ds["x_col"]].dtype, np.datetime64) else x_vals)
                all_y_vals.extend(y_vals)

        # Layout geral
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

        # Presets de eixo (aplicados se não for datetime)
        if all_x_vals:
            if not (datasets and np.issubdtype(datasets[0]["df"][datasets[0]["x_col"]].dtype, np.datetime64)):
                xr = apply_axis_preset(all_x_vals, x_preset, pad_pct, x_min, x_max)
                if xr:
                    fig.update_xaxes(range=xr)

        if all_y_vals:
            yr = apply_axis_preset(all_y_vals, y_preset, pad_pct, y_min, y_max)
            if yr:
                fig.update_yaxes(range=yr)

        # ---- Client-side export config (no Chrome/Kaleido needed) ----
        plot_config = {
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filebase,
                "scale": export_scale,
                "height": None,
                "width": None,
            }
        }
        st.plotly_chart(fig, use_container_width=True, theme=None, config=plot_config)

        st.markdown("Dica: use o **ícone da câmera** no canto do gráfico para baixar PNG em alta (cliente).")

        # ---- Optional server-side export (Kaleido) ----
        if use_server_export:
            try:
                if export_png:
                    png_bytes = pio.to_image(fig, format="png", scale=export_scale)
                    st.download_button("💾 Baixar PNG (servidor)", data=png_bytes, file_name=f"{filebase}.png", mime="image/png")
                if export_svg:
                    svg_bytes = pio.to_image(fig, format="svg", scale=export_scale)
                    st.download_button("💾 Baixar SVG (servidor)", data=svg_bytes, file_name=f"{filebase}.svg", mime="image/svg+xml")
            except Exception as e:
                st.warning("Exportação no servidor falhou (Kaleido/Chrome ausente). Use o botão da câmera no gráfico.")
                st.exception(e)

        # ---- HTML interactive export ----
        if export_html:
            try:
                html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config=plot_config)
                st.download_button("🌐 Baixar HTML interativo", data=html_str.encode("utf-8"), file_name=f"{filebase}.html", mime="text/html")
            except Exception as e:
                st.warning("Falha ao gerar HTML.")
                st.exception(e)

    else:
        st.warning("Selecione pelo menos uma coluna Y em pelo menos um arquivo.")
else:
    st.info("Envie os arquivos para começar. Dica: **duplo-clique** no gráfico faz auto-zoom, e o **range slider** no X agiliza a navegação.")


