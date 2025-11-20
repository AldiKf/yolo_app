import streamlit as st
from model_loader import load_model
from detect_video import process_video
from utils import get_linechart_data, get_barchart_data
import pandas as pd
import tempfile
import cv2
import base64
import os
import altair as alt



# PAGE CONFIG

st.set_page_config(page_title="YOLO Traffic Analyzer", layout="wide")



# BACKGROUND

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{data}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass


local_bg_paths = ["assets/bg.jpg", "assets/bg.png", "bg.jpg", "bg.png"]
bg_set = False
for p in local_bg_paths:
    if os.path.exists(p):
        add_bg_from_local(p)
        bg_set = True
        break

if not bg_set:
    st.markdown(
        """
        <style>
        .stApp{
            background: linear-gradient(180deg, #071019 0%, #071c20 40%, #05262b 100%);
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



# GLOBAL CSS

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    .stApp, .block-container {
        font-family: 'Inter', sans-serif;
        color: #e6f7f5;
    }

    .overlay {
        position: fixed;
        inset: 0;
        background: rgba(2,8,10,0.45);
        pointer-events: none;
    }

    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.04);
        backdrop-filter: blur(6px);
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(2,8,10,0.5);
        margin-bottom: 20px;
    }

    .title-text {
        font-size: 44px !important;
        font-weight: 800 !important;
        color: #7ee7d6 !important;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #bfeee0;
        margin-bottom: 18px;
    }

    .chart-card {
        padding: 14px;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
    }
    </style>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True,
)



# HEADER

st.markdown('<p class="title-text">🚦  YOLO Traffic Analyzer</p>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload video, pilih model YOLO, dan lakukan analisis trafik lengkap.</div>', unsafe_allow_html=True)



# 1. MODEL SELECTION (CARD)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "🔍 Pilih Model YOLO",
        ["YOLO v8 Final", "YOLO v12 Final"]
    )

    st.markdown('</div>', unsafe_allow_html=True)


model_path_map = {
    "YOLO v8 Final": "best_v8.pt",
    "YOLO v12 Final": "best_v12.pt"
}
selected_model_path = model_path_map[model_choice]


@st.cache_resource
def load_yolo(path):
    return load_model(path)


model = load_yolo(selected_model_path)



# 2. ANALYSIS OPTION (FITUR BARU)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    analysis_choice = st.selectbox(
        "📊 Analisis berdasarkan:",
        ["Per Frame", "Per Detik", "Per Menit", "Per Jam"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

group_key_map = {
    "Per Frame": None,
    "Per Detik": "second",
    "Per Menit": "minute",
    "Per Jam": "hour"
}
group_key = group_key_map[analysis_choice]



# 3. UPLOAD VIDEO (CARD)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    video_file = st.file_uploader("🎥 Upload Video", type=["mp4", "avi", "mov"])
    st.markdown('</div>', unsafe_allow_html=True)


if video_file:

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())

    with st.spinner(f"Processing video menggunakan model {model_choice} ..."):
        df, output_path, augmented, model_names = process_video(model, temp_input.name)

    st.success("✔ Video selesai diproses!")


    # RESAMPLE (Di sini fitur ANALISIS diterapkan)

    df_processed = df.copy()

    if group_key is not None:
        df_processed[group_key] = df_processed[group_key].round().astype(int)

        type_cols = [c for c in df.columns if c.startswith("type_")]

        agg = {c: "sum" for c in type_cols}
        agg["total"] = "sum"

        df_processed = df_processed.groupby(group_key).agg(agg).reset_index()


    # DOWNLOAD SECTION

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#7ee7d6;margin-bottom:6px;">📁 Download Hasil</h3>', unsafe_allow_html=True)

        colA, colB = st.columns([1, 1])
        with colA:
            st.download_button(
                "📄 Download CSV Hasil",
                df_processed.to_csv(index=False).encode("utf-8"),
                "hasil_deteksi.csv"
            )
        with colB:
            with open(output_path, "rb") as f:
                st.download_button(
                    "🎥 Download Video Hasil",
                    f,
                    file_name="video_output.mp4"
                )

        st.markdown('</div>', unsafe_allow_html=True)


    # LINE CHART

    line_data = get_linechart_data(df_processed)

    with st.container():
        st.markdown('<div class="card chart-card">', unsafe_allow_html=True)
        st.markdown(f'<h4 style="color:#bff1e6;margin-bottom:4px;">📈 Jumlah Kendaraan ({analysis_choice})</h4>', unsafe_allow_html=True)

        chart = alt.Chart(line_data).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X(line_data.columns[0] + ":Q", title=analysis_choice),
            y=alt.Y("count:Q", title="Jumlah Kendaraan"),
            tooltip=[line_data.columns[0], "count"]
        ).properties(height=260).interactive()

        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # BAR CHART

    bar_data = get_barchart_data(df_processed)

    with st.container():
        st.markdown('<div class="card chart-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#bff1e6;margin-bottom:4px;">📊 Jumlah Kendaraan per Jenis</h4>', unsafe_allow_html=True)

        bar_df = bar_data.reset_index()
        bar_df.columns = ["type", "count"]

        color_scale = alt.Scale(
            domain=bar_df["type"].tolist(),
            range=["#00c2a8", "#2be0d1", "#32ffd1", "#0fd1b7", "#1aa894"]
        )

        bar = alt.Chart(bar_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("type:N", sort='-y'),
            y="count:Q",
            color=alt.Color("type:N", scale=color_scale, legend=None),
            tooltip=["type", "count"]
        ).properties(height=320)

        st.altair_chart(bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# SCATTER PLOT – KMeans Clustering for Traffic Density (FIXED LABELING)

    from sklearn.cluster import KMeans
    import numpy as np

    with st.container():
        st.markdown('<div class="card chart-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#bff1e6;margin-bottom:4px;">🔵 Scatter Plot: Clustering Kepadatan (KMeans)</h4>', unsafe_allow_html=True)

    # Ambil data total kendaraan
        X = df_processed["total"].values.reshape(-1, 1)


    # KMEANS + SMART CENTROID SORT

        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        raw_labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.flatten()

    # Sort centroid: kecil → besar
        order = np.argsort(centers)

    # Mapping berdasarkan pusat cluster
        centroid_to_label = {
            order[0]: "Lancar",
            order[1]: "Ramai",
            order[2]: "Padat"
        }

        mapped_labels = [centroid_to_label[c] for c in raw_labels]


    # Scatter DF

        scatter_df = pd.DataFrame({
            "index": df_processed[df_processed.columns[0]],
            "total": df_processed["total"],
            "cluster": mapped_labels
    })

    # Scatter plot
        scatter_chart = (
            alt.Chart(scatter_df)
            .mark_circle(size=100)
            .encode(
                x=alt.X("index:Q", title=analysis_choice),
                y=alt.Y("total:Q", title="Total Kendaraan"),
                color=alt.Color("cluster:N", title="Kepadatan"),
                tooltip=["index", "total", "cluster"]
            )
            .properties(height=300)
            .interactive()
    )

    # Plot centroid

        centers_df = pd.DataFrame({
            "index": [scatter_df["index"].median()] * 3,
            "total": centers,
            "cluster": ["Lancar (Centroid)", "Ramai (Centroid)", "Padat (Centroid)"]
        })

        center_chart = (
            alt.Chart(centers_df)
            .mark_point(shape="cross", size=200, strokeWidth=3)
            .encode(
                x="index:Q",
                y="total:Q",
                color=alt.Color("cluster:N", legend=None)
            )
        )

        st.altair_chart(scatter_chart + center_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# KLASIFIKASI DENSITY TABLE BERDASARKAN CLUSTER MAP

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#bff1e6;margin-bottom:6px;">🚘 Klasifikasi Kepadatan</h4>', unsafe_allow_html=True)

        density_table = pd.DataFrame(mapped_labels, columns=["density"]).value_counts().reset_index()
        density_table.columns = ["Density", "Count"]

        st.dataframe(density_table, use_container_width=True)

        st.markdown('<hr style="border-color: rgba(255,255,255,0.04);">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#bff1e6;margin-bottom:6px;">🏷️ Model Label Mapping</h5>', unsafe_allow_html=True)
        st.write(model_names)

        st.markdown('</div>', unsafe_allow_html=True)



    # DATAFRAME

        styled = df_processed.style.set_properties(color="#e6f7f5").format(precision=0)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4 style="color:#bff1e6;margin-bottom:6px;">📋 Tabel Hasil Deteksi</h4>', unsafe_allow_html=True)
            st.dataframe(styled, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


    # AUGMENTATION IMAGES

        st.markdown('<h4 style="color:#bff1e6;margin-top:16px;">🧪 Contoh Augmentasi Data</h4>', unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, img in enumerate(augmented):
            c = cols[idx % 3]
            with c:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Augmentasi #{idx+1}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)