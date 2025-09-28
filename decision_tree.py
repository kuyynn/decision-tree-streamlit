import streamlit as st
st.set_page_config(page_title="Decision Tree Lab", layout="wide")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# CSS Styling
# ----------------------
st.markdown("""
<style>
/* Background utama */
.stApp {
    background: linear-gradient(to right, #010030, #7226ff);
    color: #f5f5f5;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #121212;
    padding: 1rem;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    font-size: 1.3rem;
    font-weight: 700;
    color: white;
}

/* Style radio container */
div[role="radiogroup"] > label {
    background-color: rgba(255,255,255,0.05);
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 6px;
    color: #bbb;
    display: flex;
    align-items: center;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}
div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.1);
}

/* Active radio (dipilih) */
div[role="radiogroup"] > label[data-selected="true"] {
    background: linear-gradient(to right, #4a3aff, #9f6bff);
    color: white !important;
    font-weight: 600;
}

/* Divider */
.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,0.15);
    margin: 15px 0;
}

/* Selectbox */
.stSelectbox label {
    color: #aaa !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Helper Functions
# ----------------------
def preprocess_basic(df, target_column, drop_cols=None):
    df_proc = df.copy()
    if drop_cols:
        df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns], errors='ignore')
    # Fill missing values
    for col in df_proc.columns:
        if col == target_column:
            continue
        if df_proc[col].dtype.kind in 'biufc':
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode().iloc[0] if not df_proc[col].mode().empty else "")
    X = pd.get_dummies(df_proc.drop(columns=[target_column]), drop_first=True)
    y = df_proc[target_column]
    return X, y

def safe_class_names(y):
    return [str(x) for x in np.unique(y)]

def plot_tree_figure(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_tree(clf, filled=True, feature_names=feature_names,
              class_names=class_names, ax=ax, rounded=True)
    st.pyplot(fig)

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title("Decision Tree Lab")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Menampilkan semua data",
        "Grouping",
        "Training dan Testing",
        "Decision Tree",
        "Prediksi Akurasi",
        "Classification Report",
        "Visualisasi Decision Tree",
        "Klasifikasi / Prediksi Input"
    ]
)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# ----------------------
# Session State
# ----------------------
if "df" not in st.session_state:
    st.session_state["df"] = None

# ----------------------
# Halaman 1: Upload & tampilkan data
# ----------------------
if menu == "Menampilkan semua data":
    st.subheader("Upload & Tampilkan Data")
    uploaded_file = st.file_uploader("Upload Dataset (CSV/XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            st.session_state["df"] = pd.read_csv(uploaded_file)
        else:
            st.session_state["df"] = pd.read_excel(uploaded_file)
    else:
        # kalau file dihapus -> reset state
        st.session_state["df"] = None

    # tampilkan kalau ada data
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.dataframe(df.head(200))
        st.write("**Ringkasan dataset:**")
        st.write(df.describe(include='all').transpose())
    else:
        st.info("📂 Silakan upload file CSV atau Excel terlebih dahulu.")

# ----------------------
# Halaman lain
# ----------------------
else:
    if st.session_state["df"] is None:
        st.warning("Silakan upload dataset dulu di halaman 'Menampilkan semua data'")
        st.stop()
    df = st.session_state["df"]

    st.sidebar.subheader("⚙️ Parameter Model")
    target_column = st.sidebar.selectbox("Kolom target (label)", df.columns.tolist())
    drop_cols = st.sidebar.multiselect("Drop columns (opsional)", options=[c for c in df.columns if c != target_column])
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

    # Train Model
    X, y = preprocess_basic(df, target_column, drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if len(np.unique(y))>1 else None
    )
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Menu logic
    if menu == "Grouping":
        st.subheader("Grouping")
        st.write(df.groupby(target_column).size())

    elif menu == "Training dan Testing":
        st.subheader("Training dan Testing")
        st.write(f"Train set: {X_train.shape[0]} rows")
        st.write(f"Test set: {X_test.shape[0]} rows")

    elif menu == "Decision Tree":
        st.subheader("Decision Tree")
        st.success("Model berhasil dilatih.")

    elif menu == "Prediksi Akurasi":
        st.subheader("Prediksi Akurasi")
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: **{acc:.4f}**")

    elif menu == "Classification Report":
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    elif menu == "Visualisasi Decision Tree":
        st.subheader("Visualisasi Decision Tree")
        plot_tree_figure(clf, feature_names=X.columns.tolist(), class_names=safe_class_names(y))

    elif menu == "Klasifikasi / Prediksi Input":
        st.subheader("Klasifikasi / Prediksi Input")
        input_vals = {}
        with st.form("manual_predict"):
            for col in X.columns:
                if X[col].dtype.kind in 'biufc':
                    val = st.number_input(col, value=float(X[col].mean()))
                else:
                    val = st.text_input(col, value=str(X[col].iloc[0]))
                input_vals[col] = val
            submitted = st.form_submit_button("Predict")
        if submitted:
            row = pd.DataFrame([input_vals])
            for col in row.columns:
                if col in X.columns and X[col].dtype.kind in 'biufc':
                    row[col] = pd.to_numeric(row[col], errors='coerce')
            for col in X.columns:
                if col not in row.columns:
                    row[col] = 0
            row = row[X.columns]
            pred = clf.predict(row)[0]
            st.success(f"Hasil Prediksi: **{pred}**")
