import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

@st.cache_resource
def load_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    model = RandomForestClassifier().fit(X, y)
    return model, iris

model, iris = load_model()

st.title("🌸 Iris Flower Prediction App")

st.sidebar.header("Single Prediction (sliders)")
sl = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sw = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
pl = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 4.0)
pw = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.0)

single_df = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)
st.subheader("Single Input")
st.write(single_df)

pred_single = model.predict(single_df)[0]
st.success(f"Single prediction: **{iris.target_names[pred_single]}**")

st.header("Batch Prediction from CSV")
st.caption("CSV must have columns: " + ", ".join(iris.feature_names))
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    missing = [c for c in iris.feature_names if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        preds = model.predict(df[iris.feature_names])
        labels = [iris.target_names[i] for i in preds]
        out = df.copy()
        out["prediction"] = labels
        st.subheader("Predictions")
        st.dataframe(out.head())
        st.download_button(
            "Download predictions as CSV",
            out.to_csv(index=False),
            file_name="iris_predictions.csv"
        )
