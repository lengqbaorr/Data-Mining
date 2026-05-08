import streamlit as st
import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="RUL Prediction System",
    layout="wide"
)


st.title("Aircraft Engine RUL Prediction")
st.markdown("Upload engine sensor sequence to predict Remaining Useful Life (RUL).")


uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Input Data")
    st.dataframe(df.head())

    st.write(f"Rows: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")

    if st.button("Predict RUL"):

        with st.spinner("Calling prediction API..."):

            payload = {
                "data": df.to_dict(orient="records")
            }

            response = requests.post(
                API_URL,
                json=payload
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction completed.")

                st.subheader("Prediction Result")
                st.json(result)

                if "predicted_rul" in result:

                    pred_df = pd.DataFrame(result["predicted_rul"])

                    st.dataframe(pred_df)

            else:

                st.error(f"API Error: {response.status_code}")
                st.text(response.text)