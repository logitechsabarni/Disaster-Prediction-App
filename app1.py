import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
import folium
from streamlit_folium import folium_static
from sklearn.metrics import confusion_matrix

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Disaster Risk Prediction System",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ AI Disaster Risk Prediction System")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Cyclone Risk", "Drought Risk"]
)

# ------------------------------------------------
# API KEY
# ------------------------------------------------

API_KEY = "YOUR_OPENWEATHER_API_KEY"

# ------------------------------------------------
# FETCH WEATHER DATA
# ------------------------------------------------

def fetch_weather(city):

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    wind_speed = data["wind"]["speed"]

    pressure = data["main"]["pressure"]

    cloud = data["clouds"]["all"]

    rainfall = data.get("rain",{}).get("1h",0)

    return lat,lon,temp,humidity,wind_speed,pressure,cloud,rainfall

# =================================================
# CYCLONE SECTION
# =================================================

elif page == "Cyclone Risk":

    st.header("🌪 Cyclone Risk Prediction")

    city = st.text_input("Enter City")

    if st.button("Run Cyclone Assessment"):

        weather = fetch_weather(city)

        if weather is None:

            st.error("City not found")

        else:

            lat,lon,temp,humidity,wind,pressure,cloud,rain = weather

            st.success(f"Location found: Latitude {lat} | Longitude {lon}")

            # Load Model
            model = joblib.load("models/cyclone_model.pkl")

            # Input Features
            input_data = np.array([[wind,pressure,temp,humidity,cloud]])

            prediction = model.predict(input_data)[0]

            risk_labels = ["Low","Medium","High"]

            risk = risk_labels[prediction]

            if risk=="High":
                st.error("🚨 HIGH CYCLONE RISK")

            elif risk=="Medium":
                st.warning("⚠ MODERATE CYCLONE RISK")

            else:
                st.success("✅ LOW CYCLONE RISK")

            # -----------------------------------
            # MAP
            # -----------------------------------

            st.subheader("📍 Location Map")

            m = folium.Map(location=[lat,lon],zoom_start=8)

            color="green"

            if risk=="Medium":
                color="orange"

            if risk=="High":
                color="red"

            folium.CircleMarker(
                location=[lat,lon],
                radius=12,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"{city} - {risk} Risk"
            ).add_to(m)

            folium_static(m,width=1200,height=400)

            # -----------------------------------
            # 24 HOUR PROJECTION
            # -----------------------------------

            st.subheader("📈 24 Hour Wind Projection")

            hours = np.arange(1,25)

            projection = wind + np.random.normal(0,1.5,24)

            fig1,ax1 = plt.subplots()

            ax1.plot(hours,projection,marker="o")

            ax1.set_xlabel("Hour")

            ax1.set_ylabel("Wind Speed")

            st.pyplot(fig1)

            # -----------------------------------
            # FEATURE IMPORTANCE
            # -----------------------------------

            st.subheader("📊 Feature Importance")

            features = ["Wind Speed","Pressure","Temperature","Humidity","Cloud"]

            importance = model.feature_importances_

            fig2,ax2 = plt.subplots()

            ax2.barh(features,importance)

            st.pyplot(fig2)

            # -----------------------------------
            # HEATMAP
            # -----------------------------------

            st.subheader("🔥 Correlation Heatmap")

            data = pd.read_csv("datasets/cyclone_data.csv")

            fig3,ax3 = plt.subplots()

            sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

            st.pyplot(fig3)

            # -----------------------------------
            # CONFUSION MATRIX
            # -----------------------------------

            st.subheader("🧠 Confusion Matrix")

            X = data.drop("risk",axis=1)

            y = data["risk"]

            y_pred = model.predict(X)

            cm = confusion_matrix(y,y_pred)

            fig4,ax4 = plt.subplots()

            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

            ax4.set_xlabel("Predicted")

            ax4.set_ylabel("Actual")

            st.pyplot(fig4)

# =================================================
# DROUGHT SECTION
# =================================================

elif page == "Drought Risk":

    st.header("🌵 Drought Risk Prediction")

    city = st.text_input("Enter City")

    if st.button("Run Drought Assessment"):

        weather = fetch_weather(city)

        if weather is None:

            st.error("City not found")

        else:

            lat,lon,temp,humidity,wind,pressure,cloud,rain = weather

            st.success(f"Location found: Latitude {lat} | Longitude {lon}")

            # Load Model
            model = joblib.load("models/drought_model.pkl")

            # Simulated environmental parameters
            soil = np.random.uniform(5,50)

            groundwater = np.random.uniform(1,20)

            input_data = np.array([[rain,temp,humidity,soil,groundwater]])

            prediction = model.predict(input_data)[0]

            risk_labels = ["Low","Medium","High"]

            risk = risk_labels[prediction]

            if risk=="High":

                st.error("🚨 HIGH DROUGHT RISK")

            elif risk=="Medium":

                st.warning("⚠ MODERATE DROUGHT RISK")

            else:

                st.success("✅ LOW DROUGHT RISK")

            # -----------------------------------
            # MAP
            # -----------------------------------

            st.subheader("📍 Location Map")

            m = folium.Map(location=[lat,lon],zoom_start=8)

            color="green"

            if risk=="Medium":
                color="orange"

            if risk=="High":
                color="red"

            folium.CircleMarker(
                location=[lat,lon],
                radius=12,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"{city} - {risk} Risk"
            ).add_to(m)

            folium_static(m,width=1200,height=400)

            # -----------------------------------
            # 24 HOUR PROJECTION
            # -----------------------------------

            st.subheader("📈 24 Hour Temperature Projection")

            hours = np.arange(1,25)

            projection = temp + np.random.normal(0,1,24)

            fig1,ax1 = plt.subplots()

            ax1.plot(hours,projection,marker="o")

            ax1.set_xlabel("Hour")

            ax1.set_ylabel("Temperature")

            st.pyplot(fig1)

            # -----------------------------------
            # FEATURE IMPORTANCE
            # -----------------------------------

            st.subheader("📊 Feature Importance")

            features = ["Rainfall","Temperature","Humidity","Soil Moisture","Groundwater"]

            importance = model.feature_importances_

            fig2,ax2 = plt.subplots()

            ax2.barh(features,importance)

            st.pyplot(fig2)

            # -----------------------------------
            # HEATMAP
            # -----------------------------------

            st.subheader("🔥 Correlation Heatmap")

            data = pd.read_csv("datasets/drought_data.csv")

            fig3,ax3 = plt.subplots()

            sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

            st.pyplot(fig3)

            # -----------------------------------
            # CONFUSION MATRIX
            # -----------------------------------

            st.subheader("🧠 Confusion Matrix")

            X = data.drop("risk",axis=1)

            y = data["risk"]

            y_pred = model.predict(X)

            cm = confusion_matrix(y,y_pred)

            fig4,ax4 = plt.subplots()

            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

            ax4.set_xlabel("Predicted")

            ax4.set_ylabel("Actual")

            st.pyplot(fig4)
elif page == "Model Insights":
    st.title("🧠 Model Insights & Performance")

    hazard = st.selectbox(
        "Select Hazard to Inspect",
        ["Flood", "Heatwave", "Cyclone", "Drought"]
    )

    # ==========================================================
    # FLOOD
    # ==========================================================
    if hazard == "Flood":
        st.subheader("🌊 Flood Risk: XGBoost Classifier")
        
        cols = st.columns(4)
        metrics = {
            "Accuracy": 0.9885,
            "Precision": 0.98,
            "Recall": 0.9849,
            "F1 Score": 0.9849
        }

        for col, (label, val) in zip(cols, metrics.items()):
            col.metric(label, f"{val * 100:.2f}%")

        st.divider()

        with st.container(border=True):

            left_col, right_col = st.columns([1, 1])
            
            with left_col:

                st.markdown("#### 📊 Feature Importance")

                feature_names = [
                    "Rainfall",
                    "River Discharge",
                    "Elevation",
                    "Water Level",
                    "Historical Flood"
                ]

                importances = flood_model.feature_importances_

                importance_data = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=True)

                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues"
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:

                st.markdown(
                    "#### 🎯 Classification Accuracy",
                    help=(
                        "- **Diagonal:** Correct predictions.\n"
                        "- **Off-Diagonal:** Model misclassifications.\n"
                        "- **Goal:** High values in the diagonal indicate a reliable model."
                    )
                )

                st.image(
                    "assets/confusion_matrix_flood.png",
                    caption="Class Labels: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk"
                )

    # ==========================================================
    # HEATWAVE
    # ==========================================================
    elif hazard == "Heatwave":

        st.subheader("🔥 Heatwave Risk: Random Forest Classifier")
        
        cols = st.columns(5)

        metrics = {
            "Accuracy": 0.9660,
            "Precision": 0.9189,
            "Recall": 0.9016,
            "F1 Score": 0.9102,
            "ROC-AUC": 0.9923
        }

        for col, (label, val) in zip(cols, metrics.items()):
            col.metric(label, f"{val * 100:.2f}%")

        st.divider()

        with st.container(border=True):

            left_col, right_col = st.columns([1, 1])

            with left_col:

                st.markdown("#### 📊 Feature Importance")

                feature_names = [
                    "Min Temp",
                    "Max Humidity",
                    "Min Humidity",
                    "Wind Speed",
                    "Pressure",
                    "Rainfall"
                ]

                importances = heatwave_model.feature_importances_

                importance_data = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=True)

                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Oranges"
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:

                st.markdown(
                    "#### 🎯 Classification Accuracy",
                    help=(
                        "- **Diagonal:** Correct predictions.\n"
                        "- **Off-Diagonal:** Model misclassifications.\n"
                        "- **Goal:** High values in the diagonal indicate a reliable model."
                    )
                )

                st.image(
                    "assets/confusion_matrix_heatwave.png",
                    caption="Class Labels: 0 = No Heatwave, 1 = Heatwave"
                )

    # ==========================================================
    # CYCLONE
    # ==========================================================
    elif hazard == "Cyclone":

        st.subheader("🌪 Cyclone Risk: Random Forest Classifier")

        cyclone_model = joblib.load("models/cyclone_model.pkl")
        cyclone_data = pd.read_csv("datasets/cyclone_data.csv")

        X = cyclone_data.drop("risk", axis=1)
        y = cyclone_data["risk"]

        cols = st.columns(4)

        cols[0].metric("Dataset Size", len(cyclone_data))
        cols[1].metric("Features", X.shape[1])
        cols[2].metric("Classes", 3)
        cols[3].metric("Model Type", "Random Forest")

        st.divider()

        with st.container(border=True):

            left_col, right_col = st.columns(2)

            with left_col:

                st.markdown("#### 📊 Feature Importance")

                feature_names = [
                    "Wind Speed",
                    "Pressure",
                    "Temperature",
                    "Humidity",
                    "Cloud Coverage"
                ]

                importances = cyclone_model.feature_importances_

                importance_data = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance")

                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Reds"
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:

                st.markdown("#### 🎯 Confusion Matrix")

                from sklearn.metrics import confusion_matrix
                import seaborn as sns

                y_pred = cyclone_model.predict(X)

                cm = confusion_matrix(y, y_pred)

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Reds",
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"],
                    ax=ax
                )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                st.pyplot(fig)

        st.divider()

        st.markdown("### 🔥 Feature Correlation Heatmap")

        fig, ax = plt.subplots()

        sns.heatmap(
            cyclone_data.corr(),
            cmap="coolwarm",
            annot=True,
            ax=ax
        )

        st.pyplot(fig)

# ======================================================
# DROUGHT INSIGHTS
# ======================================================

    elif hazard == "Drought":

        st.subheader("🌵 Drought Risk: Random Forest Classifier")

        drought_model = joblib.load("models/drought_model.pkl")
        drought_data = pd.read_csv("datasets/drought_data.csv")

        X = drought_data.drop("risk", axis=1)
        y = drought_data["risk"]

        cols = st.columns(4)

        cols[0].metric("Dataset Size", len(drought_data))
        cols[1].metric("Features", X.shape[1])
        cols[2].metric("Classes", 3)
        cols[3].metric("Model Type", "Random Forest")

        st.divider()

        with st.container(border=True):

            left_col, right_col = st.columns(2)

            with left_col:

                st.markdown("#### 📊 Feature Importance")

                feature_names = [
                    "Rainfall",
                    "Temperature",
                    "Humidity",
                    "Soil Moisture",
                    "Groundwater Level"
                ]

                importances = drought_model.feature_importances_

                importance_data = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance")

                fig = px.bar(
                    importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Oranges"
                )

                st.plotly_chart(fig, use_container_width=True)

            with right_col:

                st.markdown("#### 🎯 Confusion Matrix")

                from sklearn.metrics import confusion_matrix
                import seaborn as sns

                y_pred = drought_model.predict(X)

                cm = confusion_matrix(y, y_pred)

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Oranges",
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"],
                    ax=ax
                )

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                st.pyplot(fig)

        st.divider()

        st.markdown("### 🔥 Feature Correlation Heatmap")

        fig, ax = plt.subplots()

        sns.heatmap(
            drought_data.corr(),
            cmap="coolwarm",
            annot=True,
            ax=ax
        )

        st.pyplot(fig)
