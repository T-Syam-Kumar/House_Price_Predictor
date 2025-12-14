import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model & accuracy
with open("model.pkl", "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
accuracy = saved_data["accuracy"]

# Load dataset (for graphs)
data = pd.read_csv("house_data.csv")
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Prediction App")

st.markdown(f"### 📊 Model Accuracy (R² Score): **{accuracy:.2f}**")

st.divider()

# ---------------- User Inputs ----------------
st.subheader("Enter House Details")

area = st.slider("Area (sqft)", 500, 3000, 1000)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated House Price: ₹ {prediction[0]:,.2f}")

st.divider()

# ---------------- Graphs ----------------
st.subheader("📈 Data Visualizations")

# 1. Area vs Price
fig1, ax1 = plt.subplots()
ax1.scatter(data['area'], y)
ax1.set_xlabel("Area (sqft)")
ax1.set_ylabel("Price")
ax1.set_title("Area vs Price")
st.pyplot(fig1)

# 2. Actual vs Predicted
y_pred_all = model.predict(X)

fig2, ax2 = plt.subplots()
ax2.scatter(y, y_pred_all)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--')
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual vs Predicted Prices")
st.pyplot(fig2)

# 3. Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_
})

fig3, ax3 = plt.subplots()
ax3.bar(feature_importance["Feature"], feature_importance["Importance"])
ax3.set_title("Feature Importance")
st.pyplot(fig3)
