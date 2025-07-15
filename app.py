# %%
import pandas as pd

combined_df = pd.read_csv('home_data.csv')


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Features and target
X = combined_df[['bed', 'bath', 'acre_lot', 'house_size']]
y = combined_df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("🔷 XGBoost Regressor:")
print(f"R^2 Score: {r2_score(y_test, xgb_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, xgb_pred, squared=False):,.2f}")

# %%
import joblib

# Save the trained XGBoost model
joblib.dump(xgb_model, 'xgb_model.pkl')


# %%
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('xgb_model.pkl')

# App title
st.title("🏠 House Price Predictor")
st.markdown("Enter the house features below to predict the estimated price.")

# Input fields
bed = st.number_input("Number of Bedrooms", min_value=0, value=3)
bath = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5, value=2.0)
acre_lot = st.number_input("Lot Size (in acres)", min_value=0.0, step=0.1, value=0.25)
house_size = st.number_input("House Size (in sqft)", min_value=0, value=1800)

# When user clicks button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[bed, bath, acre_lot, house_size]],
                               columns=['bed', 'bath', 'acre_lot', 'house_size'])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")


