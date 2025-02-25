import streamlit as st
import pandas as pd
import numpy as np
import joblib
import openai
import shap
import xgboost
from sklearn.preprocessing import StandardScaler

# OpenAI API Key (ensure this is set securely in your environment)
openai.api_key = "YOUR API KEY"

# Load trained model and scaler
model = joblib.load("xgboost_heloc_model.pkl")
scaler = joblib.load("scaler_heloc.pkl")

# Load data dictionary for feature explanations
data_dict = pd.read_csv("heloc_data_dictionary.csv")
data_dict = data_dict.set_index("Variable Names")

def get_feature_description(feature):
    """Retrieve the human-readable description of a feature."""
    return data_dict.loc[feature, "Description"] if feature in data_dict.index else feature

st.title("HELOC Eligibility Predictor")
st.sidebar.header("Enter Your Financial Information")

feature_names = [
    "ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen",
    "AverageMInFile", "NumSatisfactoryTrades", "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec", "PercentTradesNeverDelq",
    "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M", "NumTradesOpeninLast12M",
    "PercentInstallTrades", "MSinceMostRecentInqexcl7days",
    "NumInqLast6M", "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden",
    "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance"
]

# Meaning of special placeholders
missing_value_explanations = {
    -9: "No Bureau Record or No Investigation",
    -8: "No Usable/Valid Trades or Inquiries",
    -7: "Condition Not Met (e.g., No Inquiries, No Delinquencies)"
}

# Create input fields and store user input
user_input = []
missing_values_report = []

for feature in feature_names:
    value = st.sidebar.text_input(f"{feature}", value="", key=feature)
    if value.strip() != "":
        num_value = float(value)
        user_input.append(num_value)
        if num_value in missing_value_explanations:
            missing_values_report.append(f"- **{feature}**: {missing_value_explanations[num_value]}")
    else:
        user_input.append(np.nan)

# Convert input to array and scale it
user_input_array = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input_array)

if st.sidebar.button("Check Eligibility"):
    if np.isnan(user_input_array).any():
        st.error("❌ Please fill in all fields before checking eligibility.")
    else:
        prediction = model.predict(user_input_scaled)[0]
        explainer = shap.Explainer(model)
        shap_values = explainer(user_input_scaled)
        
        if prediction == 1:
            st.success("✅ Your HELOC application will be sent to a loan officer for review.")
        else:
            st.error("❌ Your HELOC application is denied.")
            
            # Display missing values (if any)
            if missing_values_report:
                st.write("### Missing Inputs Detected:")
                for item in missing_values_report:
                    st.write(item)

            # Extract top negative impact features
            shap_impacts = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_values.values[0]
            })
            shap_impacts = shap_impacts.sort_values(by="SHAP Value", ascending=True)
            top_negative_factors = shap_impacts.head(5)

            # Construct OpenAI prompt
            openai_prompt = """
            A HELOC loan application was denied. The following financial indicators contributed to the denial:
            """
            for _, row in top_negative_factors.iterrows():
                feature_desc = get_feature_description(row["Feature"])
                openai_prompt += f"- {row['Feature']}: {row['SHAP Value']:.2f} ({feature_desc})\n"
            
            openai_prompt += "\nProvide a detailed explanation for the denial and suggest specific actions the customer can take to improve their chances of approval."
            
            # Query OpenAI for dynamic explanations using latest API version
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial expert providing clear and precise loan eligibility explanations."},
                    {"role": "user", "content": openai_prompt}
                ]
            )

            openai_explanation = response.choices[0].message.content

            # Display OpenAI-generated explanation
            st.write("### Explanation for Denial:")
            st.write(openai_explanation)
