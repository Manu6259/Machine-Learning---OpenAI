# HELOC Eligibility Predictor

## Overview
The **HELOC Eligibility Predictor** is a **Streamlit-based web application** that allows users to check their eligibility for a **Home Equity Line of Credit (HELOC)**. It leverages **XGBoost for machine learning predictions**, **SHAP for explainability**, and **OpenAI GPT-4 for personalized denial explanations and improvement suggestions**.

## Features
### ✅ User Input Form
- Users enter their financial data via a **sidebar form**.
- Special placeholder values are recognized and displayed:
  - `-9`: No Bureau Record or No Investigation
  - `-8`: No Usable/Valid Trades or Inquiries
  - `-7`: Condition Not Met (e.g., No Inquiries, No Delinquencies)

### ✅ HELOC Application Processing
- Uses a **pre-trained XGBoost model** to predict loan eligibility.
- **StandardScaler** ensures data normalization before making predictions.

### ✅ SHAP Explainability
- **SHAP (SHapley Additive Explanations)** identifies the top 5 negative impact features in case of denial.
- Displays a **bar chart of SHAP values** for better interpretability.

### ✅ AI-Powered Denial Explanation
- If an application is denied:
  - OpenAI GPT-4 generates a **custom explanation**.
  - The AI provides **specific financial improvement steps** based on user inputs.
  
### ✅ Real-Time Loading Indicator
- A **loading spinner** ensures a smooth user experience while the application processes the request.

## Installation & Setup
### 1️⃣ Prerequisites
Ensure you have **Python 3.8+** and the following dependencies installed:
```sh
pip install streamlit pandas numpy joblib openai shap xgboost scikit-learn
```

### 2️⃣ OpenAI API Key Setup
Obtain an API key from **[OpenAI](https://openai.com/)** and set it in your environment:
```sh
export OPENAI_API_KEY="your-api-key"
```
Alternatively, you can replace `"your-api-key"` in the script directly.

### 3️⃣ Running the Application
Execute the following command in your terminal:
```sh
streamlit run app.py
```

## File Structure
```
📂 HELOC Predictor
│── 📜 Final_app.py              # Main Streamlit app
│── 📜 heloc_data_dictionary.csv  # Feature explanations
│── 📜 xgboost_heloc_model.pkl    # Trained model
│── 📜 scaler_heloc.pkl           # Scaler for preprocessing
```

## Usage Guide
1. **Enter Financial Data**: Provide your credit-related details.
2. **Check Eligibility**: Click the `Check Eligibility` button.
3. **Receive Instant Decision**:
   - If **approved** ✅: Application proceeds for manual review.
   - If **denied** ❌: OpenAI explains why and provides improvement steps.
4. **View Feature Importance**: A SHAP-based **bar chart** shows which factors impacted the decision.

## Next Steps & Future Improvements
- ✅ **Enhance the UI** for better user experience.
- ✅ **Incorporate real-time credit data retrieval** instead of manual entry.
- ✅ **Improve SHAP visualizations** with interactive graphs.
- ✅ **Fine-tune the model** with more recent data.

## Contributors
- **Developer**: Manu Jain
- **Institution**: Simon Business School

## License
This project is for educational purposes and is **not intended for production use**.
