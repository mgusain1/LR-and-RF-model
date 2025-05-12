# 🏠 California Housing Price Prediction using ML

This project demonstrates a complete machine learning pipeline to predict median house values in California using the California Housing Dataset. It includes exploratory data analysis, feature engineering, model training, evaluation, and visualization — with a focus on explainability and reproducibility.

---

## 📊 Dataset

- Source: [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Target variable: `MedHouseVal` (Median house value in $100,000s)
- Features include:
  - Median income, average rooms, location (latitude/longitude), population, etc.

---

## 🔧 Project Pipeline

### 1. **Exploratory Data Analysis (EDA)**
- Histograms, boxplots, and correlation heatmaps were used to assess skewness, caps, and feature relationships.

📌 _Sample EDA Visuals_  
![image](https://github.com/user-attachments/assets/4ff7e985-1ad4-46c2-a02d-5487d842d30f)

![image](https://github.com/user-attachments/assets/0c50a6f8-692f-439c-8d99-7ce13e0d036f)


---

### 2. **Feature Engineering**
Created new signal-rich features:
- `BedroomsPerRoom`  
- `Popperroom` (Population per room)  
- `IncPerPerson` (Income per person estimate)

📌 _Feature Importance Visualization_  
![image](https://github.com/user-attachments/assets/5d4b81b4-c35f-4d42-b125-dd3c2576b452)


---

### 3. **Model Training**

Two models were trained and evaluated:
- `LinearRegression` (Baseline)
- `RandomForestRegressor` (Advanced, non-linear)

Metrics used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

📌 _Residuals Comparison_  
![image](https://github.com/user-attachments/assets/28b5d955-b5f2-446e-ae97-ea03d93faf07)


📌 _Actual vs Predicted_  
![image](https://github.com/user-attachments/assets/285e58ef-11b9-4036-ae80-0c1b0331baf5)
![image](https://github.com/user-attachments/assets/ccd669c0-0488-429e-a36b-5d6aa392ec6b)


---

## 📈 Results

| Model               | MAE    | RMSE   | R² Score |
|---------------------|--------|--------|----------|
| Linear Regression   | 0.526  | 0.731  | 0.594    |
| Random Forest       | 0.329  | 0.508  | 0.803    |

➡️ **Random Forest outperformed Linear Regression** in all metrics, capturing non-linear patterns and reducing prediction error by ~37%.

---

## 💡 Key Learnings

- Engineered features (like `BedroomsPerRoom`) carried **stronger predictive signals** than raw columns.
- `MedInc` remained the most correlated raw feature with `MedHouseVal`.
- Residual analysis clearly showed tighter error distribution with Random Forest.
- Data truncation (values capped at 5.0) was identified and factored into model evaluation.

---

## 💾 Tools & Libraries Used

- Python
- NumPy / Pandas
- Scikit-learn
- Seaborn / Matplotlib
- Jupyter Notebook

---

## 📁 Directory Structure

