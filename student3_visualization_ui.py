import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# استيراد الأكواد من ملفات الطلاب الآخرين
from student1_data_preprocessing import get_data, preprocess_data
from student2_model_training import train_and_evaluate

# إعدادات الصفحة
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction System")
st.markdown("---")

# جلب البيانات وتدريب الموديل بشكل سريع 
def load_and_train():
    df = get_data()
    X, y = preprocess_data(df)
    model, X_test, y_test, y_pred, mae, r2 = train_and_evaluate(X, y)
    return df, model, X, X_test, y_test, y_pred, mae, r2

df, model, X_full, X_test, y_test, y_pred, mae, r2 = load_and_train()

# تقسيم الواجهة إلى ثلاث تبويبات (لكل طالب جزء يعرض شغله)
tab1, tab2, tab3 = st.tabs(["📊 Data (Student 1)", "🧠 Model & Evaluation (Student 2)", "💻 Prediction Engine (Student 3)"])

with tab1:
    st.header("1. Data Exploration")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Scatter Plot: Area vs Price")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='Area', y='Price', hue='Location', ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("2. Model Evaluation (Linear Regression)")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
    col2.metric("R² Score (Accuracy)", f"{r2 * 100:.2f}%")
    
    st.subheader("Actual vs Predicted Price")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # خط التوقع المثالي
    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Predicted Price")
    st.pyplot(fig2)

with tab3:
    st.header("3. Predict New House Price")
    
    col1, col2 = st.columns(2)
    with col1:
        input_area = st.number_input("Area (sq meters)", min_value=50, max_value=500, value=120)
        input_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    
    with col2:
        input_age = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=5)
        input_location = st.selectbox("Location", ['Alexandria', 'Giza', 'Cairo'])
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Price 🚀", use_container_width=True):
        # تجهيز بيانات الإدخال لتتناسب مع الـ Encoding اللي عمله الطالب 1
        input_data = pd.DataFrame({
            'Area': [input_area],
            'Bedrooms': [input_bedrooms],
            'Age': [input_age],
            'Location_Cairo': [1 if input_location == 'Cairo' else 0],
            'Location_Giza': [1 if input_location == 'Giza' else 0]
        })
        
        # التأكد من ترتيب الأعمدة زي اللي اتدرب عليها الموديل
        input_data = input_data.reindex(columns=X_full.columns, fill_value=0)
        
        # التوقع باستخدام الموديل
        prediction = model.predict(input_data)[0]
        
        st.success(f"💰 The predicted house price is: **${prediction:,.2f}**")