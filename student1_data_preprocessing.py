import pandas as pd
import numpy as np

def get_data():
    # إنشاء بيانات وهمية لتوقع سعر الشقة
    np.random.seed(42)
    n_samples = 200
    
    # الخصائص الأساسية
    area = np.random.randint(80, 300, n_samples)
    bedrooms = np.random.randint(2, 6, n_samples)
    age = np.random.randint(0, 30, n_samples)
    
    # الموقع (3 مناطق مختلفة)
    locations = ['Alexandria', 'Giza', 'Cairo']
    location = np.random.choice(locations, n_samples)
    
    # السعر (معادلة تعتمد على الخصائص)
    base_price = 50000.0
    price = base_price + (area * 1000.0) + (bedrooms * 5000.0) - (age * 800.0)
    
    # إضافة تأثير الموقع
    location_effect = {'Cairo': 400000, 'Giza': 200000, 'Alexandria': 100000}
    price += [location_effect[loc] for loc in location]
    
    # إضافة ضوضاء (Noise) لتبدو البيانات واقعية
    price += np.random.normal(0, 15000, n_samples)
    
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Age': age,
        'Location': location,
        'Price': price
    })
    return df

def preprocess_data(df):
    # Encoding: تحويل النص (الموقع) لأرقام باستخدام pd.get_dummies
    df_processed = pd.get_dummies(df, columns=['Location'], drop_first=True)
    
    # تقسيم البيانات X و y
    X = df_processed.drop('Price', axis=1)
    y = df_processed['Price']
    
    return X, y

if __name__ == "__main__":
    df = get_data()
    print("Data before preprocessing:")
    print(df.head())
    
    X, y = preprocess_data(df)
    print("\nData after preprocessing (Encoding):")
    print(X.head())
