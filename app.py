import streamlit as st
import pandas as pd
import joblib


model = joblib.load('model.pkl')
le_group = joblib.load('le_group.pkl')
le_cat = joblib.load('le_cat.pkl')

st.title("Sağlamlıq Riskləri Proqnoz Tətbiqi")


fast_blood_sugar = st.number_input('fasting blood sugar', value=90)
cholesterol = st.number_input('Cholesterol', value=200)
ldl = st.number_input('LDL', value=120)
hdl = st.number_input('HDL', value=50)
ast = st.number_input('AST', value=30)
alt = st.number_input('ALT', value=35)
serum_creatinine = st.number_input('serum creatinine', value=1.0)


if st.button("Təxmin et"):
    
    data = {
        'fast_blood_sugar': [fast_blood_sugar],
        'cholesterol': [cholesterol],
        'ldl': [ldl],
        'hdl': [hdl],
        'ast': [ast],
        'alt': [alt],
        'serum_creatinine': [serum_creatinine],
        'waist': [waist]
    }
    df = pd.DataFrame(data)

    
    preds = model.predict(df)
[15:31, 2025-06-17] ChatGPT: risk_group_probs = preds[0]
    risk_cat_probs = preds[1]

    
    risk_group_class = risk_group_probs.argmax(axis=1)[0]
    risk_cat_class = risk_cat_probs.argmax(axis=1)[0]

    risk_group_label = le_group.inverse_transform([risk_group_class])[0]
    risk_cat_label = le_cat.inverse_transform([risk_cat_class])[0]

    st.success(f"Risk Qrupu: {risk_group_label}")
    st.info(f"Risk Kateqoriyası: {risk_cat_label}"
   
