import streamlit as st
from src.generate_plan import generate_business_plan

st.title("Business Plan Generator")

business_type = st.text_input("Enter your business type:")
investment_amount = st.number_input("Investment amount ($):", min_value=0, value=500000)

if st.button("Generate Business Plan"):
    prompt = f"Generate a business plan for a ${investment_amount} {business_type} business"
    with st.spinner("Generating business plan..."):
        response = generate_business_plan(prompt)
        st.write(response)