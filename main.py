import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ✅ Load the trained model and encoder
model = joblib.load('model.pkl')
c_api_encoder = joblib.load('c_api_encoder.pkl')

# 🧭 Sidebar Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Gender Prediction", "Graphs"]
)

# 🎯 Page 1: Prediction
if page == "Gender Prediction":
    st.title("Gender Prediction App")
    st.write(
        "Fill the features and click Predict to estimate gender."
    )

    # 🧠 Input widgets
    C_api = st.selectbox('C_api', c_api_encoder.classes_.tolist())
    C_man = st.number_input('C_man', value=0)
    E_NEds = st.number_input('E_NEds', value=0)
    E_Bpag = st.number_input('E_Bpag', value=0)
    NEds = st.number_input('NEds', value=0)
    NDays = st.number_input('NDays', value=0)
    NActDays = st.number_input('NActDays', value=0)
    NPages = st.number_input('NPages', value=0)
    NPcreated = st.number_input('NPcreated', value=0)
    pagesWomen = st.number_input('pagesWomen', value=0)
    wikiprojWomen = st.number_input('wikiprojWomen', value=0)
    ns_user = st.number_input('ns_user', value=0)
    ns_wikipedia = st.number_input('ns_wikipedia', value=0)
    ns_talk = st.number_input('ns_talk', value=0)
    ns_userTalk = st.number_input('ns_userTalk', value=0)
    ns_content = st.number_input('ns_content', value=0)
    weightIJ = st.number_input('weightIJ', value=0.0)
    NIJ = st.number_input('NIJ', value=0)

    # 🧠 Prediction
    if st.button('Predict'):
        data = pd.DataFrame([[
            c_api_encoder.transform([C_api])[0], C_man, E_NEds, E_Bpag,
            NEds, NDays, NActDays, NPages, NPcreated, pagesWomen, wikiprojWomen,
            ns_user, ns_wikipedia, ns_talk, ns_userTalk, ns_content, weightIJ, NIJ
        ]], columns=[
            'C_api', 'C_man', 'E_NEds', 'E_Bpag',
            'NEds', 'NDays', 'NActDays', 'NPages',
            'NPcreated', 'pagesWomen', 'wikiprojWomen',
            'ns_user', 'ns_wikipedia', 'ns_talk',
            'ns_userTalk', 'ns_content', 'weightIJ', 'NIJ'
        ])
        pred = model.predict(data)[0]
        gender_label = 'Male' if pred == 1 else 'Female'
        st.success(f'Predicted gender: {gender_label}')

# 📊 Page 2: Graphs
elif page == "Graphs":
    st.title("Graphs and Analysis")

    st.write(
        "Here are multiple example graphs — you can replace with your own data!"
    )

    # Dummy example data
    categories = ['NEds', 'NDays', 'NPages', 'NActDays']
    values = [10, 30, 45, 20]

    # 📊 Bar Chart
    st.subheader("Bar Chart Example")
    fig_bar, ax_bar = plt.subplots(figsize=(6,4))
    ax_bar.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_bar.set_title('Feature Distribution (Bar)')
    ax_bar.set_ylabel('Count')
    st.pyplot(fig_bar)

    # 🥧 Pie Chart
    st.subheader("Pie Chart Example")
    fig_pie, ax_pie = plt.subplots(figsize=(6,4))
    ax_pie.pie(values, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_pie.set_title('Feature Distribution (Pie)')
    st.pyplot(fig_pie)

    # 📈 Line Plot
    st.subheader("Line Plot Example")
    fig_line, ax_line = plt.subplots(figsize=(6,4))
    ax_line.plot(categories, values, marker='o', color='purple')
    ax_line.set_title('Feature Trend (Line)')
    ax_line.set_ylabel('Count')
    ax_line.grid(True)
    st.pyplot(fig_line)

    st.success(
        "✅ You can replace these dummy values with your own data and customize as needed!"
    )
