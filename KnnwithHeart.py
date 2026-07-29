from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbor')

col1, col2 = st.columns(2)

with col1:
   st.header("Heart Disease Prediction")
   st.image("./img/heart1.jpg")

with col2:
   st.header("Machine Learning (KNN)")
   st.image("./img/heart2.jpg")

html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Heart3.csv")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

st.subheader("สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

feature = st.selectbox("เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล", dt.columns[:-1])

st.write(f"Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

# ---------- Train Model ----------
X = dt.drop('HeartDisease', axis=1)
y = dt.HeartDisease

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = st.sidebar.slider("เลือกค่า K (จำนวนเพื่อนบ้าน)", min_value=1, max_value=21, value=3, step=2)

Knn_model = KNeighborsClassifier(n_neighbors=k)
Knn_model.fit(X_train_scaled, y_train)

y_pred = Knn_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# ---------- Model Performance ----------
st.sidebar.subheader("ประสิทธิภาพของโมเดล")
st.sidebar.metric("ความแม่นยำ (Accuracy)", f"{acc:.2%}")

with st.sidebar.expander("แสดง Classification Report"):
    st.text(classification_report(y_test, y_pred, target_names=["ไม่เป็นโรค", "เป็นโรค"]))

with st.sidebar.expander("แสดง Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=["ไม่เป็นโรค", "เป็นโรค"],
                yticklabels=["ไม่เป็นโรค", "เป็นโรค"])
    ax_cm.set_ylabel("จริง")
    ax_cm.set_xlabel("ทำนาย")
    st.pyplot(fig_cm)

# ---------- Prediction UI ----------
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

st.write("กรุณากรอกข้อมูลผู้ป่วยด้านล่าง:")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    Age = st.number_input("อายุ (Age)", min_value=28, max_value=77, value=50)
    Sex = st.selectbox("เพศ (Sex)", options=[(0, "หญิง"), (1, "ชาย")], format_func=lambda x: x[1])[0]
    ChestPainType = st.selectbox("ประเภทเจ็บหน้าอก (ChestPainType)", options=[(1, "Typical Angina"), (2, "Atypical Angina"), (3, "Non-anginal Pain"), (4, "Asymptomatic")], format_func=lambda x: x[1])[0]

with col_in2:
    RestingBP = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=80, max_value=200, value=120)
    Cholesterol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=603, value=200)
    FastingBS = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl (FastingBS)", options=[(0, "ไม่ใช่"), (1, "ใช่")], format_func=lambda x: x[1])[0]

with col_in3:
    RestingECG = st.selectbox("ผลคลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)", options=[(1, "Normal"), (2, "ST-T Wave Abnormality"), (3, "Left Ventricular Hypertrophy")], format_func=lambda x: x[1])[0]
    MaxHR = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=60, max_value=202, value=140)
    ExerciseAngina = st.selectbox(" angina ขณะออกกำลังกาย (ExerciseAngina)", options=[(0, "ไม่ใช่"), (1, "ใช่")], format_func=lambda x: x[1])[0]

col_in4, col_in5, _ = st.columns(3)
with col_in4:
    Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-2.6, max_value=6.2, value=0.0, step=0.1)
with col_in5:
    ST_Slope = st.selectbox("ความชันของ ST (ST_Slope)", options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], format_func=lambda x: x[1])[0]

if st.button("ทำนายผล"):
   x_input = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
   x_input_scaled = scaler.transform(x_input)
   out = Knn_model.predict(x_input_scaled)

   if out[0] == 1:
    st.image("./img/heart1.jpg")
    st.error("ผลลัพธ์: เป็นโรคหัวใจ")
   else:
    st.image("./img/heart2.jpg")
    st.success("ผลลัพธ์: ไม่เป็นโรคหัวใจ")
