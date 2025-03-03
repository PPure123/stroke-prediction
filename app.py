from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# โหลด Scaler และ Model
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

with open('model/gradient_boosting_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    married = int(request.form['married'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])

    # เตรียมข้อมูลเป็น numpy array
    input_data = np.array([[gender, age, hypertension, heart_disease, married, avg_glucose_level, bmi]])

    # ปรับสเกลข้อมูลด้วย Scaler
    scaled_data = scaler.transform(input_data)

    # ทำนายด้วยโมเดล
    prediction = model.predict(scaled_data)[0]

    # แปลผลลัพธ์
    result = "มีความเสี่ยงโรคหลอดเลือดสมอง" if prediction == 1 else "ไม่มีความเสี่ยงโรคหลอดเลือดสมอง"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)