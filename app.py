from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# โหลดโมเดลจากไฟล์ในโปรเจกต์
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('model/gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        married = int(request.form['married'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        input_data = np.array([[gender, age, hypertension, heart_disease, married, avg_glucose_level, bmi]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        result = "มีความเสี่ยงโรคหลอดเลือดสมอง" if prediction == 1 else "ไม่มีความเสี่ยงโรคหลอดเลือดสมอง"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)