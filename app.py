from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.losses import MeanSquaredError 
from datetime import datetime 

app = Flask(__name__)
CORS(app)

# 🔹 Load Model dan Scaler dengan Error Handling
MODEL_PATH = "model_pmv.h5"
SCALER_PATH = "scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Error saat memuat model atau scaler: {e}")
    model, scaler = None, None

# 🔹 Variabel global untuk menyimpan data terbaru
latest_sensor_data = {
    "temperature": None,
    "humidity": None,
    "air_flow": None,
    "mrt": None,
    "pmv": None,
    "thermal_comfort": None
}

# 🔹 Variabel global untuk menyimpan riwayat data sensor
sensor_data_history = []

# 🔹 Fungsi untuk menentukan status Thermal Comfort (TC)
def get_thermal_comfort_status(pmv):
    if   -1.0 < pmv < 1.0:
        return "Normal"
    elif -2.0 < pmv <= -1.0:
        return "A bit Cool"
    elif -3.0 < pmv <= -2.0:
        return "Cool"
    elif pmv <= -3.0:
        return "Cold"
    elif 1.0 <= pmv < 2.0:
        return "A bit Warm"
    elif 2.0 <= pmv < 3.0:
        return "Warm"
    else:
        return "Hot"

@app.route('/')
def home():
    return jsonify({"message": "PMV Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model atau scaler tidak dimuat dengan benar!"}), 500

    try:
        data = request.get_json()
        required_fields = ["temperature", "humidity", "air_flow", "mrt"]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Parameter berikut harus diisi: {', '.join(missing_fields)}"}), 400
        
        try:
            suhu = float(data["temperature"])
            humidity = float(data["humidity"])
            air_flow = float(data["air_flow"])
            mrt = float(data["mrt"])
        except ValueError:
            return jsonify({"error": "Semua parameter harus berupa angka!"}), 400

        # 🔹 Format input sesuai urutan (temperature, humidity, air_flow, mrt)
        input_data = np.array([[suhu, humidity, air_flow, mrt]])
        input_scaled = scaler.transform(input_data)

        # 🔹 Prediksi PMV
        prediction = model.predict(input_scaled)
        pmv_value = float(prediction[0][0])

        # 🔹 Batasi nilai PMV dalam rentang -3 hingga 3
        pmv_value = np.clip(pmv_value, -3, 3)

        # 🔹 Dapatkan status Thermal Comfort (TC)
        tc_status = get_thermal_comfort_status(pmv_value)

        # 🔹 Simpan data terbaru ke variabel global
        global latest_sensor_data, sensor_data_history
        latest_sensor_data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "temperature": suhu,
            "humidity": humidity,
            "air_flow": air_flow,
            "mrt": mrt,
            "pmv": pmv_value,
            "thermal_comfort": tc_status
        }

        # 🔹 Tambahkan data terbaru ke riwayat
        sensor_data_history.append(latest_sensor_data)

        # 🔹 Batasi riwayat data (misalnya, hanya menyimpan 100 data terbaru)
        if len(sensor_data_history) > 100:
            sensor_data_history.pop(0)

        print(f"📊 Input: {input_data}, Scaled: {input_scaled}, Predicted PMV: {pmv_value}, TC: {tc_status}")

        return jsonify({"pmv": pmv_value, "thermal_comfort": tc_status})

    except Exception as e:
        print(f"❌ Error pada prediksi: {e}")
        return jsonify({"error": str(e)}), 500

# 🔹 Endpoint untuk mengambil data terbaru
@app.route('/get-sensor-data', methods=['GET'])
def get_sensor_data():
    return jsonify(latest_sensor_data)

# 🔹 Endpoint untuk mengambil riwayat data sensor
@app.route('/get-sensor-data-history', methods=['GET'])
def get_sensor_data_history():
    return jsonify(sensor_data_history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)