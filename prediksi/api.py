import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Impor CORS

app = Flask(__name__)

# Izinkan CORS hanya untuk origin website Anda dan endpoint /predict
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5503"}}) 

# Muat model prediksi
with open('model_svm_prediksi.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tahun = data['year'] # Pastikan key-nya sesuai dengan yang dikirim dari JavaScript (year)
    prediksi = model.predict([[tahun]])[0]
    return jsonify({'prediction': prediksi}) # Pastikan key-nya sesuai dengan yang diharapkan JavaScript (prediction)

@app.route('/')  # Rute untuk URL root
def index():
    return jsonify({'message': 'Selamat datang di API Prediksi Luasan!'}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)