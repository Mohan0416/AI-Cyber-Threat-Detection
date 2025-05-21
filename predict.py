from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

logging.basicConfig(
    filename='predict.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

scaler = StandardScaler()
label_encoder = LabelEncoder()

def preprocess_input(data):
    packet_length = data.get('packet_length', 0)
    duration = data.get('duration', 0)
    source_port = data.get('source_port', 0)
    destination_port = data.get('destination_port', 0)
    bytes_sent = data.get('bytes_sent', 0)
    bytes_received = data.get('bytes_received', 0)
    flow_packets = data.get('flow_packets', 0)
    total_fwd_packets = data.get('total_fwd_packets', 0)
    total_bwd_packets = data.get('total_bwd_packets', 0)
    sub_flow_fwd_bytes = data.get('sub_flow_fwd_bytes', 0)
    sub_flow_bwd_bytes = data.get('sub_flow_bwd_bytes', 0)
    attack_type = data.get('attack_type', 'Normal')

    features = [
        packet_length,
        duration,
        source_port,
        destination_port,
        bytes_sent,
        bytes_received,
        flow_packets,
        total_fwd_packets,
        total_bwd_packets,
        sub_flow_fwd_bytes,
        sub_flow_bwd_bytes
    ]

    scaled_features = scaler.fit_transform([features])
    attack_type_encoded = label_encoder.fit_transform([attack_type])
    final_input = np.hstack([scaled_features, attack_type_encoded.reshape(1, -1)])

    return final_input

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received a prediction request:")
    print(f"Received a prediction request:")
    try:
        data = request.get_json()
        logging.info(f"Request data: {data}")
        print(f"Request data : {data} ")

        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0, 1]
        logging.info(f"Prediction result: {prediction[0]}, Probability : {probability}")
        print(f"Prediction result: {prediction[0]}, Probability : {probability}")

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
