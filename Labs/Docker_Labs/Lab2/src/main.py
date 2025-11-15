from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load the initial trained CIFAR-10 model
model = tf.keras.models.load_model('my_model.keras')
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.route('/')
def home():
    return "Welcome to the CIFAR-10 Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            if 'image' not in data:
                return jsonify({"error": "Missing 'image' key in JSON"}), 400

            input_array = np.array(data['image'], dtype=np.float32)
            if input_array.shape != (32, 32, 3):
                return jsonify({"error": "Input image must be shape (32,32,3)"}), 400

            input_array = input_array / 255.0
            input_data = np.expand_dims(input_array, axis=0)

            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method", 405

@app.route('/train', methods=['POST'])
def train():
    """
    Retrain the model on CIFAR-10 dataset.
    Accepts optional JSON payload for epochs and batch_size:
    {"epochs": 5, "batch_size": 64}
    """
    try:
        # Parse optional hyperparameters
        data = request.get_json(force=True) if request.data else {}
        epochs = int(data.get("epochs", 5))
        batch_size = int(data.get("batch_size", 64))

        # Load CIFAR-10 dataset
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train, X_test = X_train.astype("float32") / 255.0, X_test.astype("float32") / 255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()

        # Retrain the model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test)
        )

        # Save the updated model
        model.save("my_model.keras")
        model.save("saved_model")

        return jsonify({"message": f"Model retrained for {epochs} epochs, batch_size={batch_size}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
