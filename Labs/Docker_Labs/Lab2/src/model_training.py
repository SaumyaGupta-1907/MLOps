import argparse
import json
import logging
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix

def setup_logging():
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Logging is set up.")

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Normalize pixel values to [0,1]
    X_train, X_test = X_train.astype("float32") / 255.0, X_test.astype("float32") / 255.0
    logging.info(f"CIFAR-10 data loaded. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train.flatten(), y_test.flatten()

def build_small_resnet(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    # residual block
    residual = x

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    residual = layers.Conv2D(64, (1,1), padding='same')(residual)
    residual = layers.BatchNormalization()(residual)

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logging.info("Small ResNet model built and compiled.")
    return model

def evaluate_and_save_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_classes).tolist()
    
    with open("metrics.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": conf_matrix}, f)
    
    logging.info("Evaluation metrics saved to metrics.json.")
    logging.info(f"Classification report: {report}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    args = parser.parse_args()
    
    setup_logging()
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    model = build_small_resnet()
    
    logging.info("Training started.")
    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test)
    )
    logging.info("Training completed.")
    
    # Save models
    model.save("my_model.keras")
    model.save("saved_model")
    logging.info("Models saved in .keras and SavedModel formats.")
    
    evaluate_and_save_metrics(model, X_test, y_test)
    
    # Quick inference check
    sample = X_test[:5]
    predictions = model.predict(sample).argmax(axis=1)
    logging.info(f"Sample predictions: {predictions}")
    print(f"Sample predictions: {predictions}")

if __name__ == "__main__":
    main()
