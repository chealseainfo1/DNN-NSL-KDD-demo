import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ========== Custom Optimizer ==========
class AdamDoubleInertia(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, alpha=0.1, epsilon=1e-7,
                 name="AdamDoubleInertia", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "u")

    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper("learning_rate")
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        u = self.get_slot(var, "u")

        m_t = self.beta_1 * m + (1 - self.beta_1) * grad
        v_t = self.beta_2 * v + (1 - self.beta_2) * tf.square(grad)
        u_t = self.alpha * u + (1 - self.alpha) * m_t

        m.assign(m_t)
        v.assign(v_t)
        u.assign(u_t)

        return var.assign_sub(lr * u_t / (tf.sqrt(v_t) + self.epsilon))

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "alpha": self.alpha,
            "epsilon": self.epsilon
        }

# ========== Data Loading and Preprocessing ==========
def load_nsl_kdd_data():
    if not os.path.exists('NSL-KDD_train.csv'):
        raise FileNotFoundError("Missing file! Please place 'NSL-KDD_train.csv' in the script directory.")

    data = pd.read_csv('NSL-KDD_train.csv')

    # Assume label column is named 'label' and there's a mix of categorical and numerical features
    categorical_features = ['protocol_type', 'service', 'flag']  # Example categorical columns
    numerical_features = data.columns.difference(categorical_features + ['label'])

    # Preprocess label
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Preprocess features
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

    X = preprocessor.fit_transform(data)
    y = data['label']

    y_cat = tf.keras.utils.to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42, stratify=y)

    return x_train, x_test, y_train, y_test

# ========== Model Definition ==========
def build_nsl_kdd_model(input_dim, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# ========== Training and Evaluation ==========
def train_and_evaluate(x_train, x_test, y_train, y_test, optimizer, label):
    model = build_nsl_kdd_model(x_train.shape[1], y_train.shape[1])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"\nTraining with {label}...")
    start = time.time()
    model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=0)
    elapsed = time.time() - start

    preds = model.predict(x_test, verbose=0)
    preds_classes = preds.argmax(1)
    true_classes = y_test.argmax(1)

    acc = accuracy_score(true_classes, preds_classes)

    print(f"{label} - Accuracy: {acc:.4f}, Time: {elapsed:.2f} sec")
    print(classification_report(true_classes, preds_classes, digits=4))

    return acc, elapsed

# ========== Main ==========
def main():
    x_train, x_test, y_train, y_test = load_nsl_kdd_data()

    # Standard Adam
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_and_evaluate(x_train, x_test, y_train, y_test, adam, "Standard Adam")

    # Custom Optimizer
    custom_opt = AdamDoubleInertia(learning_rate=0.001)
    train_and_evaluate(x_train, x_test, y_train, y_test, custom_opt, "Adam Double Inertia")

if __name__ == "__main__":
    main()