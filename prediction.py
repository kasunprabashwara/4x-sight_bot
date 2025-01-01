import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTMForexPredictor:
    def __init__(self, look_back=30):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler()
        self.data = None

    def load_model(self, model_path):
        if self.model is not None:
            print("Model already loaded.")
            return
        try:
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        except ValueError:
            print(f"No model found at {model_path}")
            # train the model
            self.train("data\Foreign_Exchange_Rates.csv", "EURO AREA - EURO/US$", 0.8, 100, 16)
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

    def load_and_preprocess_data(self, file_path, column_name, test_split=0.8):
        # Load and preprocess the data
        data_set = pd.read_csv(file_path, na_values='ND')
        data_set.interpolate(inplace=True)  # Fill missing values
        self.data = np.array(data_set[column_name]).reshape(-1, 1)
        
        # Scale the data
        self.data = self.scaler.fit_transform(self.data)
        
        # Split into training and test sets
        split = int(len(self.data) * test_split)
        train, test = self.data[:split], self.data[split:]
        return train, test

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:i + self.look_back])
            y.append(data[i + self.look_back])
        return np.array(X), np.array(y)

    def build_model(self):
        self.model = Sequential([
            LSTM(100, activation='relu', input_shape=(self.look_back, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, file_path, column_name, test_split=0.8, epochs=100, batch_size=16):
        train, test = self.load_and_preprocess_data(file_path, column_name, test_split)
        x_train, y_train = self.create_sequences(train)
        x_test, y_test = self.create_sequences(test)

        # Reshape inputs for LSTM
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        # Build and train the model
        self.build_model()
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
        
        # Evaluate the model
        y_pred = self.model.predict(x_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_test = self.scaler.inverse_transform(y_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

    # change the predict method to accept the data list and not the index 
    def predict_future(self, past_sequence, future_steps):
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be initialized before prediction.")
        
        if len(past_sequence) != self.look_back:
            raise ValueError(f"Past sequence must have length {self.look_back}, but got {len(past_sequence)}.")
        
        # Ensure the input is a numpy array
        past_sequence = self.scaler.fit_transform(np.array(past_sequence).reshape(self.look_back, 1))
        future_predictions = []
        
        # Generate predictions iteratively
        for _ in range(future_steps):
            prediction = self.model.predict(past_sequence.reshape(1, self.look_back, 1), verbose=0)
            future_predictions.append(prediction[0, 0])
            past_sequence = np.append(past_sequence[1:], prediction, axis=0)
        
        # Inverse transform the predictions
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions
    
if __name__ == "__main__":
    predictor = LSTMForexPredictor(look_back=30)
    predictor.load_model("models/forex_lstm.keras")
    # dummy 30 values to predict the next 5 values
    predictions = predictor.predict_future([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1], 5)
    print(predictions)


    
