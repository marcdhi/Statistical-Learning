import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import joblib

class RocketPredictor:
    def __init__(self, num_kernels=10000, random_state=42):
        """
        Initialize ROCKET predictor for time series amplitude data
        
        Parameters:
        - num_kernels: Number of random convolutional kernels
        - random_state: Seed for reproducibility
        """
        self.num_kernels = num_kernels
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.kernels = None
        self.biases = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.ridge_regressor = Ridge(alpha=1.0)
        self.thres = 3.23
    
    def _generate_kernels(self, input_length):
        """
        Generate random convolutional kernels
        
        Parameters:
        - input_length: Length of input time series
        """
        # Kernel lengths
        kernel_lengths = [3, 5, 7, 9, 11]
        
        # Initialize kernels and biases
        self.kernels = []
        self.biases = np.zeros(self.num_kernels)
        
        for _ in range(self.num_kernels):
            # Randomly choose kernel length
            kernel_length = np.random.choice(kernel_lengths)
            
            # Generate kernel weights
            kernel = np.random.normal(0, 1, kernel_length)
            
            # Randomly choose dilation
            dilation = np.random.choice([1, 2, 4])
            
            # Store kernel parameters
            self.kernels.append({
                'weights': kernel,
                'length': kernel_length,
                'dilation': dilation
            })
            
            # Generate random bias
            self.biases[_] = np.random.uniform(-1, 1)
    
    def _rocket_transform(self, X):
        """
        Apply ROCKET transform to input time series
        
        Parameters:
        - X: Input time series data (shape: num_samples x time_steps)
        """
        # Initialize output matrix
        X_transformed = np.zeros((X.shape[0], self.num_kernels))
        
        # Apply convolution for each kernel
        for i, kernel_params in enumerate(self.kernels):
            convolved = np.zeros(X.shape[0])
            
            for j in range(X.shape[0]):
                # Prepare convolution with dilation
                dilated_kernel = np.repeat(kernel_params['weights'], kernel_params['dilation'])
                
                # Perform convolution
                conv_result = np.convolve(X[j], dilated_kernel, mode='valid')
                
                # Apply max pooling with bias
                convolved[j] = np.max(conv_result + self.biases[i])
            
            # Store transformed feature
            X_transformed[:, i] = convolved
        
        return X_transformed
    
    def fit(self, X, y):
        """
        Fit ROCKET predictor
        
        Parameters:
        - X: Input time series features (shape: num_samples x time_steps)
        - y: Target values
        """
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale input features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Generate random kernels
        self._generate_kernels(X.shape[1])
        
        # Transform data using ROCKET
        X_transformed = self._rocket_transform(X_scaled)
        
        # Fit Ridge regression
        self.ridge_regressor.fit(X_transformed, y_scaled)
        
        return self
    
    def predict(self, X):
        """
        Predict using fitted ROCKET model
        
        Parameters:
        - X: Input time series features
        """
        # Ensure X is numpy array
        X = np.array(X)
        
        # Scale input features
        X_scaled = self.scaler_X.transform(X)
        
        # Transform data using ROCKET
        X_transformed = self._rocket_transform(X_scaled)
        
        # Predict and inverse transform
        y_pred_scaled = self.ridge_regressor.predict(X_transformed)
        return np.maximum(3.23, self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel())
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        - X_test: Test features
        - y_test: True target values
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mean_squared_error': mse,
            'r2_score': r2,
            'predictions': y_pred
        }
    
    def plot_results(self, y_test, y_pred):
        """
        Visualize prediction results
        
        Parameters:
        - X_test: Test input features
        - y_test: True target values
        - y_pred: Predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_pred.min(), y_pred.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('ROCKET Prediction: True vs Predicted')
        plt.tight_layout()
        plt.show()

# Example usage function
def example_rocket_usage(X, y):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train ROCKET predictor
    rocket_predictor = RocketPredictor(num_kernels=5000)
    rocket_predictor.fit(X, y)
    joblib.dump(rocket_predictor, 'rocket_predictor_modelv.joblib')
    # with open('rocket_predictor_modeld.pkl', 'wb') as f:
    #     pickle.dump(rocket_predictor, f)

    #     print("Model saved as 'rocket_predictor_model.pkl'")
    # preds = loaded_model.predict(X_test[0].reshape(1,-1))
    # print(preds)
    # Evaluate model
    noise_X = np.random.normal(0, 0.8, X_test.shape)   
    noise_Y = np.random.normal(0, 0.8, y_test.shape)

    evaluation_metrics = rocket_predictor.evaluate(X_test-0.0003, (y_test+noise_Y))
    
    print("ROCKET Model Evaluation:")
    print(f"Mean Squared Error: {evaluation_metrics['mean_squared_error']}")
    print(f"R-squared Score: {evaluation_metrics['r2_score']}")
    
    # Plot results
    print(f"X_test shape: {X_test.shape}, y_pred shape: {y_test.shape}")

    rocket_predictor.plot_results(rocket_predictor.predict(X_test-0.0003), (y_test+noise_Y)) #, evaluation_metrics['predictions'])
    
    return rocket_predictor
