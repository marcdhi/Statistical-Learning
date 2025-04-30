import flask
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import random
import plotly.graph_objects as go
import plotly.io as pio
import plotly
import json
import traceback
import numpy as np 
import joblib 
from rocket import RocketPredictor
from thresholds import thres
import pandas as pd
import os
from animation import create_animation_for_web
import matplotlib.pyplot as plt


# --- Configuration ---
# Professional color scheme
PRIMARY_COLOR = "#1a365d"  # Deep blue
SECONDARY_COLOR = "#2c5282"  # Medium blue
ACCENT_COLOR = "#4299e1"  # Light blue
BACKGROUND_COLOR = "#ffffff"  # White
TEXT_COLOR = "#2d3748"  # Dark gray
GRID_COLOR = "#e2e8f0"  # Light gray

# Define material colors for consistent visualization 
MATERIAL_COLORS = [
    '#4299e1',  # Blue
    '#48bb78',  # Green
    '#ed8936',  # Orange
    '#9f7aea',  # Purple
    '#f56565',  # Red
    '#38b2ac',  # Teal
    '#ecc94b',  # Yellow
    '#667eea',  # Indigo
    '#fc8181',  # Pink
    '#4fd1c5'   # Cyan
]

# --- Initialize Flask App ---
app = Flask(__name__)

# Create the directory for storing animation files
os.makedirs('static/animations', exist_ok=True)

def get_response_data(material_name, viscosity=None, density=None, experimental_only=False):
    """
    Generate response data for experimental and theoretical models.
    If viscosity and density are provided, use those for theoretical prediction.
    If experimental_only is True, only experimental response will be returned with zeros for theoretical.
    """
    try:
        # Load data from CSV
        csv_file = 'symbolic_regression.csv'
        
        try:
            df = pd.read_csv(csv_file)
            row_df = df[df['X'].str.lower() == material_name.lower()]
            
            # Check if we actually found a row and if it has the required data
            if not row_df.empty and 'Experimental' in row_df.columns and not pd.isna(row_df['Experimental'].iloc[0]):
                row = row_df.squeeze()
                noise = 0.005 + np.random.random() * 0.015
                zeta_exp = float(row['Experimental']) + noise
                
                # If viscosity and density are provided, use them to predict damping
                if viscosity is not None and density is not None:
                    zeta_actual = predict_damping(viscosity, density)
                else:
                    zeta_actual = float(row['Symbolic'])
            else:
                # If row not found or data is missing, raise KeyError to fall back to simulated data
                raise KeyError(f"Material '{material_name}' not found or has incomplete data in CSV")
                
        except (FileNotFoundError, KeyError) as e:
            print(f"CSV file not found or key error: {e}, using simulated data")
            # Simulated data if CSV doesn't exist or material not found
            # Generating random but consistent zeta values based on material name
            seed = sum(ord(c) for c in material_name)
            np.random.seed(seed)
            zeta_exp = 0.05 + np.random.random() * 0.15  # Between 0.05 and 0.2
            
            # If viscosity and density are provided, use them to predict damping
            if viscosity is not None and density is not None:
                zeta_actual = predict_damping(viscosity, density)
            else:
                zeta_actual = 0.03 + np.random.random() * 0.1  # Between 0.03 and 0.13
        
        # For custom materials in analyze_viscosity, set experimental zeta to the predicted one
        if material_name.lower() == "custom" and viscosity is not None and density is not None:
            print(f"Using predicted damping as experimental for custom material: {zeta_actual}")
            zeta_exp = zeta_actual
        
        # Constants
        f_n = 8  # Natural frequency in Hz
        A0 = 0.236628  # Peak amplitude
        
        # Calculate time and responses
        omega_n = 2 * np.pi * f_n
        t_end = 1.5  # Display 1.5 seconds of response
        t = np.linspace(0, t_end, 1000)
        
        def damped_response(zeta):
            try:
                if not np.isscalar(zeta) or zeta <= 0 or zeta >= 1:
                    print(f"Invalid damping ratio: {zeta}, using default value")
                    zeta = 0.1  # Default damping ratio for invalid values
                
                omega_d = omega_n * np.sqrt(1 - zeta**2)
                return A0 * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t)
            except Exception as e:
                print(f"Error in damped_response: {e}")
                return np.zeros_like(t)
        
        # Calculate experimental response
        x_exp = damped_response(zeta_exp)
        envelope_exp_upper = A0 * np.exp(-zeta_exp * omega_n * t)
        envelope_exp_lower = -A0 * np.exp(-zeta_exp * omega_n * t)
        delta_exp = 2 * np.pi * zeta_exp / np.sqrt(1 - zeta_exp**2)
        
        # If experimental_only is True, set theoretical values to zero
        if experimental_only:
            x_actual = np.zeros_like(t)
            envelope_actual_upper = np.zeros_like(t)
            envelope_actual_lower = np.zeros_like(t)
            delta_actual = 0.0
        else:
            # Calculate theoretical response
            x_actual = damped_response(zeta_actual)
            envelope_actual_upper = A0 * np.exp(-zeta_actual * omega_n * t)
            envelope_actual_lower = -A0 * np.exp(-zeta_actual * omega_n * t)
            delta_actual = 2 * np.pi * zeta_actual / np.sqrt(1 - zeta_actual**2)
        
        return {
            'success': True,
            'time': t.tolist(),
            'experimental_response': x_exp.tolist(),
            'theoretical_response': x_actual.tolist(),
            'exp_envelope_upper': envelope_exp_upper.tolist(),
            'exp_envelope_lower': envelope_exp_lower.tolist(),
            'theo_envelope_upper': envelope_actual_upper.tolist(),
            'theo_envelope_lower': envelope_actual_lower.tolist(),
            'experimental_zeta': float(zeta_exp),
            'theoretical_zeta': float(zeta_actual),
            'experimental_delta': float(delta_exp),
            'theoretical_delta': float(delta_actual)
        }
    except Exception as e:
        print(f"Error in get_response_data: {e}")
        print(traceback.format_exc())  # Add this to get the full traceback
        return {'success': False, 'error': str(e)}

def ml_process(path):
    try:
        print(f"ml_process called with path: {path}")
        X_list = []
        df = pd.read_csv(path)
        print(f"CSV file loaded. Column names: {df.columns.tolist()}")
        X_series = df['Amplitude - Acceleration (FFT - (Peak))'].to_numpy()
        print(f"Extracted series shape: {X_series.shape}")
        time_steps = 30  # Example fixed length
        if len(X_series) < time_steps:
            X_series = np.pad(X_series, (0, time_steps - len(X_series)), mode='constant')
        else:
            X_series = X_series[:time_steps]  # Truncate if too long
        print(f"Processed series shape: {X_series.shape}")
        X_list.append(X_series)  
        X = np.array(X_list)
        X = X.reshape(1,-1)
        print(f"Final X shape: {X.shape}")

        print("Loading viscosity model...")
        viscosity_model = joblib.load('models/rocket_predictor_modelv.joblib')
        print("Loading density model...")
        density_model = joblib.load('models/rocket_predictor_modeld.joblib')
        
        print("Predicting with models...")
        y_preds = np.maximum(thres, viscosity_model.predict(X))
        de = density_model.predict(X)
        print(f"Predictions - viscosity: {y_preds[0]}, density: {de[0]}")
        return de[0], y_preds[0]
    except Exception as e:
        print(f"Error in ml_process: {e}")
        print(traceback.format_exc())
        return 0.0, 0.0

# --- Dummy ML Function (enhanced with more realistic values) ---
def ml_model(path):
    """Simulates ML model prediction based on path."""
    try:
        print(f"ml_model called with path: {path}")
        # Create a mapping from material names to file paths
        material_to_path = {
            "Amla Oil": os.path.join("tests", "Amla_Oil.csv"),
            "Ghee": os.path.join("tests", "Ghee.csv"),
            "Pepsi": os.path.join("tests", "Pepsi.csv"),
            "2T": os.path.join("tests", "2T.csv"),
            "15W40": os.path.join("tests", "15W40.csv"),
            "20W40": os.path.join("tests", "20W40.csv"),
            "air": os.path.join("tests", "air.csv"),
            "Badam_Oil": os.path.join("tests", "Badam_Oil.csv"),
            "Castor_Oil": os.path.join("tests", "Castor_Oil.csv"),
            "Ghee": os.path.join("tests", "Ghee.csv"),
            "Milk": os.path.join("tests", "Milk.csv"),
            "Olive_Oil": os.path.join("tests", "Olive_Oil.csv"),
            "SAE30": os.path.join("tests", "SAE30.csv"),
            "Sprite": os.path.join("tests", "Sprite.csv"),
            "Vinegar": os.path.join("tests", "Vinegar.csv"),
            "Water": os.path.join("tests", "Water.csv")
        }
        
        # Default fallback test file
        default_test_file = os.path.join("tests", "Amla_Oil.csv")
        
        # Check if the path exists in our mapping
        if path in material_to_path:
            file_path = material_to_path[path]
        else:
            # For other materials, use a default test file
            print(f"No specific test file for {path}, using default")
            file_path = default_test_file
            
        print(f"Using file path: {file_path}")
        if os.path.exists(file_path):
            result = ml_process(file_path)
            print(f"ml_process result: {result}")
            
            # Apply some variation based on material name for variety
            # This is just for demonstration - real system would use actual data
            if path not in material_to_path:
                density, viscosity = result
                # Create a hash from the material name for consistent randomization
                import hashlib
                hash_value = int(hashlib.md5(path.encode()).hexdigest(), 16) % 100
                density_factor = 0.8 + (hash_value % 40) / 100  # Between 0.8 and 1.2
                viscosity_factor = 0.8 + ((hash_value + 17) % 40) / 100  # Between 0.8 and 1.2
                
                modified_result = (density * density_factor, viscosity * viscosity_factor)
                print(f"Modified result for {path}: {modified_result}")
                return modified_result
            
            return result
        else:
            print(f"File not found: {file_path}")
            return (0.0, 0.0)
    except Exception as e:
        print(f"Error in ml_model: {e}")
        print(traceback.format_exc())
        return (0.0, 0.0)

def predict_damping(viscosity, density):
    """
    Predicts damping based on viscosity and density.
    """
    try:
        model = joblib.load('models/damping.joblib')
        new_data = pd.DataFrame([[viscosity, density]], columns=['Viscosity', 'Density'])
        prediction = model.predict(new_data)
        # noise = 0.005 + np.random.random() * 0.015
        # Return the scalar value directly, not the first element of an array
        return float(prediction[0])
    except Exception as e:
        print(f"Error in predict_damping: {e}")
        print("---DANGER---Fallback to simulated prediction---DANGER---")
        # Fallback to simulated prediction if model fails
        dampingVal = 0.05 + (viscosity / 1000) * 0.1 + (density / 1000) * 0.05
        return dampingVal
    
# --- Plot Generation Function (using Plotly) ---
def generate_plot_json(path_context=""):
    """Generates professional Plotly plot data as JSON."""
    try:
        print(f"Generating plot for path: {path_context}")
        num_points = 15  # Increased number of points for smoother curve
        x_vals = list(range(1, num_points + 1))
        
        # Get base density and viscosity from the model
        base_density, base_viscosity = ml_model(path_context)
        print(f"Base density: {base_density}, Base viscosity: {base_viscosity}")
        
        # Generate more realistic looking data with some noise and trend
        y_vals = []
        trend = random.choice([-0.1, 0.1])  # Random trend direction
        for i in range(num_points):
            noise = random.uniform(-0.2, 0.2)
            value = base_density + (i * trend * 0.1) + noise
            y_vals.append(max(0, value))  # Ensure no negative values

        fig = go.Figure()

        # Add main trace with gradient fill
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name='Material Property',
            line=dict(
                color=PRIMARY_COLOR,
                width=3,
                shape='spline'  # Smooth curve
            ),
            marker=dict(
                color=SECONDARY_COLOR,
                size=8,
                line=dict(
                    color=BACKGROUND_COLOR,
                    width=2
                )
            ),
            fill='tozeroy',
            fillcolor=f'rgba(26, 54, 93, 0.1)'  # Transparent primary color
        ))

        # Update layout for professional look
        fig.update_layout(
            title=dict(
                text="Material Property Analysis",
                font=dict(size=24, color=TEXT_COLOR, family="Inter"),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title=dict(
                    text="Sample Index",
                    font=dict(size=14, color=TEXT_COLOR)
                ),
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
                showline=True,
                linecolor=TEXT_COLOR,
                linewidth=2,
                ticks="outside"
            ),
            yaxis=dict(
                title=dict(
                    text="Property Value",
                    font=dict(size=14, color=TEXT_COLOR)
                ),
                showgrid=True,
                gridcolor=GRID_COLOR,
                zeroline=False,
                showline=True,
                linecolor=TEXT_COLOR,
                linewidth=2,
                ticks="outside"
            ),
            plot_bgcolor=BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(
                family="Inter, sans-serif",
                size=12,
                color=TEXT_COLOR
            ),
            showlegend=False,
            margin=dict(l=60, r=30, t=80, b=60),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor=BACKGROUND_COLOR,
                font_size=12,
                font_family="Inter"
            )
        )

        # Add confidence interval band (for visual effect)
        upper_bound = [y + random.uniform(0.2, 0.4) for y in y_vals]
        lower_bound = [max(0, y - random.uniform(0.2, 0.4)) for y in y_vals]  # Ensure non-negative
        
        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor=f'rgba(66, 153, 225, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Convert figure to JSON
        print("Converting figure to JSON")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        print(f"JSON data length: {len(graph_json)}")
        return graph_json
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        print(traceback.format_exc())
        # Return a simple fallback plot on error
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        fig.update_layout(
            title="Error in Plot Generation", 
            annotations=[dict(
                text=f"Error: {str(e)}",
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page."""
    try:
        test_materials = ["Amla_Oil", "Pepsi", "Ghee"]
        
        # Get all materials in the tests folder
        train_materials = []
        test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
        for file in os.listdir(test_folder):
            if file.endswith('.csv'):
                material_name = os.path.splitext(file)[0]
                if material_name not in test_materials:
                    train_materials.append(material_name)
        
        # Sort for consistent display
        train_materials.sort()

        # remove air from train_materials
        train_materials = [material for material in train_materials if material != "air"]
        
        # Updated color scheme
        colors = MATERIAL_COLORS
        
        return render_template('index.html', 
                              test_materials=test_materials,
                              train_materials=train_materials,
                              colors=colors)
    except Exception as e:
        print(f"Error in index route: {e}")
        print(traceback.format_exc())
        return f"An error occurred: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    try:
        print("Handling predict request")
        data = request.get_json()
        selected_path = data.get('path', '')
        print(f"Selected path: {selected_path}")

        if not selected_path or selected_path == "Choose a path...":
            density_str = "Density: --"
            viscosity_str = "Viscosity: --"
            graph_json = generate_plot_json()
            return jsonify({
                'density': density_str,
                'viscosity': viscosity_str,
                'plot_json': graph_json,
                'raw_density': None,
                'raw_viscosity': None
            })
        else:
            # Get the density and viscosity values
            density, viscosity = ml_model(selected_path)
            print(f"Model results - Density: {density}, Viscosity: {viscosity}")
            
            # Ensure they are numeric and format them
            try:
                density = float(density)
                viscosity = float(viscosity)
                viscosity_str = f"💧 Viscosity:<br>{viscosity:.2f} mPa·s"
                density_str = f"🧪 Density:<br>{density:.2f} kg/m³"
            except (ValueError, TypeError) as e:
                print(f"Error converting values to float: {e}")
                viscosity_str = f"💧 Viscosity:<br>Error"
                density_str = f"🧪 Density:<br>Error"
                density = None
                viscosity = None
            
            graph_json = generate_plot_json(selected_path)

        # Return the results to the frontend, including raw values
        print(f"Returning: density={density_str}, viscosity={viscosity_str}")
        return jsonify({
            'density': density_str,
            'viscosity': viscosity_str,
            'plot_json': graph_json,
            'raw_density': density,
            'raw_viscosity': viscosity
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/generate_animation', methods=['POST'])
def create_animation():
    """Return the path to a pre-generated animation GIF for the selected material."""
    try:
        data = request.get_json()
        material_name = data.get('material', '')
        
        if not material_name or material_name == "Choose a path...":
            return jsonify({'error': 'Please select a valid material'}), 400
        
        # Construct the path to the pre-generated animation file
        gif_path = f'static/animations/professional_damping_{material_name.lower().replace(" ", "_")}.gif'
        
        # Check if the file exists
        if os.path.exists(gif_path):
            return jsonify({
                'success': True,
                'animation_path': gif_path
            })
        else:
            return jsonify({'error': f'Animation not found for {material_name}'}), 404
            
    except Exception as e:
        print(f"Error in animation retrieval: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

@app.route('/get_response_data', methods=['POST'])
def response_data_endpoint():
    """Endpoint to get response data for comparison plots."""
    try:
        data = request.json
        material_name = data.get('material')
        
        if not material_name:
            return jsonify({'success': False, 'error': 'No material specified'})
        
        # Get viscosity and density if provided
        viscosity = data.get('viscosity')
        density = data.get('density')
        experimental_only = data.get('experimental_only', False)
        
        # Call the get_response_data function with the material name
        response_data = get_response_data(material_name, viscosity, density, experimental_only)
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in response_data_endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_viscosity', methods=['POST'])
def analyze_viscosity():
    """Endpoint to analyze viscosity and density from tab3."""
    try:
        data = request.json
        viscosity = data.get('viscosity')
        density = data.get('density')
        
        if viscosity is None or density is None:
            return jsonify({'success': False, 'error': 'Viscosity and density are required'})
        
        # Convert to float
        try:
            viscosity = float(viscosity)
            density = float(density)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid viscosity or density values'})
        
        # Predict damping
        print(f"Viscosity from analyze_viscosity: {viscosity}")
        print(f"Density from analyze_viscosity: {density}")
        damping_ratio = predict_damping(viscosity, density)

        print(f"Damping ratio from analyze_viscosity: {damping_ratio}")
        
        # Get response data with experimental only
        response_data = get_response_data("Custom", viscosity, density, experimental_only=True)
        
        # Include damping ratio in response
        response_data['damping_ratio'] = float(damping_ratio)
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in analyze_viscosity: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)