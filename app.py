import flask
from flask import Flask, render_template, request, jsonify, send_file
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


# --- Configuration ---
# Professional color scheme
PRIMARY_COLOR = "#1a365d"  # Deep blue
SECONDARY_COLOR = "#2c5282"  # Medium blue
ACCENT_COLOR = "#4299e1"  # Light blue
BACKGROUND_COLOR = "#ffffff"  # White
TEXT_COLOR = "#2d3748"  # Dark gray
GRID_COLOR = "#e2e8f0"  # Light gray

# --- Initialize Flask App ---
app = Flask(__name__)

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
    """Renders the main HTML page."""
    try:
        print("Rendering index page")
        initial_plot_json = generate_plot_json()
        paths = ["Pepsi", "Amla Oil", "Ghee"]
        return render_template('index.html',
                              paths=paths,
                              initial_plot_json=initial_plot_json)
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
            
            graph_json = generate_plot_json(selected_path)

        # Return the results to the frontend
        print(f"Returning: density={density_str}, viscosity={viscosity_str}")
        return jsonify({
            'density': density_str,
            'viscosity': viscosity_str,
            'plot_json': graph_json
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

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)