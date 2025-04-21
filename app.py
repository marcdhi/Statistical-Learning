import flask
from flask import Flask, render_template, request, jsonify
import random
import plotly.graph_objects as go
import plotly.io as pio
import plotly
import json
import traceback

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

# --- Dummy ML Function (enhanced with more realistic values) ---
def ml_model(path):
    """Simulates ML model prediction based on path."""
    path_data = {
        "Path A": (2.71, 0.89),  # Standard material
        "Path B": (1.58, 1.22),  # Light material
        "Path C": (3.14, 0.72),  # Dense material
    }
    return path_data.get(path, (0.0, 0.0))

# --- Plot Generation Function (using Plotly) ---
def generate_plot_json(path_context=""):
    """Generates professional Plotly plot data as JSON."""
    try:
        print(f"Generating plot for path: {path_context}")
        num_points = 15  # Increased number of points for smoother curve
        x_vals = list(range(1, num_points + 1))
        
        # Get base density and add some realistic variation
        base_density, _ = ml_model(path_context)
        print(f"Base density: {base_density}")
        
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
        paths = ["Path A", "Path B", "Path C"]
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
            density, viscosity = ml_model(selected_path)
            print(f"Model results - Density: {density}, Viscosity: {viscosity}")
            density_str = f"🧪 Density:<br>{density:.2f} g/cm³"
            viscosity_str = f"💧 Viscosity:<br>{viscosity:.2f} Pa·s"
            graph_json = generate_plot_json(selected_path)

        return jsonify({
            'density': density_str,
            'viscosity': viscosity_str,
            'plot_json': graph_json
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)