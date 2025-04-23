import os
import pandas as pd
from animation import create_animation_for_web

# Ensure the output directory exists
output_dir = 'static/animations'
os.makedirs(output_dir, exist_ok=True)

# Load the CSV with material data
csv_file = 'symbolic_regression.csv'
df = pd.read_csv(csv_file)

# Animation parameters
f_n = 8  # Natural frequency in Hz
A0 = 0.236628  # Peak amplitude

# Generate animations for each material
for index, row in df.iterrows():
    material_name = row['X']
    print(f"Generating animation for {material_name}...")
    
    output_path = f'{output_dir}/professional_damping_{material_name.lower().replace(" ", "_")}.gif'
    create_animation_for_web(
        material_name=material_name,
        csv_file=csv_file,
        f_n=f_n,
        A0=A0,
        output_path=output_path
    )
    
    print(f"Animation saved to {output_path}")

print("\nAll animations generated successfully!")
print("Run the Flask app to view them in the web interface.") 