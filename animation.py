import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# === Define color scheme to match the provided Plotly style ===
PRIMARY_COLOR = '#1a365d'       # Dark blue
SECONDARY_COLOR = '#4299e1'     # Light blue
BACKGROUND_COLOR = '#ffffff'    # White
TEXT_COLOR = '#2d3748'          # Dark gray
GRID_COLOR = '#e2e8f0'          # Light gray

# === Set up global matplotlib styling ===
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']  # Removed Inter as it might not be available
mpl.rcParams['axes.edgecolor'] = TEXT_COLOR
mpl.rcParams['axes.labelcolor'] = TEXT_COLOR
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = GRID_COLOR
mpl.rcParams['text.color'] = TEXT_COLOR
mpl.rcParams['xtick.color'] = TEXT_COLOR
mpl.rcParams['ytick.color'] = TEXT_COLOR

# === INPUT ===
csv_file = 'symbolic_regression.csv'   # CSV filename (save your data as this)
material_name = 'Amla Oil'         # Change to any material like 'Ghee', 'HP90', etc.
f_n = 8                         # Natural frequency in Hz
A0 = 0.236628                   # Peak amplitude

# === Load CSV and extract zeta values ===
df = pd.read_csv(csv_file)
row = df[df['X'].str.lower() == material_name.lower()].squeeze()
zeta_exp = row['Experimental']
zeta_actual = row['Symbolic']

# === Time and responses ===
omega_n = 2 * np.pi * f_n
t_end = 1.5  # Display 1.5 seconds of response
t = np.linspace(0, t_end, 1000)

def damped_response(zeta):
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    return A0 * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t)

x_actual = damped_response(zeta_actual)
x_exp = damped_response(zeta_exp)

# Calculate log decrements
delta_actual = 2 * np.pi * zeta_actual / np.sqrt(1 - zeta_actual**2)
delta_exp = 2 * np.pi * zeta_exp / np.sqrt(1 - zeta_exp**2)

# === Set up figure with professional styling ===
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BACKGROUND_COLOR)
ax.set_facecolor(BACKGROUND_COLOR)

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(TEXT_COLOR)
ax.spines['left'].set_color(TEXT_COLOR)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Set up labels and title with professional styling
ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold', color=TEXT_COLOR)
ax.set_ylabel('Amplitude (g)', fontsize=14, fontweight='bold', color=TEXT_COLOR)
ax.set_title(f'Damped Response Analysis - {material_name}', 
             fontsize=24, fontweight='bold', color=TEXT_COLOR, pad=20)

ax.set_ylim(-1.1 * A0, 1.1 * A0)
ax.set_xlim(0, t_end)
ax.grid(True, alpha=0.5, linestyle='--')

# Create empty lines with professional styling
line_exp, = ax.plot([], [], color='#4299e1', linewidth=2.5, alpha=0.7, 
                   label=f'Experimental (δ = {delta_exp:.3f})')

line_actual, = ax.plot([], [], color='#1a365d', linewidth=3, 
                      label=f'Symbolic (δ = {delta_actual:.3f})')

# Calculate confidence intervals/envelopes
envelope_exp_upper = A0 * np.exp(-zeta_exp * omega_n * t)
envelope_exp_lower = -A0 * np.exp(-zeta_exp * omega_n * t)
envelope_actual_upper = A0 * np.exp(-zeta_actual * omega_n * t)
envelope_actual_lower = -A0 * np.exp(-zeta_actual * omega_n * t)

# Style the legend
ax.legend(loc='upper right', frameon=True, facecolor=BACKGROUND_COLOR, 
          edgecolor=GRID_COLOR, fontsize=12)

# Add text box with styling
textbox = ax.text(
    0.05, 0.95, 
    f"",  # Will be populated in animation
    transform=ax.transAxes, 
    fontsize=12,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor=BACKGROUND_COLOR, 
        alpha=0.8,
        edgecolor=GRID_COLOR
    )
)

# Add data info text box
info_text = ax.text(
    0.95, 0.05,
    f"Natural frequency: {f_n} Hz\nAmplitude: {A0:.4f} g",
    transform=ax.transAxes,
    fontsize=10,
    ha='right',
    va='bottom',
    bbox=dict(
        boxstyle='round,pad=0.3',
        facecolor=BACKGROUND_COLOR,
        alpha=0.8,
        edgecolor=GRID_COLOR
    )
)

# Total number of frames
total_frames = 500
half_frames = total_frames // 2
frames_per_phase = [
    int(half_frames * 0.7),  # Animation buildup for exp
    int(half_frames * 0.3),  # Hold pause for exp
    int(half_frames * 0.7),  # Animation buildup for symbolic
    int(half_frames * 0.3)   # Hold pause for symbolic
]
cumulative_frames = np.cumsum(frames_per_phase)

# Storage for fill_between objects to remove them later
exp_fill = None
actual_fill = None

# Animation function
def animate(i):
    global exp_fill, actual_fill
    
    # Remove previous fill_between patches if they exist
    if exp_fill is not None:
        exp_fill.remove()
    if actual_fill is not None:
        actual_fill.remove()
    
    if i < cumulative_frames[0]:  # Experimental buildup phase
        progress = i / frames_per_phase[0]
        idx = min(int(progress * len(t)) + 1, len(t))
        
        # Update experimental curve
        line_exp.set_data(t[:idx], x_exp[:idx])
        line_actual.set_data([], [])
        
        # Update experimental envelope
        exp_fill = ax.fill_between(
            t[:idx], 
            envelope_exp_upper[:idx], 
            envelope_exp_lower[:idx], 
            alpha=0.1, 
            color='#4299e1'
        )
        
        textbox.set_text(f"Experimental Response\nLog Decrement (δ): {delta_exp:.3f}")
        
    elif i < cumulative_frames[1]:  # Experimental hold phase
        # Full experimental curve
        line_exp.set_data(t, x_exp)
        line_actual.set_data([], [])
        
        # Update experimental envelope
        exp_fill = ax.fill_between(
            t, 
            envelope_exp_upper, 
            envelope_exp_lower, 
            alpha=0.1, 
            color='#4299e1'
        )
        
        textbox.set_text(f"Experimental Response\nLog Decrement (δ): {delta_exp:.3f}")
        
    elif i < cumulative_frames[2]:  # Symbolic buildup phase
        progress = (i - cumulative_frames[1]) / frames_per_phase[2]
        idx = min(int(progress * len(t)) + 1, len(t))
        
        # Keep experimental curve fully visible but faded
        line_exp.set_data(t, x_exp)
        line_exp.set_alpha(0.5)
        
        # Build up symbolic curve
        line_actual.set_data(t[:idx], x_actual[:idx])
        
        # Update both envelopes
        exp_fill = ax.fill_between(
            t, 
            envelope_exp_upper, 
            envelope_exp_lower, 
            alpha=0.05,  # More transparent now
            color='#4299e1'
        )
        actual_fill = ax.fill_between(
            t[:idx], 
            envelope_actual_upper[:idx], 
            envelope_actual_lower[:idx], 
            alpha=0.1, 
            color='#1a365d'
        )
        
        textbox.set_text(f"Experimental Response\nLog Decrement (δ): {delta_exp:.3f}\n\nSymbolic Response\nLog Decrement (δ): {delta_actual:.3f}")
        
    else:  # Final hold phase with both curves
        # Full curves
        line_exp.set_data(t, x_exp)
        line_exp.set_alpha(0.5)
        line_actual.set_data(t, x_actual)
        
        # Update both envelopes
        exp_fill = ax.fill_between(
            t, 
            envelope_exp_upper, 
            envelope_exp_lower, 
            alpha=0.05, 
            color='#4299e1'
        )
        actual_fill = ax.fill_between(
            t, 
            envelope_actual_upper, 
            envelope_actual_lower, 
            alpha=0.1, 
            color='#1a365d'
        )
        
        textbox.set_text(f"Experimental Response\nLog Decrement (δ): {delta_exp:.3f}\n\nSymbolic Response\nLog Decrement (δ): {delta_actual:.3f}")
    
    return [line_exp, line_actual, textbox]

# Create animation - with slower animation
ani = FuncAnimation(
    fig, 
    animate, 
    frames=total_frames,
    interval=50,  # 50ms between frames
    blit=False    # Set to False to ensure proper envelope animation
)

# Add a watermark
# fig.text(0.95, 0.95, 'Material Analysis', fontsize=8, color=TEXT_COLOR, alpha=0.5,
#          ha='right', va='top', transform=ax.transAxes)

# Save as high-quality GIF
from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=20)
ani.save(f'professional_damping_{material_name.lower().replace(" ", "_")}.gif', 
         writer=writer, dpi=120)

plt.tight_layout()
plt.show()
print(f"Professional animation saved as professional_damping_{material_name.lower().replace(' ', '_')}.gif")