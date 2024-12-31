import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

# Load data from a CSV file with semicolon as the delimiter and comma as decimal separator
df = pd.read_csv('Results5.csv', sep=';', decimal=',')

# Print the column names for debugging
print("Column names:", df.columns)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Extract the necessary columns using iloc
# Second, third, fifth, and sixth columns
x = pd.to_numeric(df.iloc[:, 1].values, errors='coerce')  # Second column (Ratio of Black Pixels)
y = pd.to_numeric(df.iloc[:, 4].values, errors='coerce')  # Fifth column (AwaP)
z = pd.to_numeric(df.iloc[:, 2].values, errors='coerce')  # Third column (Entropy)
size = pd.to_numeric(df.iloc[:, 5].values, errors='coerce')  # Population size
labels = df.iloc[:, 0].values  # First column (Image names)

# Check for any NaN values and handle them
print("NaN values in columns:")
print(f"X: {np.sum(np.isnan(x))}, Y: {np.sum(np.isnan(y))}, Z: {np.sum(np.isnan(z))}")

# Drop NaN values if they exist
valid_indices = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
x = x[valid_indices]
y = y[valid_indices]
z = z[valid_indices]
labels = labels[valid_indices]  # Keep corresponding labels
size = size[valid_indices]  # Keep corresponding population sizes

# Function to print current view angles when interacting with the plot
def on_mouse_release(event):
    elev = ax.elev  # Current elevation
    azim = ax.azim  # Current azimuth
    print(f"Current view angle: Elevation = {elev}, Azimuth = {azim}")

# Check the length of the valid data
print(f"Valid points count: {len(x)}")

# Normalize the values to the range [0, 1]
def min_max_normalize(data):
    return (data - (np.min(data)*0.9)) / ((np.max(data)*1.1) - (np.min(data)*0.9))

# Add black border path effect
def create_text_effect():
    return [withStroke(linewidth=1, foreground='black')]

# Apply logarithmic scaling to AwaP values (y)
y_scaled = np.log1p(y)  # log1p is used to compute log(1 + y)

# Normalize the other columns
x_normalized = min_max_normalize(x)
y_normalized = 1 - min_max_normalize(y)  # Normalize the log-scaled values
z_normalized = 1 - min_max_normalize(z)

# Normalize population sizes for point sizes
population_size_normalized = min_max_normalize(size)

# Calculate the distance from the origin (0,0,0) using normalized values
distances = np.sqrt(x_normalized**2 + y_normalized**2 + z_normalized**2)

# Normalize distances to the range [0, 1)
norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Connect the event
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

# Create a color map for the gradient
cmap = plt.get_cmap('rainbow')
colors = cmap(norm_distances)  # Map normalized distances to colors

# Plot points and trace lines to the planes
for i in range(len(x_normalized)):
    # Scatter plot for points colored by the color map and sized by population size
    if labels[i] != " ":
        ax.scatter(x_normalized[i], y_normalized[i], z_normalized[i], color=colors[i], s=population_size_normalized[i] * 800 + 10, alpha=0.9)  # Increase size of points

        # Draw colored continuous lines to the x-y plane
        ax.plot([x_normalized[i], x_normalized[i]], [y_normalized[i], y_normalized[i]], [0, z_normalized[i]], color=colors[i], linestyle='-', linewidth=0.2)
        # Draw colored continuous lines to the y-z plane
        ax.plot([0, x_normalized[i]], [y_normalized[i], y_normalized[i]], [z_normalized[i], z_normalized[i]], color=colors[i], linestyle='-', linewidth=0.2)
        # Draw colored continuous lines to the x-z plane
        ax.plot([x_normalized[i], x_normalized[i]], [0, y_normalized[i]], [z_normalized[i], z_normalized[i]], color=colors[i], linestyle='-', linewidth=0.2)

        # Annotate point with its label above the point
        ax.text(x_normalized[i], y_normalized[i], z_normalized[i] + 0.05, labels[i], color=colors[i], fontsize=15,path_effects=create_text_effect())

# Set labels and title
#ax.set_xlabel('Density (x)')
#ax.set_ylabel('Permeability (y)')
#ax.set_zlabel('Entropy (z)')
ax.set_title('Morphospace 3D Graph Cidades')

# Set axis limits based on normalized data min and max
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Set view angle
ax.view_init(elev=33.351073049540666, azim=37.72008268548061)

# Create a color bar with custom size
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, pad=0.1, shrink=0.5, aspect=10)
cbar.set_label('Distance from Origin')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Set ticks for the color bar
cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])  # Corresponding labels for ticks

plt.show()

# Create separate 2D plots for each pair of axes

# 2D Plot for x vs y
plt.figure()
scatter = plt.scatter(x_normalized, y_normalized, c=colors, s=population_size_normalized * 100 + 10, alpha=0.7)
#plt.xlabel('Density (x)')
#plt.ylabel('Permeability (y)')
plt.title('2D Plot: Density vs Permeability')

# Annotate each point with its label (city name)
for i in range(len(x_normalized)):
    plt.text(x_normalized[i], y_normalized[i], labels[i], color=colors[i], fontsize=14,path_effects=create_text_effect())

plt.colorbar(scatter, label='Distance from Origin')
plt.show()

# 2D Plot for x vs z
plt.figure()
scatter = plt.scatter(x_normalized, z_normalized, c=colors, s=population_size_normalized * 100 + 10, alpha=0.7)
#plt.xlabel('Density (x)')
#plt.ylabel('Information (z)')
plt.title('2D Plot: Density vs Information')

# Annotate each point with its label (city name)
for i in range(len(x_normalized)):
    plt.text(x_normalized[i], z_normalized[i], labels[i], color=colors[i], fontsize=14,path_effects=create_text_effect())

plt.colorbar(scatter, label='Distance from Origin')
plt.show()

# 2D Plot for y vs z
plt.figure()
scatter = plt.scatter(y_normalized, z_normalized, c=colors, s=population_size_normalized * 100 + 10, alpha=0.7)
#plt.xlabel('Permeability (y)')
#plt.ylabel('Information (z)')
plt.title('2D Plot: Permeability vs Information')

# Annotate each point with its label (city name)
for i in range(len(y_normalized)):
    plt.text(y_normalized[i], z_normalized[i], labels[i], color=colors[i], fontsize=14,path_effects=create_text_effect())

plt.colorbar(scatter, label='Distance from Origin')
plt.show()
