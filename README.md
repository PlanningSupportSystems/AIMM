# AIMM - Analysis of Information in Morphology and Morphospace

Permeability and Spatial Analysis Tool
This project is a high-performance image analysis tool designed to calculate entropy, permeability, density, and geometric properties of binary images. The primary goal is to evaluate spatial patterns, connectivity, and density distribution within binary images, particularly focusing on the relationships between white and black pixel regions. The tool supports parallel computation for efficient processing of large images.

Key Functionalities:
Entropy Calculation:

Measures the randomness or complexity of block patterns within binary images.
Considers rotation and mirroring to identify canonical patterns and reduce redundancy in pattern matching.
Permeability Analysis:

Calculates the permeability of each white pixel by determining the "reach" of a line in an open space without any obstruction from black pixels.
Utilizes Bresenham's line algorithm for accurate line-of-sight computations, ensuring all possible paths are evaluated.
Longest Line Calculation:

Identifies the longest straight line of contiguous white pixels that are visible from each white pixel.
Supports fine-grained angular resolutions (e.g., 8, 16, or 64 directions) for precise line length detection.
Density Analysis:

Detects the outermost polygon formed by black pixels

Computes the density of black pixels within this polygon.


Heatmap Generation:

Produces heatmaps to visualize:
Permeability scores for each white pixel.
Longest line sizes for each white pixel.
Uses intuitive rainbow color maps for visualization, preserving black pixels in the outputs.

Multi-threaded Processing:

Parallelizes computations for entropy, permeability, density, and longest line analysis across multiple threads.
Provides real-time progress feedback during processing.
CSV Export:

Outputs detailed results to CSV files, including:
Entropy, permeability, density, and longest line metrics for each image.
Average permeability and longest line size for each image.
Overall density metrics.
Customizable Parameters:

Allows user-defined block sizes for entropy and density calculations.
Configurable thread count for optimized performance on large datasets.

Morphospace Visualization Tool
This is a Python-based data visualization tool designed to analyze and display multidimensional datasets in 3D and 2D morphospace representations. It leverages Matplotlib for visualization and Pandas for data manipulation, enabling intuitive and detailed exploration of complex data.

Key Functionalities:
3D Morphospace Visualization:

Displays data points in a 3D scatter plot, representing three key dimensions (e.g., Density, Permeability, Entropy).
Points are color-coded based on their normalized distance from the origin, providing insights into spatial relationships.
Point sizes reflect an additional attribute (e.g., Population Size), enabling quick identification of significant data points.
Names are displayed above the points, with a visually distinct black outline for clarity.
Dynamic Interaction:

Supports interactive rotation of the 3D plot, with functionality to retrieve the current view angles for consistent reproduction of specific perspectives.
2D Pairwise Projections:

Generates separate 2D scatter plots for each pair of dimensions (e.g., x vs. y, x vs. z, y vs. z).
Includes annotations with point names and visual attributes (color and size) consistent with the 3D plot.
Normalization and Filtering:

All dimensions are normalized to a consistent scale, considering extreme values while ensuring no data point is strictly 0 or 1.
Points with empty labels are excluded from plots but retained in calculations to ensure accurate normalization.
Customizability:

Allows scaling adjustments (e.g., logarithmic scaling for specific axes).
Supports dynamic color mapping and size adjustments based on user-defined criteria.
Data Integration:

Reads CSV files with customizable delimiters and decimal separators.
Handles missing or malformed data gracefully, ensuring robust operation.
