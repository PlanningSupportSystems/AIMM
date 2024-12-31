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
