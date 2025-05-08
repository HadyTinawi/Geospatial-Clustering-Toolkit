# K-Means Clustering Implementation and Analysis

This project implements a custom K-Means clustering algorithm from scratch and applies it to analyze geospatial patterns in NYC Schools and Public Art installations. It also includes an extension for balanced K-Means clustering that addresses the problem of uneven cluster sizes.

## Project Overview

This implementation demonstrates the application of clustering algorithms to real-world geospatial data and compares the performance between custom implementations and industry-standard libraries.

### Key Features

- Custom implementation of K-Means clustering algorithm
- Performance comparison with scikit-learn's implementation
- Application to NYC public schools and public art installation datasets
- Implementation of DBSCAN for comparison with K-Means
- Extension for balanced K-Means clustering (maintaining similar-sized clusters)
- Comprehensive analysis and visualization of clustering results

## Datasets

The project uses two spatial datasets from New York City:

1. **NYC Public Schools Dataset** - Contains locations of public schools across NYC's five boroughs.
   - Source: [NYC Open Data - School Locations](https://data.cityofnewyork.us/Education/2019-2020-School-Locations/wg9x-4ke6)

2. **NYC Public Art Dataset** - Contains locations of public art installations across NYC.
   - Source: [NYC Open Data - Directory of Public Art](https://data.cityofnewyork.us/Recreation/Directory-of-Temporary-Public-Art/zhrf-jnt6/about_data)

## Implementation Details

### Standard K-Means

The standard K-Means algorithm implements the following steps:
1. Initialize centroids randomly from the data points
2. Assign each point to its nearest centroid based on Euclidean distance
3. Update centroids by calculating the mean of all points assigned to each cluster
4. Repeat until convergence or maximum iterations reached

### Balanced K-Means Extension

The balanced K-Means extension modifies the standard algorithm to create more evenly sized clusters:
1. Uses a balance factor parameter (0.0 to 1.0) to control the trade-off between cluster quality and size balance
2. Implements a sophisticated assignment method that attempts to maintain clusters of similar sizes while preserving relative cluster quality
3. Can be adjusted from minimal balancing (slight improvement in size distribution) to strong balancing (significant size equalization)

## Results

The implementation successfully:

1. Achieves comparable performance to scikit-learn's implementation on synthetic data
2. Identifies meaningful geographic clusters in NYC school locations that largely correspond to the five boroughs
3. Discovers clusters of public art installations that reflect the cultural geography of NYC
4. Demonstrates how balanced K-Means can reduce size variance between clusters with a controlled trade-off in clustering quality

## Applications

The balanced clustering approach is particularly useful for:

- School district planning and resource allocation
- Emergency service coverage area optimization
- Sales territory design and workforce distribution
- Equal distribution of public resources across geographic regions

## Usage

```python
# Standard K-Means example
kmeans_model = KMeans(n_clusters=5, random_state=42)
kmeans_model.fit(data)
labels = kmeans_model.labels_

# Balanced K-Means example
balanced_kmeans = BalancedKMeans(n_clusters=5, balance_factor=0.5, random_state=42)
balanced_kmeans.fit(data)
balanced_labels = balanced_kmeans.labels_
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Future Work

Potential extensions to this project include:
- Implementation of other clustering algorithms (hierarchical clustering, spectral clustering)
- More sophisticated initialization methods for K-Means (K-Means++)
- Application to additional spatial datasets
- Multi-dimensional scaling for visualizing high-dimensional datasets
- Incorporating additional features beyond spatial coordinates for clustering
