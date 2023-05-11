# Comparing Traditional Recommender Systems with Graph-Based Approaches

In recent years, recommender systems have gained significant popularity, and there has been an increase in the utilization of graphs to represent data. This project aims to compare traditional recommender systems with graph-based approaches to determine if the latter can provide recommendations with similar accuracy to the former.

## Methods Reviewed

Several methods for recommender systems are reviewed, including:

- Collaborative filtering
- Matrix factorization
- Neural networks
- Personalized PageRank
- Graph embeddings

## Dataset

The project utilizes a Goodreads 10k dataset for evaluation purposes.

## Results

The obtained results indicate that graph-based approaches perform similarly to traditional methods in terms of RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) scores. However, graph-based approaches outperform traditional methods in terms of precision@k and recall@k metrics. This demonstrates that graph-based approaches are a viable alternative for recommender systems, especially in situations where graphs are a natural way to represent data and where traditional methods become memory inefficient due to increase in dataset sizes.
