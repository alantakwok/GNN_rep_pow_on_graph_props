# Understanding representation power of graph neural network for learning graph properties

Neural networks show great potential in various applications such as computer vision and natu-
ral language processing. At the same time, it becomes uncompetitive when trained on small-scale
datasets such as tabular datasets that is still dominated by Gradient-Boosted Decision Trees (GBDT).
There are few studies on the representation power of graph neural network in learning graph topol-
ogy. This project investigates and compares the effectiveness of different types of graph neural net-
works at learning and predicting different graph properties.  

# Subsection  

For this project, we intend to extend on the paper’s research by looking into how different architectures of GCNs
perform at capturing various graph properties. Our goal is to demonstrate that the neural networks are able to compute
graph properties to reasonable accuracy.
The dataset under investigation consists of synthetic graphs generated from the Erdos-Renyi (ER) and Barab ́asi–Albert
(BA) models. Analysis of graph properties are observed under traditional GNNs, GCNs without self-attention and
GCNs with self-attention. The list of graph properties examined include clustering coefficients, path lengths between
node pairs, page ranks, and more.
Through this work we compare validation losses and their convergence for properties between different models and
discover which models are best suited for computing specific properties.
