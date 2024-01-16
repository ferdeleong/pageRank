# PageRank Algorithm

## Background
The PageRank algorithm was published and named after Google founder Larry Page and colleagues in 1998. It is extensively used by Google to determine the order in which websites are displayed in search results.

## Importance of a Website
A website's importance is determined by the links to and from other websites, creating a web graph where nodes represent webpages and edges represent hyperlinks.

## Web as a Graph
- **Nodes:** Webpages
- **Edges:** Hyperlinks

## Normalize Weight



## Hyperlink Matrix (𝐻)
Normalized hyperlink squared matrix:
$$ H_{ij} = \frac{1}{|S_i|} $$ if a node $S_i$ has a link to $S_j$, otherwise 0.

## Page Relevance
A page is more important if it has more incoming links. Outgoing links are easier to fake.

## Markov Chains
Stochastic model describing states and transition probabilities.

### Dangling Nodes
Nodes without outgoing links.

### Stochasticity Adjustment
Adjustments made to ensure convergence in the Markov chain.

## Google Matrix (𝐺)
$$ G = \alpha S + (1 - \alpha)\frac{1}{n}ee^T $$

### Power Method
Iterative method for finding the dominant eigenvector.

#### Convergence Speed
Determined by the L1 norm; typically converges in 50-100 iterations.

#### α Parameter
Controls the proportion of time the random surfer follows hyperlinks.

### Eigenvalue & Unique Solution
The principal eigenvector with eigenvalue λ = 1 represents the stationary distribution of web surfing probabilities.

## Computational Efficiency
The power method is matrix-free, offering storage-friendly and computationally efficient calculations with O(n) complexity.

### Perron-Frobenius Theorem
Ensures a stochastic matrix has a unique dominant eigenvalue.

## Conclusion: Efficiency
The power method is storage-friendly with O(n) computational efficiency.
