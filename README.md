# PageRank Algorithm

## Background
The PageRank algorithm was published and named after Google founder Larry Page and colleagues in 1998. It is extensively used by Google to determine the order in which websites are displayed in search results.

## Web as a Graph

- **Nodes:** Webpages
- **Edges:** Hyperlinks

![image](https://github.com/ferdeleong/pagerank/assets/78885738/3ad34f11-d610-4d9f-ae77-20e7cf1edc8d)

## Normalize Weight

| Node | Outgoing Links |
| :--- | :----:         |
| A    | 3              |
| B    | 2              |
| C    | 1              |
| D    | 2              |

|      | A     | B     | C     | D    |
| :--- | :----:| :----:| :----:|:----:|
| H_A  | 0     | 1/3   | 1/3   | 1/3  |
| H_B  | 1/2   | 0     | 0     | 1/2  |
| H_C  | 0     | 0     | 0     | 1    |
| H_D  | 0     | 1/2   | 1/2   | 0    |

The total click probability in the PageRank algorithm is normalized to sum to one, ensuring a coherent distribution of probabilities. This normalization reflects the likelihood of ending up on each of the pages within the system. Notably, the algorithm accounts for self-referential probabilities, acknowledging the potential for a page to refer back to itself in the web graph.

The page importance would be equally divided by among all pages it refers to.

$$ r_{i}=\sum_{S_{j}\in B_{i}}\frac{r_j}{|S_j|},i=1,2,...,n $$

The matrix 𝐻 is a normalized hyperlink squared matrix where:

$$ 
  H_{ij} =
    \begin{cases}
      \frac{1}{|S_i|} & \text{if a node $S_i$ has a link to $S_j$}\\
      0 & \text{otherwise}
    \end{cases}       
$$

## Page Relevance

In the PageRank algorithm, the importance of a page is directly proportional to the number of incoming links it receives. The rationale behind this is that a higher volume of inbound links signifies greater relevance and importance within the web ecosystem.

Conversely, outgoing links from a page are considered easier to manipulate or "fake" in terms of influence. Therefore, the algorithm places a heavier emphasis on the authenticity and credibility of incoming links.

Additionally, links originating from pages deemed important carry more weight in the determination of a page's significance. It's important to note that the algorithm faces challenges such as rank sinks and cycles, which can impose limitations on its effectiveness in certain scenarios.

![image](https://github.com/ferdeleong/pagerank/assets/78885738/653516f1-3e3f-4bc2-b350-073097e5844c)


## Markov Chains

A Markov chain is a mathematical concept that describes a stochastic process in which a system transitions from one state to another over discrete time steps. The key defining characteristic of a Markov chain is the Markov property, which states that the probability of transitioning to any particular state in the future depends solely on the current state and not on the sequence of events that preceded it. In other words, the future behavior of the system is independent of its past given its present state.

A Markov chain is typically represented by a state space, a set of possible states the system can occupy, and a transition matrix that specifies the probabilities of moving from one state to another. Each element in the matrix represents the probability of transitioning from one state to another in a single time step

### Stochasticity Adjustment

- Dangling Nodes
  
Nodes within the web graph that lack outgoing links (pdf files, image files). These pose a challenge as they can disrupt the flow of the algorithm, potentially impacting convergence.

$$ 
  a_{i} =
    \begin{cases}
      1 & \text{if page i is a dangling node}\\
      0 & \text{otherwise}
    \end{cases}       
$$

- Disconnected Sub-Graphs
  
The presence of disconnected sub-graphs in the web structure can lead to multiple eigenvectors with eigenvalue 1. This occurrence can complicate the determination of a unique and dominant solution in the PageRank calculation.

- Random Surfer
The concept of a random surfer characterizes the unpredictable navigation of a user on the web. This surfer follows the hyperlink structure randomly, contributing to the dynamics of the PageRank algorithm. The time spent on a page is utilized as a metric to gauge its relative importance, with repeated visits indicating higher significance in the ranking process.

By replacing ${0}^{T}$ rows with $\frac{1}{n}{e}^{T}$, `H` becomes stochastic.

![image](https://github.com/ferdeleong/pagerank/assets/78885738/069937f9-242e-4581-ad10-4b45385e6137)

a_i = {
  1, \text{if page i is a dangling node} \
  0, \text{otherwise}
}

$$ 
  a_{i} =
    \begin{cases}
      1 & \text{if node $S_i$ has no link to $S_j$}\\
      0 & \text{if node $S_i$ has links}
    \end{cases}       
$$

Adding random probability distribution, it can hyperlink to any page at random

$$ 
S = H + \frac{1}{n}ae^T
$$

### Primitivity Adjustment

A user could abandon the hyperlink surfing and entering a new destination in the browser’s URL line

Teleportation involves abandoning the traditional hyperlink surfing and instead directly entering a new destination in the browser’s URL line. The parameter α controls the balance between the random surfer following hyperlinks and teleporting to a new location.

After introducing teleportation, the hyperlink matrix undergoes modifications, rendering it irreducible and aperiodic. The **uniqueness of a positive PageRank vector** is guaranteed when the Google matrix satisfies conditions of stochasticity, irreducibility, and aperiodicity. 
 
$$ G = \alpha S + (1 - \alpha)\frac{1}{n}ee^T $$

### Power Method

In the PageRank algorithm, π represents the vector of initial rankings. The initial values for this vector are set as follows:

$$ \pi=\frac{1}{\left|G\right|} $$

The power method is then applied by multiplying the transpose matrix G by the current values of π to obtain the next iteration. 

$$ G^{T}\pi=\pi $$

This iterative process is crucial in converging towards a stable PageRank vector (dominant eigenvector) that accurately reflects the relative importance of each web page within the network.


#### Convergence Speed

How rapidly the algorithm is able to reach a state where further iterations produce negligible changes in the solution, meaning a stable ranking of web pages.

We determine when we reach convergence by using the L1 norm. This means calculating the sum of the absolute values in our vectors. When the difference between iterations is negligible, convergence is reached.
The average number of iterations before convergence and reach PageRank vector is 50-100

### Eigenvalue & Unique Solution
The principal eigenvector with eigenvalue λ = 1 represents the stationary distribution of web surfing probabilities,or the solution to PageRank algorithm.

- G is a stochastic
  
Ensures that columns sum to 1 and form a valid probability distribution.

- G is primitive

  - Irreducibility of G: No disconnected subgraphs, only 1 eigenvalue.
  
  - Aperiodicity of G: No trapped in a subset of states, convergence assured.

- Dominant Eigenvalue λ1 ​= 1 is implies that power method will converge to a unique positive vector.

- Perron-Frobenius Theorem: 
Ensures that a stochastic matrix has a stationary distribution with non-negative entries, irreducibility, and a unique dominant eigenvalue with a corresponding eigenvector

## Computational Efficiency

H is a very sparse matrix (a large proportion of its elements are 0) because most webpages link to only a handful of other pages
 
In the power method, the G matrix is accessed solely through vector-matrix multiplication. There is no need for extensive modification or storage of the matrix during the iterative process. This approach not only enhances computational efficiency but also makes the algorithm storage-friendly. The matrix-free characteristic ensures that the PageRank algorithm can effectively determine webpage rankings without imposing undue demands on memory resources.

Vector-matrix multiplication involving a sparse matrix requires much less effort than the $O(n^2)$ dense computation. In fact, it requires $O(nnz(H))$ computation, where $nnz(H)$ is the number of nonzeros in H.

The vector-matrix multiplication reduces to **O(n) effort**.

The power method applied to G can actually be expressed in terms of the very sparse H.

$\pi^{(k+1)T} = \pi^{(k)T} G$

$= \alpha \pi^{(k)T} S + \frac{(1 - \alpha)}{n} \pi^{(k)T} ee^T$

$= \alpha \pi^{(k)T} H + \frac{\alpha \pi^{(k)T} a + (1 - \alpha) e^T}{n}$

The vector-matrix multiplications $(π(k)T H)$ are executed on the extremely sparse H, and S and G are never formed or stored, only their rank-one components, a and e, are needed.
