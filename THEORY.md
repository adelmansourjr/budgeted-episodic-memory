# Best fixed-batch coverage in a memory-routing tree

A finite tree indexes possible evidence chains. Each node costs a positive integer number of read units. Exactly one leaf is relevant, drawn before selection with known probability mass $p_ell$. A complete evidence set consists of every node from the root to that leaf. Node costs and tree structure are public. Before any paid read, a strategy knows the prior distribution and no information about actual leaf identity. A read of a node on the realized route identifies the next child exactly. We score evidence completeness independently of any final answer guessed from incomplete evidence. This model can represent preview information by substituting a posterior for $p_ell$, as long as that posterior is fixed before the batch selection.

Let $F_v(b)$ be the largest probability mass of fully covered leaf paths *within* the subtree rooted at $v$, conditioned on having paid for the ancestors outside it but measured in unconditional leaf mass. For cost $c_v$, a budget $b<c_v$ gives zero. A leaf costs $c_v$ and contributes $p_v$ when affordable. For an internal node,

$$F_v(b)=max_{sum_i b_i le b-c_v}sum_i F_{u_i}(b_i),$$

where $u_i$ range over the children. The optimum fixed-batch coverage is $F_{root}(B)$.

**Proof.** Every fixed batch may discard any purchased node not on a fully covered leaf path, since it cannot contribute to the objective. The retained union of paths is prefix-closed. If it covers any leaf in subtree $v$ it must pay $c_v$, leaving the rest of the budget to divide among the disjoint child subtrees. This gives the upper bound by induction. Conversely, the root and feasible optima in each child use the stated budget and attain the summed probability, proving equality. A randomized batch is a distribution over deterministic feasible batches, so cannot improve the best value. This is a direct leaf-mass version of rooted [tree knapsack](https://pubsonline.informs.org/doi/10.1287/ijoc.9.4.431).

An adaptive policy follows the exact revealed route, returning full support for every leaf whose path cost is at most $B$. No policy can complete a more expensive path. Therefore its optimal coverage is exactly $sum_ell p_ell \mathbf{1}[\sum_{v\in path(\ell)} c_v\le B]$.

For depth $d$, branching factor $m$, uniform leaves and unit costs, budget $d+1$ can buy exactly one path in a batch, giving coverage $m^{-d}$. Adaptive traversal reaches probability 1 at the same budget. A 16-branch star with six affordable reads has tight fixed-batch coverage 5/16 because one of the reads buys the root. If the prior knows the branch beforehand, both approaches can succeed. A missing or wrong pointer would invalidate the adaptive achievability argument. Natural text can provide noisy links, ambiguous alternatives, cross edges and partial evidence sufficient to answer. The theorem establishes an information gap under the specified assumptions, without transferring its magnitude to those settings.
