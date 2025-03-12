## Solution

### Approach & Analysis

Based on the initial results data, I noticed that the top 20 target nodes tended to fall within the first 20 node IDs, with especially high frequency for nodes 0, 1, 2, 3, 4. Based on the query frequency graph it also seemed like
there were two types of nodes; some which were queried more than 11 times and others which were queried fewer than 11 times, and this formed two humps in the bar graph.

### Optimization Strategy

In general, we reinforce edges on successful paths and weaken edges that lead to failures. We improve connectivity to priority nodes (0-20) by adding connections to priority nodes. For nodes which have room for more edges, we make them into hub nodes which connect priority targets. Because we have added so many connections, we finally work on pruning the graph to meet the limits of 3 neighbors each: We rank edges by utility and then getting rid of the low utility edges, while taking care to never get rid of edges which are the only one connecting a node to the rest of the graph.

### Implementation Details

There were 5 phases, effectively described as above. Nodes 0-4 are tier 1 nodes and nodes 5-19 are tier 2. Reinforce successful edges, weaken edges on failures with double weight for tier 2 and triple weight for tier 1. Search for nodes with room for more edges; link up to 20 potential sources to join tier 1 nodes and up to 10 potential sources to join tier 2 nodes. Then, create hub nodes (sorted by most priority connections and then fewest total edges) to
connect them with tier1/tier2 nodes which they are not already connected to. Finally, create a list of utilities for all edges and prune lowest edge utility.

### Results

The optimized graph improved both in terms of success rate and path lengths. I pasted the data below:
SUCCESS RATE:
Initial: 79.5% (159/200)
Optimized: 90.5% (181/200)
✅ Improvement: 11.0%

PATH LENGTHS (successful queries only):
Initial: 548.5 (159/200 queries)
Optimized: 78.5 (181/200 queries)
✅ Improvement: 85.7%

COMBINED SCORE (success rate × path efficiency):
Score: 278.55
Higher is better, rewards both success and shorter paths

### Trade-offs & Limitations

One trade off is that reinforcing successful edges might cause a few paths to become very strongly favored, which limits exploration and might decrease success rate for target nodes inaccessible from those paths. Another trade off is that while improving connectivity to the tier1/tier2 nodes makes paths to them shorter and these nodes are more important to our queries, overcrowding these specific nodes might make certain paths redundant. I'm also worried
that my pruning might create disconnected components in the graph, or that it makes non-prioritized nodes less accessible.

The strategy is also heavily tailored towards these specific queries since it separates nodes into tiers, and doesn't generalize well.

### Iteration Journey

At first, my general approach was to lower the weights of edges on failed paths and increase weights of edges
on successful paths; I was hoping that this would help the optimized graph "learn" from the results of the initial iteration. I actually didn't realize that the queries were biased towards lower numbered nodes until later, and added in priority weighting for that later, as well as improving connectivity
from farther nodes to priority nodes in order to shorten path lengths overall.

---

- Be concise but thorough - aim for 500-1000 words total
- Include specific data and metrics where relevant
- Explain your reasoning, not just what you did
