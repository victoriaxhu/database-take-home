#!/usr/bin/env python3
import json
import os
import sys
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Add project root to path to import scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import constants
from scripts.constants import (
    NUM_NODES,
    MAX_EDGES_PER_NODE,
    MAX_TOTAL_EDGES,
)


def load_graph(graph_file):
    """Load graph from a JSON file."""
    with open(graph_file, "r") as f:
        return json.load(f)


def load_results(results_file):
    """Load query results from a JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def save_graph(graph, output_file):
    """Save graph to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)


def verify_constraints(graph, max_edges_per_node, max_total_edges):
    """Verify that the graph meets all constraints."""
    # Check total edges
    total_edges = sum(len(edges) for edges in graph.values())
    if total_edges > max_total_edges:
        print(
            f"WARNING: Graph has {total_edges} edges, exceeding limit of {max_total_edges}"
        )
        return False

    # Check max edges per node
    max_node_edges = max(len(edges) for edges in graph.values())
    if max_node_edges > max_edges_per_node:
        print(
            f"WARNING: A node has {max_node_edges} edges, exceeding limit of {max_edges_per_node}"
        )
        return False

    # Check all nodes are present
    if len(graph) != NUM_NODES:
        print(f"WARNING: Graph has {len(graph)} nodes, should have {NUM_NODES}")
        return False

    # Check edge weights are valid (between 0 and 10)
    for node, edges in graph.items():
        for target, weight in edges.items():
            if weight <= 0 or weight > 10:
                print(f"WARNING: Edge {node} -> {target} has invalid weight {weight}")
                return False

    return True

def optimize_graph(
    initial_graph,
    results,
    num_nodes=NUM_NODES,
    max_total_edges=int(MAX_TOTAL_EDGES),
    max_edges_per_node=MAX_EDGES_PER_NODE,
):
    """
    Optimize the graph to improve random walk query performance.
    Special focus on improving access to nodes 0-19, particularly 0-4.

    Args:
        initial_graph: Initial graph adjacency list (nodes 0-499)
        results: Results from queries on the initial graph
        num_nodes: Number of nodes in the graph (500)
        max_total_edges: Maximum total edges allowed
        max_edges_per_node: Maximum edges per node (3)

    Returns:
        Optimized graph
    """
    print("Starting graph optimization...")

    # Create a copy of the initial graph to modify
    optimized_graph = {}
    for node, edges in initial_graph.items():
        optimized_graph[node] = dict(edges)

    # Define priority nodes (top 20 most visited, with 0-4 being highest priority)
    highest_priority_nodes = set(range(5))  # Nodes 0-4
    high_priority_nodes = set(range(5, 20))  # Nodes 5-19
    
    # Extract the detailed results which contain the query paths
    detailed_results = results.get("detailed_results", [])
    
    if not detailed_results:
        print("WARNING: No detailed results found. Using original graph.")
        return optimized_graph

    # Initialize counters
    edge_success_counter = Counter()
    edge_failure_counter = Counter()
    node_failure_counter = Counter()
    
    # Process each query result
    for query_result in detailed_results:
        target = query_result.get("target")
        is_success = query_result.get("is_success")
        paths = query_result.get("paths", [])
        
        if not is_success:
            # Weight failures by priority
            if target in highest_priority_nodes:
                node_failure_counter[target] += 3  # Triple weight for highest priority
            elif target in high_priority_nodes:
                node_failure_counter[target] += 2  # Double weight for high priority
            else:
                node_failure_counter[target] += 1
        
        # Process each path in the paths list
        for path_entry in paths:
            if len(path_entry) >= 2:
                success = path_entry[0]  # First element is success flag
                path = path_entry[1]     # Second element is the path list
                
                # Process edges in the path
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    edge = (src, dst)
                    
                    # If the destination is a priority node, give this edge special attention
                    priority_multiplier = 1
                    if dst in highest_priority_nodes:
                        priority_multiplier = 3
                    elif dst in high_priority_nodes:
                        priority_multiplier = 2
                    
                    if success:
                        edge_success_counter[edge] += 1 * priority_multiplier
                    else:
                        edge_failure_counter[edge] += 1
    
    # Determine node type for consistent keys
    node_type = type(next(iter(optimized_graph.keys())))
    
    # PHASE 1: Reinforce successful edges
    for (src, dst), count in edge_success_counter.items():
        src_key = src if isinstance(src, node_type) else node_type(src)
        dst_key = dst if isinstance(dst, node_type) else node_type(dst)
        
        if src_key in optimized_graph and dst_key in optimized_graph[src_key]:
            # Increase weight for successful edges
            optimized_graph[src_key][dst_key] = min(optimized_graph[src_key][dst_key] + count, 10)

    # PHASE 2: Weaken edges that lead to failures
    for (src, dst), count in edge_failure_counter.items():
        src_key = src if isinstance(src, node_type) else node_type(src)
        dst_key = dst if isinstance(dst, node_type) else node_type(dst)
        
        if src_key in optimized_graph and dst_key in optimized_graph[src_key]:
            # Don't reduce weight for edges leading to priority nodes as much
            reduction = count
            if dst_key in highest_priority_nodes:
                reduction = max(1, count // 3)  # Reduced penalty
            elif dst_key in high_priority_nodes:
                reduction = max(1, count // 2)  # Somewhat reduced penalty
            
            optimized_graph[src_key][dst_key] = max(optimized_graph[src_key][dst_key] - reduction, 1)

    # PHASE 3: Improve connectivity to priority nodes
    # First, handle highest priority nodes (0-4)
    for target in highest_priority_nodes:
        target_key = target if isinstance(target, node_type) else node_type(target)
        
        # Find nodes that could benefit from a direct connection to this priority target
        potential_sources = []
        for node in optimized_graph:
            if node != target_key and target_key not in optimized_graph[node] and len(optimized_graph[node]) < max_edges_per_node:
                potential_sources.append(node)
        
        # Sort potential sources by distance from other high-priority nodes
        potential_sources.sort(key=lambda x: sum(1 for pri in highest_priority_nodes if node_type(pri) in optimized_graph[x]))
        
        # Add connections from up to 20 nodes to this high priority target
        connections_to_add = min(20, len(potential_sources))
        for i in range(connections_to_add):
            optimized_graph[potential_sources[i]][target_key] = 10  # Maximum weight
    
    # Then handle secondary priority nodes (5-19)
    for target in high_priority_nodes:
        target_key = target if isinstance(target, node_type) else node_type(target)
        
        # Add fewer connections to these secondary priority nodes
        potential_sources = []
        for node in optimized_graph:
            if node != target_key and target_key not in optimized_graph[node] and len(optimized_graph[node]) < max_edges_per_node:
                potential_sources.append(node)
        
        potential_sources.sort(key=lambda x: sum(1 for pri in high_priority_nodes if node_type(pri) in optimized_graph[x]))
        
        # Add connections from up to 10 nodes to this secondary priority target
        connections_to_add = min(10, len(potential_sources))
        for i in range(connections_to_add):
            optimized_graph[potential_sources[i]][target_key] = 8  # High weight
    
    # PHASE 4: Create "hub" nodes that connect to multiple priority targets
    # Find nodes that have space for more connections and make them hubs
    potential_hubs = []
    for node in optimized_graph:
        if len(optimized_graph[node]) < max_edges_per_node - 1:  # Need at least 2 slots
            priority_connections = sum(1 for dst in optimized_graph[node] if dst in highest_priority_nodes or dst in high_priority_nodes)
            potential_hubs.append((node, priority_connections, len(optimized_graph[node])))
    
    # Sort by: most existing priority connections, then fewest total edges
    potential_hubs.sort(key=lambda x: (-x[1], x[2]))
    
    # Select top 25 potential hubs and connect them to priority nodes they're not already connected to
    for i, (hub, _, _) in enumerate(potential_hubs[:25]):
        # Connect to highest priority nodes first
        for target in highest_priority_nodes:
            target_key = target if isinstance(target, node_type) else node_type(target)
            if target_key not in optimized_graph[hub] and len(optimized_graph[hub]) < max_edges_per_node:
                optimized_graph[hub][target_key] = 10
        
        # Then connect to secondary priority nodes if space
        for target in high_priority_nodes:
            target_key = target if isinstance(target, node_type) else node_type(target)
            if target_key not in optimized_graph[hub] and len(optimized_graph[hub]) < max_edges_per_node:
                optimized_graph[hub][target_key] = 8
    
    # PHASE 5: Prune edges if we exceed the maximum total
    total_edges = sum(len(edges) for edges in optimized_graph.values())
    
    if total_edges > max_total_edges:
        print(f"Need to prune {total_edges - max_total_edges} edges")
        
        # Create a list of all edges with their utility scores
        edge_utility = []
        for src, edges in optimized_graph.items():
            for dst, weight in edges.items():
                # Calculate utility (higher = more useful)
                dst_int = int(dst) if not isinstance(dst, int) else dst
                
                # Base utility on priority and usage
                priority_factor = 1
                if dst_int in highest_priority_nodes:
                    priority_factor = 10  # Strongly protect edges to highest priority nodes
                elif dst_int in high_priority_nodes:
                    priority_factor = 5   # Protect edges to high priority nodes
                
                success_count = edge_success_counter.get((src, dst), 0)
                failure_count = edge_failure_counter.get((src, dst), 0) + 1  # +1 to avoid division by zero
                
                utility = (success_count + 1) * priority_factor * weight / failure_count
                edge_utility.append((utility, src, dst))
        
        # Sort by utility (ascending - we'll remove lowest utility edges first)
        edge_utility.sort()
        
        # Remove edges until we're within the limit
        edges_to_remove = total_edges - max_total_edges
        removed = 0
        
        for _, src, dst in edge_utility:
            # Never remove an edge if it's the node's only outgoing edge
            # Also protect edges to priority nodes if possible
            dst_int = int(dst) if not isinstance(dst, int) else dst
            
            # Skip this edge if:
            # 1. It's the node's only edge (would disconnect the node)
            # 2. It's to a highest priority node and node has >2 edges
            # 3. It's to a high priority node and node has >1 edge and we still have other options
            if len(optimized_graph[src]) <= 1:
                continue
            
            if dst_int in highest_priority_nodes and len(optimized_graph[src]) <= 2:
                continue
                
            if dst_int in high_priority_nodes and len(optimized_graph[src]) <= 1 and removed < edges_to_remove - 1:
                continue
            
            # Remove this edge
            del optimized_graph[src][dst]
            removed += 1
            
            if removed >= edges_to_remove:
                break
    
    return optimized_graph

if __name__ == "__main__":
    # Get file paths
    initial_graph_file = os.path.join(project_dir, "data", "initial_graph.json")
    results_file = os.path.join(project_dir, "data", "initial_results.json")
    output_file = os.path.join(
        project_dir, "candidate_submission", "optimized_graph.json"
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading initial graph from {initial_graph_file}")
    initial_graph = load_graph(initial_graph_file)

    print(f"Loading query results from {results_file}")
    results = load_results(results_file)

    print("Optimizing graph...")
    optimized_graph = optimize_graph(initial_graph, results)

    print(f"Saving optimized graph to {output_file}")
    save_graph(optimized_graph, output_file)

    print("Done! Optimized graph has been saved.")
