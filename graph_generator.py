import os
import networkx as nx
import argparse
import random

def generate_data(n_graphs, n_nodes, p, prompt_type):
    for i in range(n_graphs):
        # Create directories if they don't exist
        os.makedirs("data/input_graphs", exist_ok=True)
        os.makedirs("data/solutions", exist_ok=True)

        # Generate Erdos-Renyi graph
        graph = nx.erdos_renyi_graph(n_nodes, p)

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        if prompt_type == "add_edge":
            # Select two random nodes that are not connected
            unconnected_nodes = []
            for node_a in graph.nodes():
                for node_b in graph.nodes():
                    if node_a != node_b and not graph.has_edge(node_a, node_b):
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]

            # Create prompt string
            prompt = f"Add an edge between node {node_a} and node {node_b}"

            # Save prompt to file
            prompt_filename = f"data/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(prompt)

            # Create new graph with added edge
            graph.add_edge(node_a, node_b)

        elif prompt_type == "remove_edge":
            # Select a random edge
            edge = random.choice(list(graph.edges()))

            # Create prompt string
            node_a, node_b = edge
            prompt = f"Remove the edge between node {node_a} and node {node_b}"

            # Save prompt to file
            prompt_filename = f"data/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(prompt)

            # Create new graph with edge removed
            graph.remove_edge(*edge)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write new graph to file
        solution_filename = f"data/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, help="number of graphs to generate")
    parser.add_argument("--n_nodes", type=int, help="number of nodes in the graph")
    parser.add_argument("--p", type=float, help="probability of an edge between any two nodes")
    parser.add_argument("--prompt_type", type=str, default="add_edge", help="type of prompt")
    args = parser.parse_args()

    n_graphs = args.n_graphs
    n_nodes = args.n_nodes
    p = args.p
    prompt_type = args.prompt_type

    generate_data(n_graphs, n_nodes, p, prompt_type)