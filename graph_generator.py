import os
import networkx as nx
import argparse
import random

def generate_data(n_graphs, n_nodes, p):
    # Create directories if they don't exist
    os.makedirs("data/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir("data/input_graphs"):
        for file_name in os.listdir("data/input_graphs"):
            file_path = os.path.join("data/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs("data/prompts", exist_ok=True)
    os.makedirs("data/prompts/add_edge", exist_ok=True)
    os.makedirs("data/prompts/remove_edge", exist_ok=True)
    os.makedirs("data/prompts/node_count", exist_ok=True)
    os.makedirs("data/prompts/edge_count", exist_ok=True)
    os.makedirs("data/prompts/node_degree", exist_ok=True)
    os.makedirs("data/prompts/edge_exists", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        "data/prompts/add_edge",
        "data/prompts/remove_edge",
        "data/prompts/node_count",
        "data/prompts/edge_count",
        "data/prompts/node_degree",
        "data/prompts/edge_exists"
    ]
    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs("data/solutions", exist_ok=True)
    os.makedirs("data/solutions/add_edge", exist_ok=True)
    os.makedirs("data/solutions/remove_edge", exist_ok=True)
    os.makedirs("data/solutions/node_count", exist_ok=True)
    os.makedirs("data/solutions/edge_count", exist_ok=True)
    os.makedirs("data/solutions/node_degree", exist_ok=True)
    os.makedirs("data/solutions/edge_exists", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        "data/solutions/add_edge",
        "data/solutions/remove_edge",
        "data/solutions/node_count",
        "data/solutions/edge_count",
        "data/solutions/node_degree",
        "data/solutions/edge_exists"
    ]
    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)
    
    for i in range(n_graphs):
        print(f"Generating graph {i}")
        # Generate Erdos-Renyi graph that is not connected
        graph = nx.erdos_renyi_graph(n_nodes, p)

        while nx.is_connected(graph):
            graph = nx.erdos_renyi_graph(n_nodes, p)

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
        #end_count_prompt = ", and surround your final answer in parentheses, like this: (answer). \nA:" # kind of works
        end_count_prompt = "A: Final answer: The final answer is: "
        end_yes_no_prompt = "A: Final answer: The final answer is: " #TODO: something like "present answer as _"
        # Graph augmentation tasks

        # ----------------------------
        # --- Add edge  ---
        # ----------------------------

        # Select two random nodes that are not connected
        unconnected_nodes = []
        for node_a in graph.nodes():
            for node_b in graph.nodes():
                if node_a != node_b and not graph.has_edge(node_a, node_b):
                    unconnected_nodes.append((node_a, node_b))
        #print(f"unconnected_nodes: {unconnected_nodes}")
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]

        # Create prompt string
        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}. Only write the resulting adjacency matrix.\n"
        full_add_edge_prompt = init_prompt + graph_str + "\n" + add_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/add_edge/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/solutions/add_edge/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)

        # ----------------------------
        # --- Remove edge  ---
        # ----------------------------

        # Select a random edge
        edge = random.choice(list(graph.edges()))

        # Create prompt string
        node_a, node_b = edge
        remove_edge_prompt = f"Q: Remove the edge between node {node_a} and node {node_b}. Only write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = init_prompt + graph_str + "\n" + remove_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/remove_edge/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_edge_graph).todense().astype(int))

        # Write new graph to file
        solution_filename = f"data/solutions/remove_edge/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)

        # Basic graph tasks

        # ----------------------------
        # --- Node count  ---
        # ----------------------------

        node_count = graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in this graph?\n"
        full_node_count_prompt = init_prompt + graph_str + "\n" + node_count_prompt + end_count_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/node_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/node_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_count))

        # ----------------------------
        # --- Edge count  ---
        # ----------------------------

        edge_count = graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in this graph?\n"
        full_edge_count_prompt = init_prompt + graph_str + "\n" + edge_count_prompt + end_count_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/edge_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/edge_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(edge_count))

        # ----------------------------
        # --- Node degree  ---
        # ----------------------------

        # Select a random node
        node = random.choice(list(graph.nodes()))
        node_degree = graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: What is the degree of node {node}?\n"
        full_node_degree_prompt = init_prompt + graph_str + "\n" + node_degree_prompt + end_count_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/node_degree/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/node_degree/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_degree))

        # ----------------------------
        # --- Edge exists  ---
        # ----------------------------

        # Select two random nodes from the graph
        random_nodes = random.sample(list(graph.nodes()), 2)
        node_a, node_b = random_nodes

        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}? Only write 'Yes' or 'No'.\n"
        full_edge_exists_prompt = init_prompt + graph_str + "\n" + edge_exists_prompt + end_yes_no_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/edge_exists/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/edge_exists/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if graph.has_edge(node_a, node_b):
                solution_file.write("Yes")
            else:
                solution_file.write("No")

    print("Data generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, help="number of graphs to generate")
    parser.add_argument("--n_nodes", type=int, help="number of nodes in the graph")
    parser.add_argument("--p", type=float, help="probability of an edge between any two nodes")
    #parser.add_argument("--prompt_type", type=str, default="add_edge", help="type of prompt")
    args = parser.parse_args()

    n_graphs = args.n_graphs
    n_nodes = args.n_nodes
    p = args.p
    #prompt_type = args.prompt_type

    generate_data(n_graphs, n_nodes, p)