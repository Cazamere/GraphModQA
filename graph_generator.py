import os
import networkx as nx
import argparse
import random
import sys
import numpy as np

from graph_generator_utils import add_edge, remove_edge, add_node, remove_node, node_count, edge_count, node_degree, edge_exists, connected_nodes, cycle, chain

def generate_data(n_graphs):
    # TODO: put all dir prep stuff this in a function

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
    os.makedirs("data/prompts/add_node", exist_ok=True)
    os.makedirs("data/prompts/remove_node", exist_ok=True)
    os.makedirs("data/prompts/node_count", exist_ok=True)
    os.makedirs("data/prompts/edge_count", exist_ok=True)
    os.makedirs("data/prompts/node_degree", exist_ok=True)
    os.makedirs("data/prompts/edge_exists", exist_ok=True)
    os.makedirs("data/prompts/connected_nodes", exist_ok=True)
    os.makedirs("data/prompts/cycle", exist_ok=True)
    os.makedirs("data/prompts/chain_node_count", exist_ok=True)
    os.makedirs("data/prompts/chain_edge_count", exist_ok=True)
    os.makedirs("data/prompts/chain_node_degree", exist_ok=True)
    os.makedirs("data/prompts/chain_edge_exists", exist_ok=True)
    os.makedirs("data/prompts/chain_connected_nodes", exist_ok=True)
    os.makedirs("data/prompts/chain_cycle", exist_ok=True)
    os.makedirs("data/prompts/chain_print", exist_ok=True)
    # Empty the directories if they are not empty
    prompt_directories = [
        "data/prompts/add_edge",
        "data/prompts/remove_edge",
        "data/prompts/add_node",
        "data/prompts/remove_node",
        "data/prompts/node_count",
        "data/prompts/edge_count",
        "data/prompts/node_degree",
        "data/prompts/edge_exists",
        "data/prompts/connected_nodes",
        "data/prompts/cycle",
        "data/prompts/chain_node_count",
        "data/prompts/chain_edge_count",
        "data/prompts/chain_node_degree",
        "data/prompts/chain_edge_exists",
        "data/prompts/chain_connected_nodes",
        "data/prompts/chain_cycle",
        "data/prompts/chain_print"
    ]
    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs("data/solutions", exist_ok=True)
    os.makedirs("data/solutions/add_edge", exist_ok=True)
    os.makedirs("data/solutions/remove_edge", exist_ok=True)
    os.makedirs("data/solutions/add_node", exist_ok=True)
    os.makedirs("data/solutions/remove_node", exist_ok=True)
    os.makedirs("data/solutions/node_count", exist_ok=True)
    os.makedirs("data/solutions/edge_count", exist_ok=True)
    os.makedirs("data/solutions/node_degree", exist_ok=True)
    os.makedirs("data/solutions/edge_exists", exist_ok=True)
    os.makedirs("data/solutions/connected_nodes", exist_ok=True)
    os.makedirs("data/solutions/cycle", exist_ok=True)
    os.makedirs("data/solutions/chain_node_count", exist_ok=True)
    os.makedirs("data/solutions/chain_edge_count", exist_ok=True)
    os.makedirs("data/solutions/chain_node_degree", exist_ok=True)
    os.makedirs("data/solutions/chain_edge_exists", exist_ok=True)
    os.makedirs("data/solutions/chain_connected_nodes", exist_ok=True)
    os.makedirs("data/solutions/chain_cycle", exist_ok=True)
    os.makedirs("data/solutions/chain_print", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        "data/solutions/add_edge",
        "data/solutions/remove_edge",
        "data/solutions/add_node",
        "data/solutions/remove_node",
        "data/solutions/node_count",
        "data/solutions/edge_count",
        "data/solutions/node_degree",
        "data/solutions/edge_exists",
        "data/solutions/connected_nodes",
        "data/solutions/cycle",
        "data/solutions/chain_node_count",
        "data/solutions/chain_edge_count",
        "data/solutions/chain_node_degree",
        "data/solutions/chain_edge_exists",
        "data/solutions/chain_connected_nodes",
        "data/solutions/chain_cycle",
        "data/solutions/chain_print"
    ]
    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]

    possible_final_tasks = static_tasks + ["print_adjacency_matrix"]
    
    for i in range(n_graphs):
        print(f"Generating graph {i}")
        # Generate Erdos-Renyi graph that is not connected
        p = random.uniform(0, 1)
        n_nodes = random.randint(5, 20)
        graph = nx.erdos_renyi_graph(n_nodes, p)

        # Ensure that the graph is not fully connected and has at least one edge
        while nx.is_connected(graph) or graph.number_of_edges() == 0:
            p = random.uniform(0, 1)
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
        add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False)
        remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False)
        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False)
        node_count(graph, graph_str, init_prompt, end_count_prompt, i)
        edge_count(graph, graph_str, init_prompt, end_count_prompt, i)
        node_degree(graph, graph_str, init_prompt, end_count_prompt, i)
        edge_exists(graph, graph_str, init_prompt, end_yes_no_prompt, i)
        connected_nodes(graph, graph_str, init_prompt, end_count_prompt, i)
        cycle(graph, graph_str, init_prompt, end_yes_no_prompt, i)
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "edge_exists")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "cycle")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_count")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "edge_count")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_degree")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "connected_nodes")
        chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_matrix_prompt, i, "print_adjacency_matrix")

    print("Data generation complete!")


def generate_chains(n_graphs):
    for chain_length in range(1, 11):
        # Create directories if they don't exist
        os.makedirs(f"data/prompts/chain_{chain_length}_node_count", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_edge_count", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_node_degree", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_edge_exists", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_connected_nodes", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_cycle", exist_ok=True)
        os.makedirs(f"data/prompts/chain_{chain_length}_print", exist_ok=True)

        # Empty the directories if they are not empty
        prompt_directories = [
            f"data/prompts/chain_{chain_length}_node_count",
            f"data/prompts/chain_{chain_length}_edge_count",
            f"data/prompts/chain_{chain_length}_node_degree",
            f"data/prompts/chain_{chain_length}_edge_exists",
            f"data/prompts/chain_{chain_length}_connected_nodes",
            f"data/prompts/chain_{chain_length}_cycle",
            f"data/prompts/chain_{chain_length}_print"
        ]

        for directory in prompt_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

        os.makedirs(f"data/solutions/chain_{chain_length}_node_count", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_edge_count", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_node_degree", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_edge_exists", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_connected_nodes", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_cycle", exist_ok=True)
        os.makedirs(f"data/solutions/chain_{chain_length}_print", exist_ok=True)

        # Empty the directories if they are not empty
        solution_directories = [
            f"data/solutions/chain_{chain_length}_node_count",
            f"data/solutions/chain_{chain_length}_edge_count",
            f"data/solutions/chain_{chain_length}_node_degree",
            f"data/solutions/chain_{chain_length}_edge_exists",
            f"data/solutions/chain_{chain_length}_connected_nodes",
            f"data/solutions/chain_{chain_length}_cycle",
            f"data/solutions/chain_{chain_length}_print"
        ]

        for directory in solution_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"

    for chain_length in range(1, 11):
        for i in range(n_graphs):
            print(f"Generating chain prompt {i}")
            # Generate Erdos-Renyi graph that is not connected
            input_filename = f"graph_{i}.txt"

            # Read input graph
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                graph_str = input_file.read()

            graph = nx.from_numpy_matrix(np.array(eval(graph_str)))

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: Final answer: The final answer is: "
            end_yes_no_prompt = "A: Final answer: The final answer is: "

            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "edge_exists", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "cycle", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_count", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "edge_count", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_degree", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "connected_nodes", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_matrix_prompt, i, "print_adjacency_matrix", chain_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, help="number of graphs to generate")
    #parser.add_argument("--n_nodes", type=int, help="number of nodes in the graph")
    #parser.add_argument("--p", type=float, help="probability of an edge between any two nodes")
    #parser.add_argument("--prompt_type", type=str, default="add_edge", help="type of prompt")
    parser.add_argument("--chain", type=bool, help="whether to generate chain prompts")
    
    args = parser.parse_args()

    n_graphs = args.n_graphs
    #n_nodes = args.n_nodes
    #p = args.p
    #prompt_type = args.prompt_type
    chain = args.chain
    chain = True
    if chain:
        print("Generating chain prompts")
        generate_chains(n_graphs)
    else:  
        generate_data(n_graphs)