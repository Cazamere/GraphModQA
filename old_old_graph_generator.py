import os
import networkx as nx
import argparse
import random
import sys
import numpy as np

from graph_generator_utils import *

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
    for chain_length in range(1, 6):
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

    for chain_length in range(1, 6):
        for i in range(n_graphs):
            print(f"Generating chain prompt {i}")
            # Generate Erdos-Renyi graph that is not connected
            input_filename = f"graph_{i}.txt"

            # Read input graph
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                graph_str = input_file.read()

            # add commas in between the numbers in graph_str
            graph_str = graph_str.replace(" ", ", ")

            graph = nx.from_numpy_array(np.array(eval(graph_str)))

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: Final answer: The final answer is: "
            end_yes_no_prompt = "A: Final answer: The final answer is: "

            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_count", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "edge_count", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "node_degree", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "edge_exists", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_count_prompt, i, "connected_nodes", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_yes_no_prompt, i, "cycle", chain_length)
            chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_matrix_prompt, i, "print_adjacency_matrix", chain_length)

def generate_chains_same(n_graphs):
    for final_task in ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]:
        for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
            for chain_length in range(1, 6):
                # Create directories if they don't exist
                ablation_dir = "chains_same"
                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", exist_ok=True)

                # Empty the directory if it's not empty
                if os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                    for file_name in os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                        file_path = os.path.join(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", file_name)
                        os.remove(file_path)

                
                # Create directories if they don't exist
                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/prompts", exist_ok=True)

                # Empty the directories if they are not empty
                prompt_directories = [
                    f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/prompts"
                ]

                for directory in prompt_directories:
                    if os.listdir(directory):
                        for file_name in os.listdir(directory):
                            file_path = os.path.join(directory, file_name)
                            os.remove(file_path)

                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/solutions", exist_ok=True)

                # Empty the directories if they are not empty
                solution_directories = [
                    f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/solutions"
                ]

                for directory in solution_directories:
                    if os.listdir(directory):
                        for file_name in os.listdir(directory):
                            file_path = os.path.join(directory, file_name)
                            os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"

    for chain_length in range(1, 6):
        for task in augment_tasks:
            for i in range(n_graphs):
                print(f"Generating chain prompt {i}")
                p = random.uniform(0, 1)
                if task == "remove_node":
                    n = random.randint(6, 20)
                else:
                    n = random.randint(5, 20)
                graph = nx.erdos_renyi_graph(n, p)

                def is_complete_graph(G):
                    n = len(G.nodes)
                    # A complete graph with n nodes has n*(n-1)/2 edges
                    expected_num_edges = n * (n - 1) // 2
                    actual_num_edges = len(G.edges)
                    return actual_num_edges == expected_num_edges

                # Check if the graph is complete
                is_complete = is_complete_graph(graph)
                #print("Graph is a complete graph:", is_complete)

                # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                while (task == "remove_edge" and graph.number_of_edges() < 5):
                    p = random.uniform(0, 1)
                    graph = nx.erdos_renyi_graph(n, p)

                # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
                while (task == "add_edge" and graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                    p = random.uniform(0, 1)
                    graph = nx.erdos_renyi_graph(n, p)

                # Convert graph to string
                graph_str = str(nx.adjacency_matrix(graph).todense())

                

                init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
                end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
                end_count_prompt = "A: Final answer: The final answer is: "
                end_yes_no_prompt = "A: Final answer: The final answer is: "

                final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

                for final_task in final_tasks:
                    # Write graph to file
                    graph_filename = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/graph_{i}.txt"
                    with open(graph_filename, "w") as graph_file:
                        graph_file.write(graph_str)

                    if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                        chain_same(graph, graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_task, chain_length)
                    elif final_task == "print_adjacency_matrix":
                        chain_same(graph, graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, chain_length)

def generate_chains_same_cot(n_graphs):
    for final_task in ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]:
        for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
            for chain_length in range(1, 6):
                # Create directories if they don't exist
                ablation_dir = "chains_same_cot"
                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", exist_ok=True)

                # Empty the directory if it's not empty
                if os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                    for file_name in os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                        file_path = os.path.join(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", file_name)
                        os.remove(file_path)

                
                # Create directories if they don't exist
                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/prompts", exist_ok=True)

                # Empty the directories if they are not empty
                prompt_directories = [
                    f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/prompts"
                ]

                for directory in prompt_directories:
                    if os.listdir(directory):
                        for file_name in os.listdir(directory):
                            file_path = os.path.join(directory, file_name)
                            os.remove(file_path)

                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/solutions", exist_ok=True)

                # Empty the directories if they are not empty
                solution_directories = [
                    f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/solutions"
                ]

                for directory in solution_directories:
                    if os.listdir(directory):
                        for file_name in os.listdir(directory):
                            file_path = os.path.join(directory, file_name)
                            os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"

    num_examples = 2

    final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

    for chain_length in range(1, 6):
        for task in augment_tasks:
            for i in range(n_graphs):
                for final_task in final_tasks:
                    #print(f"Generating chain cot prompt {i}")

                    examples = []
                    examples_strs = []

                    print(f'Chain length: {chain_length}, Task: {task}, Final task: {final_task}, Graph: {i}')
                    
                    for e in range(num_examples):
                        p = random.uniform(0, 1)
                        if task == "remove_node":
                            n = random.randint(6, 20)
                        else:
                            n = random.randint(5, 20)
                        graph = nx.erdos_renyi_graph(n, p)

                        def is_complete_graph(G):
                            n = len(G.nodes)
                            # A complete graph with n nodes has n*(n-1)/2 edges
                            expected_num_edges = n * (n - 1) // 2
                            actual_num_edges = len(G.edges)
                            return actual_num_edges == expected_num_edges

                        # Check if the graph is complete
                        is_complete = is_complete_graph(graph)
                        #print("Graph is a complete graph:", is_complete)

                        # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                        while (task == "remove_edge" and graph.number_of_edges() < 5):
                            p = random.uniform(0, 1)
                            graph = nx.erdos_renyi_graph(n, p)

                        # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
                        while (task == "add_edge" and graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                            p = random.uniform(0, 1)
                            graph = nx.erdos_renyi_graph(n, p)

                        # Convert graph to string
                        graph_str = str(nx.adjacency_matrix(graph).todense())

                        examples.append(graph)
                        examples_strs.append(graph_str)

                    #print("Examples: ", examples)

                    # Extract final graph from input file
                    input_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/"
                    input_filename = f"graph_{i}.txt"
                    
                    with open(os.path.join(input_dir, input_filename), "r") as input_file:
                        final_graph_str = input_file.read()

                    # Add a comma after every integer in final_graph_str
                    final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

                    #print("Final graph string: ", final_graph_str)

                    # Convert final_graph_str into a networkx graph
                    final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))
                    #print("Final graph: ", final_graph)
                    #return

                    # Convert graph to string
                    #graph_str = str(nx.adjacency_matrix(graph).todense())
                    final_graph_str = final_graph_str.replace(",", "") # remove extra commas
                    final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas

                    #init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
                    #end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
                    #end_count_prompt = "A: Final answer: The final answer is: "
                    #end_yes_no_prompt = "A: Final answer: The final answer is: "

                    #ablation_dir += "/"

                    # Build a giant prompt filled with examples

                    # Graph augmentation tasks
                    #cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_edge", examples, examples_strs)
                    #cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_edge", examples, examples_strs)
                    #cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_node", examples, examples_strs)
                    #cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_node", examples, examples_strs)
                    #return

                    init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
                    end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
                    end_count_prompt = "A: Final answer: The final answer is: "
                    end_yes_no_prompt = "A: Final answer: The final answer is: "

                    

                    
                    # Write graph to file
                    graph_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/input_graphs/graph_{i}.txt"
                    with open(graph_filename, "w") as graph_file:
                        graph_file.write(final_graph_str)

                    if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                        chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_task, chain_length, examples, examples_strs)
                    elif final_task == "print_adjacency_matrix":
                        chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, chain_length, examples, examples_strs)

                    #return

def generate_data_p(n_graphs, p, n):
    # Create directories if they don't exist
    ablation_dir = f"ablation_p/{str(p)}/{str(n)}"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_edge",
        f"data/{ablation_dir}/prompts/remove_edge",
        f"data/{ablation_dir}/prompts/add_node",
        f"data/{ablation_dir}/prompts/remove_node",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_edge",
        f"data/{ablation_dir}/solutions/remove_edge",
        f"data/{ablation_dir}/solutions/add_node",
        f"data/{ablation_dir}/solutions/remove_node",
    ]

    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")
        # Generate Erdos-Renyi graph that is not connected
        graph = nx.erdos_renyi_graph(n, p)

        def is_complete_graph(G):
            n = len(G.nodes)
            # A complete graph with n nodes has n*(n-1)/2 edges
            expected_num_edges = n * (n - 1) // 2
            actual_num_edges = len(G.edges)
            return actual_num_edges == expected_num_edges

        # Check if the graph is complete
        is_complete = is_complete_graph(graph)
        #print("Graph is a complete graph:", is_complete)

        # Ensure that the graph is not complete and has at least one edge
        j = 0
        while (graph.number_of_edges() == 0 and p != 0) or (is_complete_graph(graph) and p != 1.0):
            graph = nx.erdos_renyi_graph(n, p)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n}")
                graph_str = str(nx.adjacency_matrix(graph).todense())
                print("Graph: ", graph_str)
                sys.exit(1)

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/ablation_p/{str(p)}/{str(n)}/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
        end_count_prompt = "A: Final answer: The final answer is: "
        end_yes_no_prompt = "A: Final answer: The final answer is: "

        ablation_dir += "/"

        # Graph augmentation tasks
        if p != 1.0: # don't generate add_edge prompt if p=1.0
            add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        if p != 0: # don't generate remove_edge prompt if p=0
            remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)

        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)

def generate_data_n(n_graphs, n):
    # Create directories if they don't exist
    ablation_dir = f"ablation_n/{str(n)}"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_node",
        f"data/{ablation_dir}/prompts/remove_node",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_node",
        f"data/{ablation_dir}/solutions/remove_node",
    ]

    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")
        # Generate Erdos-Renyi graph that is not connected
        graph = nx.erdos_renyi_graph(n, 0.5)
        """

        def is_complete_graph(G):
            n = len(G.nodes)
            # A complete graph with n nodes has n*(n-1)/2 edges
            expected_num_edges = n * (n - 1) // 2
            actual_num_edges = len(G.edges)
            return actual_num_edges == expected_num_edges

        # Check if the graph is complete
        is_complete = is_complete_graph(graph)
        #print("Graph is a complete graph:", is_complete)

        # Ensure that the graph is not complete and has at least one edge
        j = 0
        while (graph.number_of_edges() == 0 and p != 0) or (is_complete_graph(graph) and p != 1.0):
            graph = nx.erdos_renyi_graph(n, p)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n}")
                graph_str = str(nx.adjacency_matrix(graph).todense())
                print("Graph: ", graph_str)
                sys.exit(1)
        """

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/ablation_n/{str(n)}/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
        end_count_prompt = "A: Final answer: The final answer is: "
        end_yes_no_prompt = "A: Final answer: The final answer is: "

        ablation_dir += "/"

        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)

def generate_data_d(n_graphs, n, ablationType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_d/{str(n)}"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/edge_count", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_edge",
        f"data/{ablation_dir}/prompts/remove_edge",
        f"data/{ablation_dir}/prompts/add_node",
        f"data/{ablation_dir}/prompts/remove_node",
        f"data/{ablation_dir}/prompts/edge_count",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/edge_count", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_edge",
        f"data/{ablation_dir}/solutions/remove_edge",
        f"data/{ablation_dir}/solutions/add_node",
        f"data/{ablation_dir}/solutions/remove_node",
        f"data/{ablation_dir}/solutions/edge_count",
    ]

    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")
        # Generate a directed Erdos-Renyi graph
        p = 0.5
        graph = nx.erdos_renyi_graph(n, p, directed=True)

        def is_complete_graph(G):
            n = len(G.nodes)
            # A complete graph with n nodes has n*(n-1)/2 edges
            expected_num_edges = n * (n - 1) // 2
            actual_num_edges = len(G.edges)
            return actual_num_edges == expected_num_edges

        # Check if the graph is complete
        is_complete = is_complete_graph(graph)
        #print("Graph is a complete graph:", is_complete)

        # Ensure that the graph is not complete and has at least one edge
        j = 0
        while (graph.number_of_edges() == 0 ) or (is_complete_graph(graph)):
            graph = nx.erdos_renyi_graph(n, p, directed=True)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n}")
                graph_str = str(nx.adjacency_matrix(graph).todense())
                print("Graph: ", graph_str)
                sys.exit(1)

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/ablation_d/{str(n)}/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of a directed graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
        end_count_prompt = "A: Final answer: The final answer is: "
        end_yes_no_prompt = "A: Final answer: The final answer is: "

        ablation_dir += "/"

        # Graph augmentation tasks
        #print("Generating add_edge prompt")
        #sys.exit(1)
        add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir, ablationType)
        remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir, ablationType)

        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)

        edge_count(graph, graph_str, init_prompt, end_count_prompt, i, ablation_dir)

def generate_data_few(n_graphs, ablationType, max_num_examples = 5):
    # Create directories if they don't exist
    ablation_dir = f"ablation_few"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    for n in range(1, max_num_examples+1):

        os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/add_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/remove_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/add_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/remove_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/node_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/edge_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/node_degree/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/edge_exists/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/connected_nodes/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/cycle/{n}", exist_ok=True)

        # Empty the directories if they are not empty
        prompt_directories = [
            f"data/{ablation_dir}/prompts/add_edge/{n}",
            f"data/{ablation_dir}/prompts/remove_edge/{n}",
            f"data/{ablation_dir}/prompts/add_node/{n}",
            f"data/{ablation_dir}/prompts/remove_node/{n}",
            f"data/{ablation_dir}/prompts/node_count/{n}",
            f"data/{ablation_dir}/prompts/edge_count/{n}",
            f"data/{ablation_dir}/prompts/node_degree/{n}",
            f"data/{ablation_dir}/prompts/edge_exists/{n}",
            f"data/{ablation_dir}/prompts/connected_nodes/{n}",
            f"data/{ablation_dir}/prompts/cycle/{n}",
        ]

        for directory in prompt_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

        os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/add_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/remove_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/add_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/remove_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/node_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/edge_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/node_degree/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/edge_exists/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/connected_nodes/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/cycle/{n}", exist_ok=True)

        # Empty the directories if they are not empty
        solution_directories = [
            f"data/{ablation_dir}/solutions/add_edge/{n}",
            f"data/{ablation_dir}/solutions/remove_edge/{n}",
            f"data/{ablation_dir}/solutions/add_node/{n}",
            f"data/{ablation_dir}/solutions/remove_node/{n}",
            f"data/{ablation_dir}/solutions/node_count/{n}",
            f"data/{ablation_dir}/solutions/edge_count/{n}",
            f"data/{ablation_dir}/solutions/node_degree/{n}",
            f"data/{ablation_dir}/solutions/edge_exists/{n}",
            f"data/{ablation_dir}/solutions/connected_nodes/{n}",
            f"data/{ablation_dir}/solutions/cycle/{n}",
        ]

        for directory in solution_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")

        examples = []
        examples_strs = []

        # Generate CoT examples
        for e in range(max_num_examples):
            p = random.uniform(0, 1)
            n_nodes = random.randint(5, 20)
            graph = nx.erdos_renyi_graph(n_nodes, p)

            def is_complete_graph(G):
                n = len(G.nodes)
                # A complete graph with n nodes has n*(n-1)/2 edges
                expected_num_edges = n * (n - 1) // 2
                actual_num_edges = len(G.edges)
                return actual_num_edges == expected_num_edges

            # Check if the graph is complete
            is_complete = is_complete_graph(graph)
            #print("Graph is a complete graph:", is_complete)

            # Ensure that the graph is not complete and has at least one edge
            j = 0
            while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
                p = random.uniform(0, 1)
                n_nodes = random.randint(5, 20)
                graph = nx.erdos_renyi_graph(n_nodes, p)
                j += 1
                if j > 100:
                    print("Could not generate a valid graph after 100 attempts")
                    print(f"p: {p}, n: {n}")
                    graph_str = str(nx.adjacency_matrix(graph).todense())
                    print("Graph: ", graph_str)
                    sys.exit(1)

            # Convert graph to string
            graph_str = str(nx.adjacency_matrix(graph).todense())

            examples.append(graph)
            examples_strs.append(graph_str)

            # Extract final graph from input file
            input_dir = "data/input_graphs"
            input_filename = f"graph_{i}.txt"
            
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                final_graph_str = input_file.read()

            # Add a comma after every integer in final_graph_str
            final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

            # Convert final_graph_str into a networkx graph
            final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))
            #print("Final graph: ", final_graph)
            #return

            # Convert graph to string
            #graph_str = str(nx.adjacency_matrix(graph).todense())
            final_graph_str = final_graph_str.replace(",", "") # remove extra commas
            final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas

            # Write graph to file
            graph_filename = f"data/ablation_few/input_graphs/graph_{i}.txt"
            with open(graph_filename, "w") as graph_file:
                graph_file.write(final_graph_str)

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: Final answer: The final answer is: "
            end_yes_no_prompt = "A: Final answer: The final answer is: "

            ablation_dir += "/"

            # Graph property tasks
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "node_count", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "edge_count", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "node_degree", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_yes_no_prompt, i, "edge_exists", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "connected_nodes", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_yes_no_prompt, i, "cycle", examples, examples_strs, True)

            # Graph augmentation tasks
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_edge", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_edge", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_node", examples, examples_strs, True)
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_node", examples, examples_strs, True)



def generate_data_cot(n_graphs, ablationType, max_num_examples = 5):
    # Create directories if they don't exist
    ablation_dir = f"ablation_cot"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    for n in range(1, max_num_examples+1):

        os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/add_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/remove_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/add_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/remove_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/node_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/edge_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/node_degree/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/edge_exists/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/connected_nodes/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/cycle/{n}", exist_ok=True)

        # Empty the directories if they are not empty
        prompt_directories = [
            f"data/{ablation_dir}/prompts/add_edge/{n}",
            f"data/{ablation_dir}/prompts/remove_edge/{n}",
            f"data/{ablation_dir}/prompts/add_node/{n}",
            f"data/{ablation_dir}/prompts/remove_node/{n}",
            f"data/{ablation_dir}/prompts/node_count/{n}",
            f"data/{ablation_dir}/prompts/edge_count/{n}",
            f"data/{ablation_dir}/prompts/node_degree/{n}",
            f"data/{ablation_dir}/prompts/edge_exists/{n}",
            f"data/{ablation_dir}/prompts/connected_nodes/{n}",
            f"data/{ablation_dir}/prompts/cycle/{n}",
        ]

        for directory in prompt_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

        os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/add_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/remove_edge/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/add_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/remove_node/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/node_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/edge_count/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/node_degree/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/edge_exists/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/connected_nodes/{n}", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/cycle/{n}", exist_ok=True)

        # Empty the directories if they are not empty
        solution_directories = [
            f"data/{ablation_dir}/solutions/add_edge/{n}",
            f"data/{ablation_dir}/solutions/remove_edge/{n}",
            f"data/{ablation_dir}/solutions/add_node/{n}",
            f"data/{ablation_dir}/solutions/remove_node/{n}",
            f"data/{ablation_dir}/solutions/node_count/{n}",
            f"data/{ablation_dir}/solutions/edge_count/{n}",
            f"data/{ablation_dir}/solutions/node_degree/{n}",
            f"data/{ablation_dir}/solutions/edge_exists/{n}",
            f"data/{ablation_dir}/solutions/connected_nodes/{n}",
            f"data/{ablation_dir}/solutions/cycle/{n}",
        ]

        for directory in solution_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")

        examples = []
        examples_strs = []

        # Generate CoT examples
        for e in range(max_num_examples):
            p = random.uniform(0, 1)
            n_nodes = random.randint(5, 20)
            graph = nx.erdos_renyi_graph(n_nodes, p)

            def is_complete_graph(G):
                n = len(G.nodes)
                # A complete graph with n nodes has n*(n-1)/2 edges
                expected_num_edges = n * (n - 1) // 2
                actual_num_edges = len(G.edges)
                return actual_num_edges == expected_num_edges

            # Check if the graph is complete
            is_complete = is_complete_graph(graph)
            #print("Graph is a complete graph:", is_complete)

            # Ensure that the graph is not complete and has at least one edge
            j = 0
            while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
                p = random.uniform(0, 1)
                n_nodes = random.randint(5, 20)
                graph = nx.erdos_renyi_graph(n_nodes, p)
                j += 1
                if j > 100:
                    print("Could not generate a valid graph after 100 attempts")
                    print(f"p: {p}, n: {n}")
                    graph_str = str(nx.adjacency_matrix(graph).todense())
                    print("Graph: ", graph_str)
                    sys.exit(1)

            # Convert graph to string
            graph_str = str(nx.adjacency_matrix(graph).todense())

            examples.append(graph)
            examples_strs.append(graph_str)

            # Extract final graph from input file
            input_dir = "data/input_graphs"
            input_filename = f"graph_{i}.txt"
            
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                final_graph_str = input_file.read()

            # Add a comma after every integer in final_graph_str
            final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

            # Convert final_graph_str into a networkx graph
            final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))
            #print("Final graph: ", final_graph)
            #return

            # Convert graph to string
            #graph_str = str(nx.adjacency_matrix(graph).todense())
            final_graph_str = final_graph_str.replace(",", "") # remove extra commas
            final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas

            # Write graph to file
            graph_filename = f"data/ablation_cot/input_graphs/graph_{i}.txt"
            with open(graph_filename, "w") as graph_file:
                graph_file.write(final_graph_str)

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: Final answer: The final answer is: "
            end_yes_no_prompt = "A: Final answer: The final answer is: "

            ablation_dir += "/"

            # Graph property tasks
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "node_count", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "edge_count", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "node_degree", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_yes_no_prompt, i, "edge_exists", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "connected_nodes", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_yes_no_prompt, i, "cycle", examples, examples_strs)

            # Graph augmentation tasks
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_edge", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_edge", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "add_node", examples, examples_strs)
            cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, "remove_node", examples, examples_strs)

def generate_data_graph_type(n_graphs, ablationType, graphType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_graph_type_{graphType}"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_edge",
        f"data/{ablation_dir}/prompts/remove_edge",
        f"data/{ablation_dir}/prompts/add_node",
        f"data/{ablation_dir}/prompts/remove_node",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_node", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_edge",
        f"data/{ablation_dir}/solutions/remove_edge",
        f"data/{ablation_dir}/solutions/add_node",
        f"data/{ablation_dir}/solutions/remove_node",
    ]

    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    num_examples = 2

    for i in range(n_graphs):
        print(f"Generating graph {i}")

        p = random.uniform(0, 1)
        n_nodes = random.randint(5, 20)
        if graphType == "erdos_renyi":
            return
        elif graphType == "barabasi_albert":
            m = random.randint(1, 4)
            graph = nx.barabasi_albert_graph(n_nodes, m)
        elif graphType == "watts_strogatz":
            k = random.randint(1, 5)
            p = random.uniform(0, 1)
            graph = nx.watts_strogatz_graph(n_nodes, k, p)
        elif graphType == "star":
            graph = nx.star_graph(n_nodes)
        elif graphType == "path":
            graph = nx.path_graph(n_nodes)
        elif graphType == "stochastic_block":
            n_blocks = random.randint(2, 5)
            p = random.uniform(0, 1)
            graph = nx.stochastic_block_model([n_nodes // n_blocks] * n_blocks, [[p] * n_blocks] * n_blocks)
        elif graphType == "scale_free":
            graph = nx.scale_free_graph(n_nodes)

        def is_complete_graph(G):
            n = len(G.nodes)
            # A complete graph with n nodes has n*(n-1)/2 edges
            expected_num_edges = n * (n - 1) // 2
            actual_num_edges = len(G.edges)
            return actual_num_edges == expected_num_edges

        # Check if the graph is complete
        is_complete = is_complete_graph(graph)
        #print("Graph is a complete graph:", is_complete)

        # Ensure that the graph is not complete and has at least one edge
        j = 0
        while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
            graph = nx.erdos_renyi_graph(n_nodes, p)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n}")
                graph_str = str(nx.adjacency_matrix(graph).todense())
                print("Graph: ", graph_str)
                sys.exit(1)

        # Convert graph to string
        graph_str = str(nx.adjacency_matrix(graph).todense())

        # Write graph to file
        graph_filename = f"data/ablation_graph_type_{graphType}/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
        end_count_prompt = "A: Final answer: The final answer is: "
        end_yes_no_prompt = "A: Final answer: The final answer is: "

        ablation_dir += "/"

        # Build a giant prompt filled with examples

        # Graph augmentation tasks
        add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir, ablationType)
        remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir, ablationType)
        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablation_dir)
        #return

def generate_data_encoding(n_graphs, ablationType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_encoding"
    

    encoding_types = ["incidence", "coauthorship", "friendship", "social_network"]

    for encoding_type in encoding_types:
        os.makedirs(f"data/{ablation_dir}/input_graphs/{encoding_type}", exist_ok=True)

        # Empty the directory if it's not empty
        if os.listdir(f"data/{ablation_dir}/input_graphs/{encoding_type}"):
            for file_name in os.listdir(f"data/{ablation_dir}/input_graphs/{encoding_type}"):
                file_path = os.path.join(f"data/{ablation_dir}/input_graphs/{encoding_type}", file_name)
                os.remove(file_path)

        os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/add_edge", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/remove_edge", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/add_node", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/remove_node", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/node_count", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/edge_count", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/node_degree", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/edge_exists", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/connected_nodes", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/prompts/{encoding_type}/cycle", exist_ok=True)

        # Empty the directories if they are not empty
        prompt_directories = [
            f"data/{ablation_dir}/prompts/{encoding_type}/add_edge",
            f"data/{ablation_dir}/prompts/{encoding_type}/remove_edge",
            f"data/{ablation_dir}/prompts/{encoding_type}/add_node",
            f"data/{ablation_dir}/prompts/{encoding_type}/remove_node",
            f"data/{ablation_dir}/prompts/{encoding_type}/node_count",
            f"data/{ablation_dir}/prompts/{encoding_type}/edge_count",
            f"data/{ablation_dir}/prompts/{encoding_type}/node_degree",
            f"data/{ablation_dir}/prompts/{encoding_type}/edge_exists",
            f"data/{ablation_dir}/prompts/{encoding_type}/connected_nodes",
            f"data/{ablation_dir}/prompts/{encoding_type}/cycle",
        ]

        for directory in prompt_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

        os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/add_edge", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/remove_edge", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/add_node", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/remove_node", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/node_count", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/edge_count", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/node_degree", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/edge_exists", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/connected_nodes", exist_ok=True)
        os.makedirs(f"data/{ablation_dir}/solutions/{encoding_type}/cycle", exist_ok=True)

        # Empty the directories if they are not empty
        solution_directories = [
            f"data/{ablation_dir}/solutions/{encoding_type}/add_edge",
            f"data/{ablation_dir}/solutions/{encoding_type}/remove_edge",
            f"data/{ablation_dir}/solutions/{encoding_type}/add_node",
            f"data/{ablation_dir}/solutions/{encoding_type}/remove_node",
            f"data/{ablation_dir}/solutions/{encoding_type}/node_count",
            f"data/{ablation_dir}/solutions/{encoding_type}/edge_count",
            f"data/{ablation_dir}/solutions/{encoding_type}/node_degree",
            f"data/{ablation_dir}/solutions/{encoding_type}/edge_exists",
            f"data/{ablation_dir}/solutions/{encoding_type}/connected_nodes",
            f"data/{ablation_dir}/solutions/{encoding_type}/cycle",
        ]

        for directory in solution_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")

        # Extract final graph from input file
        input_dir = "data/input_graphs"
        input_filename = f"graph_{i}.txt"
        
        with open(os.path.join(input_dir, input_filename), "r") as input_file:
            final_graph_str = input_file.read()

        # Add a comma after every integer in final_graph_str
        final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

        # Convert final_graph_str into a networkx graph
        final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))
        #print("Final graph: ", final_graph)
        #return

        # Convert graph to string
        #graph_str = str(nx.adjacency_matrix(graph).todense())
        final_graph_str = final_graph_str.replace(",", "") # remove extra commas
        final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas

        # create a list of 20 strings of common names
        names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher"]

        for graph_type in encoding_types:

            encoding_dict = {}

            # get the list of nodes of final_graph
            nodes = list(final_graph.nodes)

            # enumerate over names
            for n, name in enumerate(names):
                if graph_type in ["coauthorship", "friendship", "social_network"]:
                    encoding_dict[n] = name
                else:
                    encoding_dict[n] = n

            # construct encoding_graph_str
            encoding_graph_str = graph_to_string_encoder(final_graph, graph_type, encoding_dict)        

            # Write graph to file
            graph_filename = f"data/ablation_encoding/input_graphs/{graph_type}/graph_{i}.txt"
            with open(graph_filename, "w") as graph_file:
                graph_file.write(encoding_graph_str)

            # construct init_prompt
            if graph_type == "incidence": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
                # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
                nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
                init_prompt = f"G describes a graph among {nodes_str}.\nIn this graph:\n"
                end_mod_prompt = "A: Final answer: The resulting graph is: "
            elif graph_type == "coauthorship":
                nodes_str = ', '.join([encoding_dict[n] for n in nodes[:-2]]) + ', ' + encoding_dict[nodes[-2]] + ', and ' + encoding_dict[nodes[-1]]
                init_prompt = f"G describes a co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
                end_mod_prompt = "A: Final answer: The resulting co-authorship graph is: "
            elif graph_type == "friendship":
                nodes_str = ', '.join([encoding_dict[n] for n in nodes[:-2]]) + ', ' + encoding_dict[nodes[-2]] + ', and ' + encoding_dict[nodes[-1]]
                init_prompt = f"G describes a friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
                end_mod_prompt = "A: Final answer: The resulting friendship graph is: "
            elif graph_type == "social_network":
                nodes_str = ', '.join([encoding_dict[n] for n in nodes[:-2]]) + ', ' + encoding_dict[nodes[-2]] + ', and ' + encoding_dict[nodes[-1]]
                init_prompt = f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
                end_mod_prompt = "A: Final answer: The resulting social network graph is: "

            #init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            #end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: Final answer: The final answer is: "
            end_yes_no_prompt = "A: Final answer: The final answer is: "

            ablation_dir += "/"

            # Graph augmentation tasks
            add_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            remove_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            add_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            remove_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)

            # Graph property tasks
            node_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            edge_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            node_degree(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            edge_exists(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            connected_nodes(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)
            cycle(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False, graph_type=graph_type, encoding_dict=encoding_dict)

        #return

def generate_data_no_force(n_graphs, ablationType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_no_force"
    
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/remove_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/node_count", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/edge_count", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/node_degree", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/edge_exists", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/connected_nodes", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/cycle", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_edge",
        f"data/{ablation_dir}/prompts/remove_edge",
        f"data/{ablation_dir}/prompts/add_node",
        f"data/{ablation_dir}/prompts/remove_node",
        f"data/{ablation_dir}/prompts/node_count",
        f"data/{ablation_dir}/prompts/edge_count",
        f"data/{ablation_dir}/prompts/node_degree",
        f"data/{ablation_dir}/prompts/edge_exists",
        f"data/{ablation_dir}/prompts/connected_nodes",
        f"data/{ablation_dir}/prompts/cycle",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_edge", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/remove_node", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/node_count", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/edge_count", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/node_degree", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/edge_exists", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/connected_nodes", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/cycle", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_edge",
        f"data/{ablation_dir}/solutions/remove_edge",
        f"data/{ablation_dir}/solutions/add_node",
        f"data/{ablation_dir}/solutions/remove_node",
        f"data/{ablation_dir}/solutions/node_count",
        f"data/{ablation_dir}/solutions/edge_count",
        f"data/{ablation_dir}/solutions/node_degree",
        f"data/{ablation_dir}/solutions/edge_exists",
        f"data/{ablation_dir}/solutions/connected_nodes",
        f"data/{ablation_dir}/solutions/cycle",
    ]

    for directory in solution_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]

    for i in range(n_graphs):
        print(f"Generating graph {i}")

        # Extract final graph from input file
        input_dir = "data/input_graphs"
        input_filename = f"graph_{i}.txt"
        
        with open(os.path.join(input_dir, input_filename), "r") as input_file:
            final_graph_str = input_file.read()

        # Add a comma after every integer in final_graph_str
        final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

        # Convert final_graph_str into a networkx graph
        final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))
        #print("Final graph: ", final_graph)
        #return

        # Convert graph to string
        #graph_str = str(nx.adjacency_matrix(graph).todense())
        final_graph_str = final_graph_str.replace(",", "") # remove extra commas
        encoding_graph_str = final_graph_str.replace("  ", " ") # remove extra commas       

        # Write graph to file
        graph_filename = f"data/ablation_no_force/input_graphs/graph_{i}.txt"
        with open(graph_filename, "w") as graph_file:
            graph_file.write(encoding_graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: "
        end_count_prompt = "A: "
        end_yes_no_prompt = "A: "

        ablation_dir += "/"

        # Graph augmentation tasks
        add_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        remove_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        add_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        remove_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)

        # Graph property tasks
        node_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        edge_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        node_degree(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        edge_exists(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        connected_nodes(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)
        cycle(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, cot=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, help="number of graphs to generate")
    #parser.add_argument("--n_nodes", type=int, help="number of nodes in the graph")
    #parser.add_argument("--p", type=float, help="probability of an edge between any two nodes")
    #parser.add_argument("--prompt_type", type=str, default="add_edge", help="type of prompt")
    parser.add_argument("--base", type=bool, required=True, help="whether to generate graphs for base tasks (property or modification)")
    parser.add_argument("--chain", type=bool, required=True, help="whether to generate chain prompts")
    parser.add_argument("--ablation", type=bool, required=True, help="whether to generate graphs for ablation studies")
    parser.add_argument("--ablationType", choices=["p", "n", "d", "few", "cot", "cot_chain", "encoding", "graph_type", "no_force"], help="what type of graphs to generate for ablation studies")
    
    args = parser.parse_args()

    n_graphs = args.n_graphs
    base = args.base
    gen_chain = args.chain
    ablation = args.ablation
    ablationType = args.ablationType
    gen_chain = args.chain

    base = False
    gen_chain = False
    ablation = True

    if base:
        print("Generating base prompts")
        #generate_data(n_graphs)
    elif gen_chain:
        print("Generating chain prompts")
        generate_chains_same(500)
    elif ablation:
        print("Generating ablation prompts")
        if ablationType == "p":
            print("Density Ablation")
            for p in np.linspace(0, 1.0, 11):
                for n in range(5, 11):
                    print(f"Generating graphs for p={p} and n={n}")
                    generate_data_p(n_graphs, p, n)
        elif ablationType == "n":
            print("Size Ablation")
            for n in range(5, 21):
                print(f"Generating graphs for n={n}")
                #generate_data_n(n_graphs, n)
        elif ablationType == "d": # directed
            print("Directed Ablation")
            for n in range(5, 21):
                print(f"Generating graphs for n={n}")
                generate_data_d(n_graphs, n, ablationType)
        elif ablationType == "few": # few-shot
            generate_data_few(n_graphs, ablationType)
        elif ablationType == "cot": # CoT
            generate_data_cot(n_graphs, ablationType)  
        elif ablationType == "cot_chain":
            generate_chains_same_cot(n_graphs)
        elif ablationType == "encoding":
            generate_data_encoding(100, ablationType)
        elif ablationType == "graph_type":
            for graphType in ["barabasi_albert", "star", "path", "stochastic_block"]:
                generate_data_graph_type(n_graphs, ablationType, graphType)
        elif ablationType == "no_force":
            generate_data_no_force(n_graphs, ablationType)
        else:
            print("Please specify what type of ablation study to run")
            sys.exit(1)
    else:
        print("Please specify what type of prompts to generate")
        sys.exit(1)