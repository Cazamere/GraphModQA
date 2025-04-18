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
        # Create a random seed
        random.seed(i)
        # Generate Erdos-Renyi graph that is not connected
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

        # Ensure that the graph is not complete and has at least one edge
        j = 0
        while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
            p = random.uniform(0, 1)
            n_nodes = random.randint(5, 20)
            graph = nx.erdos_renyi_graph(n_nodes, p)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n_nodes}")
                graph_str = str(nx.adjacency_matrix(graph).todense())
                print("Graph: ", graph_str)
                sys.exit(1)

        # Write graph to file using write_graphml
        graph_original_filename = f"data/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_original_filename)

        graph_info_filename = f"data/input_graphs/{i}.txt"
        with open(graph_info_filename, "w") as graph_file:
            graph_file.write(f"n: {n_nodes}, p: {p}")

        # Convert graph to string
        graph_str = graph_to_string_encoder(graph)

        # Write graph to file
        #graph_filename = f"data/input_graphs/graph_{i}.txt"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: "
        #end_count_prompt = ", and surround your final answer in parentheses, like this: (answer). \nA:" # kind of works
        end_count_prompt = "A: "
        end_yes_no_prompt = "A: " #TODO: something like "present answer as _"
        
        # Graph augmentation tasks
        add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablationType = "no_force")
        remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablationType = "no_force")
        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablationType = "no_force")
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_matrix_prompt, i, False, ablationType = "no_force")
        node_count(graph, graph_str, init_prompt, end_count_prompt, i, ablationType = "no_force")
        edge_count(graph, graph_str, init_prompt, end_count_prompt, i, ablationType = "no_force")
        node_degree(graph, graph_str, init_prompt, end_count_prompt, i, ablationType = "no_force")
        edge_exists(graph, graph_str, init_prompt, end_yes_no_prompt, i, ablationType = "no_force")
        connected_nodes(graph, graph_str, init_prompt, end_count_prompt, i, ablationType = "no_force")
        cycle(graph, graph_str, init_prompt, end_yes_no_prompt, i, ablationType = "no_force")

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

            input_filename = f"{i}.graphml"

            # Read input graph
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                graph = nx.read_graphml(input_file)
                graph_str = graph_to_string_encoder(graph)

            # add commas in between the numbers in graph_str
            #graph_str = graph_str.replace(" ", ", ")

            #graph = nx.from_numpy_array(np.array(eval(graph_str)))

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: "
            end_count_prompt = "A: "
            end_yes_no_prompt = "A: "

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

    for i in range(n_graphs):

        print(f"Generating chain prompt {i}")
        p = random.uniform(0, 1)
        if task == "remove_node":
            n = random.randint(6, 15)
        else:
            n = random.randint(5, 15)
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
        graph_str = graph_to_string_encoder(graph)

        #for chain_length in range(1, 6):
        max_chain_length = 5
        for task in augment_tasks:
            
            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_matrix_prompt = "A: "
            end_count_prompt = "A: "
            end_yes_no_prompt = "A: "

            final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

            for final_task in final_tasks:
                # Write graph to file
                #graph_filename = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/{i}.graphml"
                #with open(graph_filename, "w") as graph_file:
                #    graph_file.write(graph_str)
                #nx.write_graphml(graph, graph_filename)
                if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                    chain_same(graph, graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_task, max_chain_length)
                elif final_task == "print_adjacency_matrix":
                    chain_same(graph, graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, max_chain_length)

def generate_chains_same_few(n_graphs, max_num_examples = 5):
    for final_task in ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]:
        for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
            for chain_length in range(1, 6):

                # Create directories if they don't exist
                ablation_dir = "chains_same_few"
                os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", exist_ok=True)

                # Empty the directory if it's not empty
                if os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                    for file_name in os.listdir(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs"):
                        file_path = os.path.join(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/input_graphs", file_name)
                        os.remove(file_path)

                for n in range(1, max_num_examples+1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts", exist_ok=True)

                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts"
                    ]

                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions"
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"


    final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

    for i in range(n_graphs):
        for chain_length in range(1, 6):
            for task in augment_tasks:
                #print(f"Generating chain cot prompt {i}")

                examples = []
                examples_strs = []
                
                for e in range(max_num_examples):
                    p = random.uniform(0, 1)
                    if task == "remove_node":
                        n = random.randint(6, 15)
                    else:
                        n = random.randint(5, 15)
                    example_graph = nx.erdos_renyi_graph(n, p)

                    def is_complete_graph(G):
                        n = len(G.nodes)
                        # A complete graph with n nodes has n*(n-1)/2 edges
                        expected_num_edges = n * (n - 1) // 2
                        actual_num_edges = len(G.edges)
                        return actual_num_edges == expected_num_edges

                    # Check if the graph is complete
                    is_complete = is_complete_graph(example_graph)
                    #print("Graph is a complete graph:", is_complete)

                    # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                    while (task == "remove_edge" and example_graph.number_of_edges() < 5):
                        p = random.uniform(0, 1)
                        example_graph = nx.erdos_renyi_graph(n, p)

                    # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
                    while (task == "add_edge" and example_graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                        p = random.uniform(0, 1)
                        example_graph = nx.erdos_renyi_graph(n, p)

                    # Convert graph to string
                    example_graph_str = graph_to_string_encoder(example_graph)

                    examples.append(example_graph)
                    examples_strs.append(example_graph_str)

                    #print("Examples: ", examples)

                # Extract final graph from input file
                input_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/"
                input_filename = f"{i}.graphml"

                # Read input graph
                with open(os.path.join(input_dir, input_filename), "r") as input_file:
                    final_graph = nx.read_graphml(input_file)
                    final_graph_str = graph_to_string_encoder(final_graph)

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
                end_prompt = "A: "
                #end_prompt = "A: "
                #end_yes_no_prompt = "A: "

                # Write graph to file
                
                #print(f"Writing graph to file {graph_filename}")
                #with open(graph_filename, "w") as graph_file:
                #    graph_file.write(final_graph_str)

                for final_task in final_tasks:
                    print(f'Chain length: {chain_length}, Task: {task}, Final task: {final_task}, Graph: {i}')
                    chain_same_few(final_graph, final_graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs)

                    graph_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/input_graphs/{i}.graphml"
                    nx.write_graphml(final_graph, graph_filename)

                    """
                    if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                        chain_same_few(final_graph, final_graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_task, chain_length, examples, examples_strs)
                    elif final_task == "print_adjacency_matrix":
                        chain_same_few(final_graph, final_graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, chain_length, examples, examples_strs)
                    else:
                        print("Final task not recognized")
                        pass
                    """

def generate_chains_same_cot(n_graphs, max_num_examples = 5):
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

                for n in range(1, max_num_examples+1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts", exist_ok=True)

                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts"
                    ]

                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions"
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"


    final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

    for i in range(n_graphs):
        for chain_length in range(1, 6):
            for task in augment_tasks:
                #print(f"Generating chain cot prompt {i}")

                examples = []
                examples_strs = []
                
                for e in range(max_num_examples):
                    p = random.uniform(0, 1)
                    if task == "remove_node":
                        n = random.randint(6, 15)
                    else:
                        n = random.randint(5, 15)
                    example_graph = nx.erdos_renyi_graph(n, p)

                    def is_complete_graph(G):
                        n = len(G.nodes)
                        # A complete graph with n nodes has n*(n-1)/2 edges
                        expected_num_edges = n * (n - 1) // 2
                        actual_num_edges = len(G.edges)
                        return actual_num_edges == expected_num_edges

                    # Check if the graph is complete
                    is_complete = is_complete_graph(example_graph)
                    #print("Graph is a complete graph:", is_complete)

                    # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                    while (task == "remove_edge" and example_graph.number_of_edges() < 5):
                        p = random.uniform(0, 1)
                        example_graph = nx.erdos_renyi_graph(n, p)

                    # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
                    while (task == "add_edge" and example_graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                        p = random.uniform(0, 1)
                        example_graph = nx.erdos_renyi_graph(n, p)

                    # Convert graph to string
                    example_graph_str = graph_to_string_encoder(example_graph)

                    examples.append(example_graph)
                    examples_strs.append(example_graph_str)

                    #print("Examples: ", examples)

                # Extract final graph from input file
                input_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/"
                input_filename = f"{i}.graphml"

                # Read input graph
                with open(os.path.join(input_dir, input_filename), "r") as input_file:
                    final_graph = nx.read_graphml(input_file)
                    final_graph_str = graph_to_string_encoder(final_graph)

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
                end_prompt = "A: "
                #end_prompt = "A: "
                #end_yes_no_prompt = "A: "

                # Write graph to file
                
                #print(f"Writing graph to file {graph_filename}")
                #with open(graph_filename, "w") as graph_file:
                #    graph_file.write(final_graph_str)

                for final_task in final_tasks:
                    print(f'Chain length: {chain_length}, Task: {task}, Final task: {final_task}, Graph: {i}')
                    chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs)

                    


"""
def generate_chains_same_cot(n_graphs, max_num_examples = 5):
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

                for n in range(1, max_num_examples+1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts", exist_ok=True)

                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/prompts"
                    ]

                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                        f"data/{ablation_dir}/{final_task}/{task}/{chain_length}/{n}/solutions"
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes"]
    input_dir = "data/input_graphs"


    final_tasks = ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]

    for chain_length in range(1, 6):
        for task in augment_tasks:
            for i in range(n_graphs):
                for final_task in final_tasks:
                    #print(f"Generating chain cot prompt {i}")

                    examples = []
                    examples_strs = []

                    print(f'Chain length: {chain_length}, Task: {task}, Final task: {final_task}, Graph: {i}')
                    
                    for e in range(max_num_examples):
                        p = random.uniform(0, 1)
                        if task == "remove_node":
                            n = random.randint(6, 15)
                        else:
                            n = random.randint(5, 15)
                        example_graph = nx.erdos_renyi_graph(n, p)

                        def is_complete_graph(G):
                            n = len(G.nodes)
                            # A complete graph with n nodes has n*(n-1)/2 edges
                            expected_num_edges = n * (n - 1) // 2
                            actual_num_edges = len(G.edges)
                            return actual_num_edges == expected_num_edges

                        # Check if the graph is complete
                        is_complete = is_complete_graph(example_graph)
                        #print("Graph is a complete graph:", is_complete)

                        # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                        while (task == "remove_edge" and example_graph.number_of_edges() < 5):
                            p = random.uniform(0, 1)
                            example_graph = nx.erdos_renyi_graph(n, p)

                        # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
                        while (task == "add_edge" and example_graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                            p = random.uniform(0, 1)
                            example_graph = nx.erdos_renyi_graph(n, p)

                        # Convert graph to string
                        example_graph_str = graph_to_string_encoder(example_graph)

                        examples.append(example_graph)
                        examples_strs.append(example_graph_str)

                        #print("Examples: ", examples)

                        # Extract final graph from input file
                        input_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/"
                        input_filename = f"{i}.graphml"

                        with open(os.path.join(input_dir, input_filename), "r") as input_file:
                            final_graph = nx.read_graphml(input_file)
                            final_graph_str = graph_to_string_encoder(final_graph)
                        

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
                        end_prompt = "A: "
                        #end_count_prompt = "A: "
                        #end_yes_no_prompt = "A: "

                        # Write graph to file
                        graph_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/input_graphs/{i}.graphml"
                        #with open(graph_filename, "w") as graph_file:
                        #    graph_file.write(final_graph_str)

                        nx.write_graphml(final_graph, graph_filename)

                        #if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                        #    chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_task, chain_length, examples, examples_strs)
                        #elif final_task == "print_adjacency_matrix":
                        #    chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, chain_length, examples, examples_strs)
                        chain_same_cot(final_graph, final_graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs)
"""

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
        graph_str = graph_to_string_encoder(graph)

        # Write graph to file
        graph_filename = f"data/ablation_p/{str(p)}/{str(n)}/input_graphs/{i}.graphml"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)
        nx.write_graphml(graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_prompt = "A: "
        #end_count_prompt = "A: "
        #end_yes_no_prompt = "A: "

        ablation_dir += "/"

        # Graph augmentation tasks
        if p != 1.0: # don't generate add_edge prompt if p=1.0
            add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)
        if p != 0: # don't generate remove_edge prompt if p=0
            remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)

        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)

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
        graph_str = graph_to_string_encoder(graph)

        # Write graph to file
        graph_filename = f"data/ablation_n/{str(n)}/input_graphs/{i}.graphml"
        #print(f"Writing graph to file {graph_filename}")
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)
        nx.write_graphml(graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_prompt = "A: "
        #end_count_prompt = "A: "
        #end_yes_no_prompt = "A: "

        ablation_dir += "/"

        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)

        #return

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
        while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
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
        graph_filename = f"data/ablation_d/{str(n)}/input_graphs/{i}.graphml"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)
        nx.write_graphml(graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of a directed graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: "
        end_count_prompt = "A: "
        end_yes_no_prompt = "A: "

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

        #if i < 227:
        #    continue

        examples = []
        examples_strs = []
        #random.seed(i)

        # Generate CoT examples
        for e in range(max_num_examples):
            p = random.uniform(0, 1)
            n_nodes = random.randint(5, 20)
            example_graph = nx.erdos_renyi_graph(n_nodes, p)

            def is_complete_graph(G):
                n = len(G.nodes)
                # A complete graph with n nodes has n*(n-1)/2 edges
                expected_num_edges = n * (n - 1) // 2
                actual_num_edges = len(G.edges)
                return actual_num_edges == expected_num_edges

            # Check if the graph is complete
            is_complete = is_complete_graph(example_graph)
            #print("Graph is a complete graph:", is_complete)

            # Ensure that the graph is not complete and has at least one edge
            j = 0
            while (example_graph.number_of_edges() == 0) or (is_complete_graph(example_graph)):
                p = random.uniform(0, 1)
                n_nodes = random.randint(5, 20)
                example_graph = nx.erdos_renyi_graph(n_nodes, p)
                j += 1
                if j > 100:
                    print("Could not generate a valid graph after 100 attempts")
                    print(f"p: {p}, n: {n}")
                    example_graph_str = graph_to_string_encoder(example_graph)
                    print("Graph: ", example_graph_str)
                    sys.exit(1)

            # Convert graph to string
            example_graph_str = graph_to_string_encoder(example_graph)

            examples.append(example_graph)
            examples_strs.append(example_graph_str)

            # Extract final graph from input file
            input_dir = "data/input_graphs"
            input_filename = f"{i}.graphml"
            
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                final_graph = nx.read_graphml(input_file)
                final_graph_str = graph_to_string_encoder(final_graph)

            """

            print("Final graph string extracted from base file: ", final_graph_str)

            # Add a comma after every integer in final_graph_str
            final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

            def adjacency_matrix_from_string(adj_matrix_str):
                #print("Adjacency matrix string: ", adj_matrix_str)
                adj_matrix_str = adj_matrix_str.replace("[[", "[")
                adj_matrix_str = adj_matrix_str.replace("]]", "]")
                #print("Adjacency matrix string: ", adj_matrix_str)
                adj_matrix_str = adj_matrix_str.replace(",", "")
                #print("Adjacency matrix string: ", adj_matrix_str)
                adj_matrix_str = adj_matrix_str.replace("  ", " ")
                adj_matrix_str = adj_matrix_str.replace(" [", "[")
                #print("Adjacency matrix string: ", adj_matrix_str)
                adj_matrix_list = adj_matrix_str.split('\n')
                #print("Adjacency matrix string after newline split: ", adj_matrix_list)
                adj_matrix = [[int(num) for num in row.strip("[]").replace(" ", "")] for row in adj_matrix_list]
                adj_matrix = np.array(adj_matrix)
                return adj_matrix
            
            print("Final graph string after converting from string to numpy array:")
            print(adjacency_matrix_from_string(final_graph_str))

            def convert_string_list(string_list):
                # Remove the square brackets from the string
                string_list = string_list.strip("[]")
                # Split the string by spaces
                numbers = string_list.split()
                # Convert each number to an integer and create a list
                result = [int(num) for num in numbers]
                return result

            #return
            
            # Convert final_graph_str into a networkx graph
            #final_graph = nx.from_numpy_array(np.array(eval(final_graph_str)))

            final_graph = nx.from_numpy_array(adjacency_matrix_from_string(final_graph_str), parallel_edges=True)
            print(str(nx.adjacency_matrix(final_graph).todense()))
            print("Final graph (converted into nx graph): ", final_graph)
            #return

            # Convert graph to string
            #graph_str = str(nx.adjacency_matrix(graph).todense())
            final_graph_str = final_graph_str.replace(",", "") # remove extra commas
            final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas

            print("Final graph string: ", final_graph_str)

            # Read graph to file using write_adjlist
            graph_original_filename = f"data/input_graphs/graph_nx_{i}.graphml"
            G = nx.read_graphml(graph_original_filename)
            print("G extracted: ", G)
            print(str(nx.adjacency_matrix(G).todense()))


            return
            """

            # Write graph to file
            graph_filename = f"data/ablation_few/input_graphs/{i}.graphml"
            #with open(graph_filename, "w") as graph_file:
            #    graph_file.write(final_graph_str)
            nx.write_graphml(final_graph, graph_filename)

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_prompt = "A: "
            #end_count_prompt = "A: "
            #end_yes_no_prompt = "A: "

            ablation_dir += "/"

            # Graph property tasks
            for task in ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "cycle", "add_edge", "remove_edge", "add_node", "remove_node"]: 

                # Extract prompts
                prompt_dir = f"data/prompts/{task}" 
                prompt_filename = f"prompt_{i}.txt"

                with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
                    prompt = prompt_file.read()

                # Extract solutions
                solution_dir = f"data/solutions/{task}"
                if task in augment_tasks:
                    solution_filename = f"solution_{i}.graphml"

                    with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
                        solution = nx.read_graphml(solution_file)
                else:
                    solution_filename = f"solution_{i}.txt"

                    with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
                        solution = solution_file.read()
                #print("Task: ", task)
                #print("Solution filename: ", solution_filename)

                few(final_graph, final_graph_str, augment_tasks, init_prompt, end_prompt, i, task, examples, examples_strs, True, prompt, solution)
            """
            few(final_graph, final_graph_str, augment_tasks, init_prompt, end_count_prompt, i, "node_count", examples, examples_strs, True)
            #return
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
            """

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

        #random.seed(i)

        # Generate CoT examples
        for e in range(max_num_examples):
            p = random.uniform(0, 1)
            n_nodes = random.randint(5, 20)
            example_graph = nx.erdos_renyi_graph(n_nodes, p)

            def is_complete_graph(G):
                n = len(G.nodes)
                # A complete graph with n nodes has n*(n-1)/2 edges
                expected_num_edges = n * (n - 1) // 2
                actual_num_edges = len(G.edges)
                return actual_num_edges == expected_num_edges

            # Check if the graph is complete
            is_complete = is_complete_graph(example_graph)
            #print("Graph is a complete graph:", is_complete)

            # Ensure that the graph is not complete and has at least one edge
            j = 0
            while (example_graph.number_of_edges() == 0) or (is_complete_graph(example_graph)):
                p = random.uniform(0, 1)
                n_nodes = random.randint(5, 20)
                example_graph = nx.erdos_renyi_graph(n_nodes, p)
                j += 1
                if j > 100:
                    print("Could not generate a valid graph after 100 attempts")
                    print(f"p: {p}, n: {n}")
                    example_graph_str = str(nx.adjacency_matrix(example_graph).todense())
                    print("Graph: ", example_graph_str)
                    sys.exit(1)

            # Convert graph to string
            example_graph_str = graph_to_string_encoder(example_graph)

            examples.append(example_graph)
            examples_strs.append(example_graph_str)

            # Extract final graph from input file
            input_dir = "data/input_graphs"
            input_filename = f"{i}.graphml"
            
            with open(os.path.join(input_dir, input_filename), "r") as input_file:
                final_graph = nx.read_graphml(input_file)
                final_graph_str = graph_to_string_encoder(final_graph)
            """
            # Add a comma after every integer in final_graph_str
            final_graph_str = final_graph_str.replace(" ", ", ") # TODO: this adds a few extra commas we don't want

            # Convert final_graph_str into a networkx graph
            final_graph = nx.fsom_numpy_array(np.array(eval(final_graph_str)))
            #print("Final graph: ", final_graph)
            #return

            # Convert graph to string
            #graph_str = str(nx.adjacency_matrix(graph).todense())
            final_graph_str = final_graph_str.replace(",", "") # remove extra commas
            final_graph_str = final_graph_str.replace("  ", " ") # remove extra commas
            """

            # Write graph to file
            graph_filename = f"data/ablation_cot/input_graphs/{i}.graphml"
            #with open(graph_filename, "w") as graph_file:
            #    graph_file.write(final_graph_str)
            nx.write_graphml(final_graph, graph_filename)

            init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            end_prompt = "A: "
            #end_count_prompt = "A: "
            #end_yes_no_prompt = "A: "

            ablation_dir += "/"

            # Graph property tasks
            for task in ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "cycle", "add_edge", "remove_edge", "add_node", "remove_node"]:   
                
                # Extract prompts
                prompt_dir = f"data/prompts/{task}" 
                prompt_filename = f"prompt_{i}.txt"

                with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
                    prompt = prompt_file.read()

                # Extract solutions
                solution_dir = f"data/solutions/{task}"
                if task in augment_tasks:
                    solution_filename = f"solution_{i}.graphml"

                    with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
                        solution = nx.read_graphml(solution_file)
                else:
                    solution_filename = f"solution_{i}.txt"

                    with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
                        solution = solution_file.read()
                
                cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_prompt, i, task, examples, examples_strs, prompt, solution)
            """
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
            """

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

    #num_examples = 2

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
        while (graph.number_of_edges() == 0) or (is_complete_graph(graph)):
            graph = nx.erdos_renyi_graph(n_nodes, p)
            j += 1
            if j > 100:
                print("Could not generate a valid graph after 100 attempts")
                print(f"p: {p}, n: {n}")
                graph_str = graph_to_string_encoder(graph)
                print("Graph: ", graph_str)
                sys.exit(1)
        """

        # Convert graph to string
        graph_str = graph_to_string_encoder(graph)

        # Write graph to file
        graph_filename = f"data/ablation_graph_type_{graphType}/input_graphs/{i}.graphml"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)
        nx.write_graphml(graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_prompt = "A: "
        #end_count_prompt = "A: "
        #end_yes_no_prompt = "A: "

        ablation_dir += "/"

        # Build a giant prompt filled with examples

        # Graph augmentation tasks
        add_edge_graph, _ = add_edge(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir, ablationType)
        remove_edge_graph, _ = remove_edge(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir, ablationType)
        add_node_graph, _ = add_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)
        remove_node_graph, _ = remove_node(graph, graph_str, init_prompt, end_prompt, i, False, ablation_dir)
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
        input_filename = f"{i}.graphml"
        
        with open(os.path.join(input_dir, input_filename), "r") as input_file:
            #final_graph_str = input_file.read()
            final_graph = nx.read_graphml(input_file)
            final_graph_str = graph_to_string_encoder(final_graph)

        """
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
        """

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
            graph_filename = f"data/ablation_encoding/input_graphs/{graph_type}/{i}.graphml"
            #with open(graph_filename, "w") as graph_file:
            #    graph_file.write(encoding_graph_str)
            nx.write_graphml(final_graph, graph_filename)

            # construct init_prompt
            if graph_type == "incidence": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
                # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
                nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
                init_prompt = f"G describes a graph among {nodes_str}.\nIn this graph:\n"
                end_mod_prompt = "A: "
            elif graph_type == "coauthorship":
                nodes_str = ', '.join([encoding_dict[int(n)] for n in nodes[:-2]]) + ', ' + encoding_dict[int(nodes[-2])] + ', and ' + encoding_dict[int(nodes[-1])]
                init_prompt = f"G describes a co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
                end_mod_prompt = "A: "
            elif graph_type == "friendship":
                nodes_str = ', '.join([encoding_dict[int(n)] for n in nodes[:-2]]) + ', ' + encoding_dict[int(nodes[-2])] + ', and ' + encoding_dict[int(nodes[-1])]
                init_prompt = f"G describes a friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
                end_mod_prompt = "A: "
            elif graph_type == "social_network":
                nodes_str = ', '.join([encoding_dict[int(n)] for n in nodes[:-2]]) + ', ' + encoding_dict[int(nodes[-2])] + ', and ' + encoding_dict[int(nodes[-1])]
                init_prompt = f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
                end_mod_prompt = "A: "

            #init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
            #end_matrix_prompt = "A: Final answer: The resulting adjacency matrix is: "
            end_count_prompt = "A: "
            end_yes_no_prompt = "A: "

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
        input_filename = f"{i}.graphml"
        
        with open(os.path.join(input_dir, input_filename), "r") as input_file:
            #final_graph_str = input_file.read()
            final_graph = nx.read_graphml(input_file)
            encoding_graph_str = graph_to_string_encoder(final_graph)

        graph_info_filename = f"data/input_graphs/{i}.txt"
        with open(graph_info_filename, "r") as graph_file:
            graph_info = graph_file.read()
            #numbers_list = graph_info.split(", ")
            #n = int(numbers_list[0].split(": ")[1])
            #p = float(numbers_list[1].split(": ")[1])
        

        """

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
        """

        # Write graph to file
        graph_filename = f"data/ablation_no_force/input_graphs/{i}.graphml"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(encoding_graph_str)
        nx.write_graphml(final_graph, graph_filename)

        new_graph_info_filename = f"data/ablation_no_force/input_graphs/{i}.txt"
        with open(new_graph_info_filename, "w") as graph_file:
            graph_file.write(graph_info)

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
    parser.add_argument("--experimentType", choices=["base", "chain", "p", "n", "d", "few", "cot", "few_chain", "cot_chain", "encoding", "graph_type", "no_force"], help="what type of graphs to generate for experiment")
    
    args = parser.parse_args()

    n_graphs = args.n_graphs

    experimentType = args.experimentType

    if experimentType == "base":
        print("Generating base prompts")
        generate_data(n_graphs)
    elif experimentType == "chain":
        print("Generating chain prompts")
        generate_chains_same(n_graphs)
    elif experimentType == "p":
        print("Density Ablation")
        for p in np.linspace(0, 1.0, 11):
            for n in range(5, 11):
                print(f"Generating graphs for p={p} and n={n}")
                generate_data_p(n_graphs, p, n)
    elif experimentType == "n":
        print("Size Ablation")
        for n in range(5, 15):
            print(f"Generating graphs for n={n}")
            generate_data_n(n_graphs, n)
    elif experimentType == "d": # directed
        print("Directed Ablation")
        for n in range(5, 15):
            print(f"Generating graphs for n={n}")
            generate_data_d(n_graphs, n, experimentType)
    elif experimentType == "few": # few-shot
        generate_data_few(n_graphs, experimentType)
    elif experimentType == "cot": # CoT
        generate_data_cot(n_graphs, experimentType)  
    elif experimentType == "few_chain":
        generate_chains_same_few(n_graphs)
    elif experimentType == "cot_chain":
        generate_chains_same_cot(n_graphs)
    elif experimentType == "encoding":
        generate_data_encoding(n_graphs, experimentType)
    elif experimentType == "graph_type":
        for graphType in ["barabasi_albert", "star", "path"]:
            generate_data_graph_type(n_graphs, experimentType, graphType)
    elif experimentType == "no_force":
        generate_data_no_force(n_graphs, experimentType)
    else:
        print("Please specify what type of prompts to generate")
        sys.exit(1)