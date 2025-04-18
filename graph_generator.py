import os
import networkx as nx
import argparse
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json

from graph_generator_utils import *

def generate_data(n_graphs):
    # TODO: put all dir prep stuff this in a function
    """
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
    """
    encodings = ["adjacency_matrix", "adjacency_list", "incident", "coauthorship", "friendship", "social_network", "expert", "politician", "got", "sp"]

    for encoding in encodings:
        # Create directories if they don't exist
        os.makedirs(f"data/{encoding}/input_graphs", exist_ok=True)

        # Empty the directory if it's not empty
        if os.listdir(f"data/{encoding}/input_graphs"):
            for file_name in os.listdir(f"data/{encoding}/input_graphs"):
                file_path = os.path.join(f"data/{encoding}/input_graphs", file_name)
                os.remove(file_path)

        os.makedirs(f"data/{encoding}/prompts", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/add_edge", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/remove_edge", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/add_node", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/remove_node", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/node_count", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/edge_count", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/node_degree", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/edge_exists", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/connected_nodes", exist_ok=True)
        os.makedirs(f"data/{encoding}/prompts/cycle", exist_ok=True)

        # Empty the directories if they are not empty
        prompt_directories = [
            f"data/{encoding}/prompts/add_edge",
            f"data/{encoding}/prompts/remove_edge",
            f"data/{encoding}/prompts/add_node",
            f"data/{encoding}/prompts/remove_node",
            f"data/{encoding}/prompts/node_count",
            f"data/{encoding}/prompts/edge_count",
            f"data/{encoding}/prompts/node_degree",
            f"data/{encoding}/prompts/edge_exists",
            f"data/{encoding}/prompts/connected_nodes",
            f"data/{encoding}/prompts/cycle",
        ]
        for directory in prompt_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)

        os.makedirs(f"data/{encoding}/solutions", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/add_edge", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/remove_edge", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/add_node", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/remove_node", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/node_count", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/edge_count", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/node_degree", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/edge_exists", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/connected_nodes", exist_ok=True)
        os.makedirs(f"data/{encoding}/solutions/cycle", exist_ok=True)

        # Empty the directories if they are not empty
        solution_directories = [
        f"data/{encoding}/solutions/add_edge",
        f"data/{encoding}/solutions/remove_edge",
        f"data/{encoding}/solutions/add_node",
        f"data/{encoding}/solutions/remove_node",
        f"data/{encoding}/solutions/node_count",
        f"data/{encoding}/solutions/edge_count",
        f"data/{encoding}/solutions/node_degree",
        f"data/{encoding}/solutions/edge_exists",
        f"data/{encoding}/solutions/connected_nodes",
        f"data/{encoding}/solutions/cycle",
        ]

        for directory in solution_directories:
            if os.listdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)
    """
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

        # create a list of 20 strings of common names
        names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher"]
        south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red"]
        game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm"]

        for encoding in encodings:

             # Write graph to file using write_graphml
            graph_filename = f"data/{encoding}/input_graphs/{i}.graphml"
            nx.write_graphml(graph, graph_filename)

            encoding_dict = {}

            # get the list of nodes of final_graph
            nodes = list(graph.nodes)

            # enumerate over names
            for n, name in enumerate(names):
                if graph_type in ["coauthorship", "friendship", "social_network"]:
                    encoding_dict[n] = name
                else:
                    encoding_dict[n] = n

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
    """
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
    
        graph_to_prompts(graph, i)

    print("Data generation complete!")

def generate_chains_encodings(n_graphs):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["print_graph"]
    max_chain_length = 5

    for encoding in encodings:
        for modification in modifications:
            for final_task in final_tasks:
                for chain_length in range(1, max_chain_length + 1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/input_graphs", exist_ok=True)

                    # Empty the directory if it's not empty
                    if os.listdir(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/input_graphs"):
                        for file_name in os.listdir(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/input_graphs"):
                            file_path = os.path.join(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/input_graphs", file_name)
                            os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/prompts", exist_ok=True)


                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/prompts",
                    ]
                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                    f"data/{encoding}_chain_big/{final_task}/{modification}/{chain_length}/solutions",
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)
    
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating chain prompt {i}")
        p = random.uniform(0, 1)
        n = random.randint(7, 20)
        graph = nx.erdos_renyi_graph(n, p)

        # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
        while (graph.number_of_edges() < 5) or (graph.number_of_edges() > (n * (n - 1) // 2) - 5):
            p = random.uniform(0, 1)
            graph = nx.erdos_renyi_graph(n, p)

        #if i == 5:
        print(graph_to_string_encoder(graph))
        
        graph_to_prompts_chain(graph, i, max_chain_length)


        #sys.exit(0)

    print("Data generation complete!")

def generate_chains_encodings_no_print(n_graphs):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["triangle", "isolated"]
    max_chain_length = 5

    for encoding in encodings:
        for modification in modifications:
            for final_task in final_tasks:
                for chain_length in range(1, max_chain_length + 1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/input_graphs", exist_ok=True)

                    # Empty the directory if it's not empty
                    if os.listdir(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/input_graphs"):
                        for file_name in os.listdir(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/input_graphs"):
                            file_path = os.path.join(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/input_graphs", file_name)
                            os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/prompts", exist_ok=True)


                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/prompts",
                    ]
                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                    f"data/{encoding}_chain_no_print/{final_task}/{modification}/{chain_length}/solutions",
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)
    
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating chain prompt {i}")
        p = random.uniform(0, 1)
        n = random.randint(7, 20)
        graph = nx.erdos_renyi_graph(n, p)

        # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
        while (graph.number_of_edges() < 5) or (graph.number_of_edges() > (n * (n - 1) // 2) - 5):
            p = random.uniform(0, 1)
            graph = nx.erdos_renyi_graph(n, p)

        #if i == 5:
        print(graph_to_string_encoder(graph))
        
        graph_to_prompts_chain_no_print(graph, i, max_chain_length)


        #sys.exit(0)

    print("Data generation complete!")

def generate_chains_encodings_p(n_graphs):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["print_graph"]
    max_chain_length = 5
    for p in [0.1, 0.5, 0.9]:
        for n in [7, 10, 15, 20]:
            for encoding in encodings:
                for modification in modifications:
                    for final_task in final_tasks:
                        for chain_length in range(1, max_chain_length + 1):
                            # Create directories if they don't exist
                            os.makedirs(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/input_graphs", exist_ok=True)

                            # Empty the directory if it's not empty
                            if os.listdir(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/input_graphs"):
                                for file_name in os.listdir(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/input_graphs"):
                                    file_path = os.path.join(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/input_graphs", file_name)
                                    os.remove(file_path)

                            os.makedirs(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/prompts", exist_ok=True)


                            # Empty the directories if they are not empty
                            prompt_directories = [
                                f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/prompts",
                            ]
                            for directory in prompt_directories:
                                if os.listdir(directory):
                                    for file_name in os.listdir(directory):
                                        file_path = os.path.join(directory, file_name)
                                        os.remove(file_path)

                            os.makedirs(f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/solutions", exist_ok=True)

                            # Empty the directories if they are not empty
                            solution_directories = [
                            f"data/{encoding}_chain_p/{final_task}/{modification}/{chain_length}/{p}/{n}/solutions",
                            ]

                            for directory in solution_directories:
                                if os.listdir(directory):
                                    for file_name in os.listdir(directory):
                                        file_path = os.path.join(directory, file_name)
                                        os.remove(file_path)
    for p in [0.1, 0.5, 0.9]:
        for n in [7, 10, 15, 20]:
            print(f"Generating graphs with p = {p} and n = {n}")
            for i in range(n_graphs):
                random.seed(i)
                print(f"Generating chain prompt {i}")
                #p = random.uniform(0, 1)
                #n = random.randint(7, 20)
                graph = nx.erdos_renyi_graph(n, p)

                # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
                tries = 0
                while (graph.number_of_edges() < 5) or (graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                    #p = random.uniform(0, 1)
                    graph = nx.erdos_renyi_graph(n, p)
                    tries += 1
                    if tries > 1000:
                        print("Could not generate a valid graph after 1000 attempts")
                        print(f"p: {p}, n: {n}")
                        graph_str = str(nx.adjacency_matrix(graph).todense())
                        print("Graph: ", graph_str)
                        sys.exit(1)

                #if i == 5:
                #print(graph_to_string_encoder(graph))
                
                graph_to_prompts_chain_no_print(graph, i, max_chain_length, p, n)


        #sys.exit(0)

    print("Data generation complete!")

def generate_chains_encodings_graph_types(n_graphs, graphType):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix", "incident", "coauthorship"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "print_graph"]
    max_chain_length = 5

    for encoding in encodings:
        for modification in modifications:
            for final_task in final_tasks:
                for chain_length in range(1, max_chain_length + 1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/input_graphs", exist_ok=True)

                    # Empty the directory if it's not empty
                    if os.listdir(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/input_graphs"):
                        for file_name in os.listdir(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/input_graphs"):
                            file_path = os.path.join(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/input_graphs", file_name)
                            os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/prompts", exist_ok=True)


                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/prompts",
                    ]
                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                    f"data/{encoding}_chain_big_{graphType}/{final_task}/{modification}/{chain_length}/solutions",
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)
    
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating chain prompt {i}")
        n = random.randint(7, 20)
        if graphType == "star":
            # Randomly select an integer from 0 to n-1
            random_node = random.randint(0, n-1)

            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            
            # Put random_node at the front of the iterable
            nodes.remove(random_node)
            nodes.insert(0, random_node)
            print("Nodes: ", nodes)

            # Create a star graph with the randomly selected node as the center
            graph = nx.star_graph(nodes)
            print("Star graph: ", graph.edges)
        elif graphType == "path":
            """
            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            # Shuffle the iterable
            random.shuffle(nodes)

            # Create a path graph with the shuffled nodes
            graph = nx.path_graph(nodes)
            """
            graph = nx.path_graph(n)
        elif graphType == "empty":
            # Generate Erdos-Renyi graph that is not connected
            graph = nx.empty_graph(n)
        elif graphType == "complete":
            # Generate Erdos-Renyi graph that is not connected
            graph = nx.complete_graph(n)
        else:
            print("Graph type not supported")
            return
        
        # Initialize a graph of size n with no edges
        new_graph = nx.Graph()
        new_graph.add_nodes_from(range(n))

        # Loop through the edges of the graph and add them to the new graph
        for edge in graph.edges:
            new_graph.add_edge(edge[0], edge[1])

        # Convert graph to string
        graph_str = graph_to_string_encoder(new_graph)
        print("Graph: ", graph_str)
        
        graph_to_prompts_chain_graph_types(new_graph, graphType, i, max_chain_length)

        #sys.exit(0)

    print("Data generation complete!")

def generate_chains_encodings_graph_types_no_perform(n_graphs, graphType):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix", "incident", "coauthorship"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "print_graph"]
    max_chain_length = 5

    for encoding in encodings:
        for modification in modifications:
            for final_task in final_tasks:
                for chain_length in range(1, max_chain_length + 1):
                    # Create directories if they don't exist
                    os.makedirs(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/input_graphs", exist_ok=True)

                    # Empty the directory if it's not empty
                    if os.listdir(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/input_graphs"):
                        for file_name in os.listdir(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/input_graphs"):
                            file_path = os.path.join(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/input_graphs", file_name)
                            os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/prompts", exist_ok=True)


                    # Empty the directories if they are not empty
                    prompt_directories = [
                        f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/prompts",
                    ]
                    for directory in prompt_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)

                    os.makedirs(f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/solutions", exist_ok=True)

                    # Empty the directories if they are not empty
                    solution_directories = [
                    f"data/{encoding}_chain_big_{graphType}_no_perform/{final_task}/{modification}/{chain_length}/solutions",
                    ]

                    for directory in solution_directories:
                        if os.listdir(directory):
                            for file_name in os.listdir(directory):
                                file_path = os.path.join(directory, file_name)
                                os.remove(file_path)
    
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating chain prompt {i}")
        n = random.randint(7, 20)
        if graphType == "star":
            # Randomly select an integer from 0 to n-1
            random_node = random.randint(0, n-1)

            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            
            # Put random_node at the front of the iterable
            nodes.remove(random_node)
            nodes.insert(0, random_node)
            print("Nodes: ", nodes)

            # Create a star graph with the randomly selected node as the center
            graph = nx.star_graph(nodes)
            print("Star graph: ", graph.edges)
        elif graphType == "path":
            """
            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            # Shuffle the iterable
            random.shuffle(nodes)

            # Create a path graph with the shuffled nodes
            graph = nx.path_graph(nodes)
            """
            graph = nx.path_graph(n)
        elif graphType == "empty":
            # Generate Erdos-Renyi graph that is not connected
            graph = nx.empty_graph(n)
        elif graphType == "complete":
            # Generate Erdos-Renyi graph that is not connected
            graph = nx.complete_graph(n)
        else:
            print("Graph type not supported")
            return
        
        # Initialize a graph of size n with no edges
        new_graph = nx.Graph()
        new_graph.add_nodes_from(range(n))

        # Loop through the edges of the graph and add them to the new graph
        for edge in graph.edges:
            new_graph.add_edge(edge[0], edge[1])

        # Convert graph to string
        graph_str = graph_to_string_encoder(new_graph)
        print("Graph: ", graph_str)
        
        graph_to_prompts_chain_graph_types_no_perform(new_graph, graphType, i, max_chain_length)

        #sys.exit(0)

    print("Data generation complete!")

def get_chain_info(n_graphs):
    max_chain_length = 5
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating chain prompt {i}")
        p = random.uniform(0, 1)
        n = random.randint(7, 20)
        graph = nx.erdos_renyi_graph(n, p)

        # if the task if remove edge, while the graph has less than 5 edges, generate a new graph
        while (graph.number_of_edges() < 5) or (graph.number_of_edges() > (n * (n - 1) // 2) - 5):
            p = random.uniform(0, 1)
            graph = nx.erdos_renyi_graph(n, p)

        #if i == 5:
        print(graph_to_string_encoder(graph))
        get_info(graph, i, max_chain_length) # basically we regenerate all graphs and dictionaries, and we save just the relevant dictionaries somewhere.
        #return

def generate_chains_encodings_fc(n_graphs):
    # TODO: put all dir prep stuff this in a function
    encodings = ["adjacency_matrix"]
    modifications = ["add_edge"]
    final_tasks = ["print_graph"]
    max_chain_length = 3
    num_examples = 1
    #print(1)
    #sys.exit(0)

    for encoding in encodings:
        for modification in modifications:
            for final_task in final_tasks:
                for chain_length in range(1, max_chain_length + 1):
                    for example in range(1, num_examples + 1):
                        for ex_type in ["cot"]:
                            # Create directories if they don't exist
                            os.makedirs(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/input_graphs", exist_ok=True)

                            # Empty the directory if it's not empty
                            if os.listdir(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/input_graphs"):
                                for file_name in os.listdir(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/input_graphs"):
                                    file_path = os.path.join(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/input_graphs", file_name)
                                    os.remove(file_path)

                            os.makedirs(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/prompts", exist_ok=True)


                            # Empty the directories if they are not empty
                            prompt_directories = [
                                f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/prompts",
                            ]
                            for directory in prompt_directories:
                                if os.listdir(directory):
                                    for file_name in os.listdir(directory):
                                        file_path = os.path.join(directory, file_name)
                                        os.remove(file_path)

                            os.makedirs(f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/solutions", exist_ok=True)

                            # Empty the directories if they are not empty
                            solution_directories = [
                            f"data/{encoding}_chain_big_{ex_type}/{final_task}/{modification}/{chain_length}/{example}/solutions",
                            ]

                            for directory in solution_directories:
                                if os.listdir(directory):
                                    for file_name in os.listdir(directory):
                                        file_path = os.path.join(directory, file_name)
                                        os.remove(file_path)
    
    #print(2)
    #sys.exit(0)

    for i in range(n_graphs):
        print(f"Generating chain prompt {i}")
        
        graph_to_prompts_chain_fc(i, max_chain_length)

        #return


        #sys.exit(0)

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

        random.seed(i)

        print(f"Generating chain prompt {i}")
        p = random.uniform(0, 1)
        #if task == "remove_node":
        #    n = random.randint(6, 15)
        #else:
        #    n = random.randint(5, 15)
        n = random.randint(6, 15)
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
        while (graph.number_of_edges() < 5) or (graph.number_of_edges() > (n * (n - 1) // 2) - 5):
            p = random.uniform(0, 1)
            graph = nx.erdos_renyi_graph(n, p)

        # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
        #while (task == "add_edge" and graph.number_of_edges() > (n * (n - 1) // 2) - 5):
        #    p = random.uniform(0, 1)
        #    graph = nx.erdos_renyi_graph(n, p)

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

            chain_same(graph, graph_str, task, static_tasks, init_prompt, end_count_prompt, i, final_tasks, max_chain_length)
            """
            for final_task in final_tasks:
                # Write graph to file
                #graph_filename = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/{i}.graphml"
                #with open(graph_filename, "w") as graph_file:
                #    graph_file.write(graph_str)
                #nx.write_graphml(graph, graph_filename)
                if final_task == "node_count" or final_task == "edge_count" or final_task == "node_degree":
                    
                elif final_task == "print_adjacency_matrix":
                    chain_same(graph, graph_str, task, static_tasks, init_prompt, end_matrix_prompt, i, final_task, max_chain_length)
            """

def generate_chains_same_few_cot(n_graphs, max_num_examples = 5):
    for final_task in ["node_count", "edge_count", "node_degree", "print_adjacency_matrix"]:
        for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
            for chain_length in range(1, 6):
                for ablation_dir in ["chains_same_few", "chains_same_cot"]:
                    # Create directories if they don't exist
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

    max_chain_length = 5

    for i in range(n_graphs):

        examples = []
        examples_strs = []
        
        for e in range(max_num_examples):
            random.seed(i + e + 12345678)
            p = random.uniform(0, 1)
            #if task == "remove_node":
            #    n = random.randint(6, 15)
            #else:
            #    n = random.randint(5, 15)
            n = random.randint(6, 15)
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
            while (example_graph.number_of_edges() < 5) or (example_graph.number_of_edges() > (n * (n - 1) // 2) - 5):
                p = random.uniform(0, 1)
                example_graph = nx.erdos_renyi_graph(n, p)

            # if the task if add edge, while the graph has more than the maximum number of edges - 5, generate a new graph
            #while (task == "add_edge" and example_graph.number_of_edges() > (n * (n - 1) // 2) - 5):
            #    p = random.uniform(0, 1)
            #    example_graph = nx.erdos_renyi_graph(n, p)

            # Convert graph to string
            example_graph_str = graph_to_string_encoder(example_graph)

            examples.append(example_graph)
            examples_strs.append(example_graph_str)

            #print("Examples: ", examples)


            for task in augment_tasks:
                #print(f"Generating chain cot prompt {i}")

                init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
                end_prompt = "A: "

                print(f'Task: {task}, Graph: {i}')
                chain_same_few_cot(task, static_tasks, init_prompt, end_prompt, i, final_tasks, max_chain_length, examples, examples_strs)

"""
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

def generate_data_few_cot(n_graphs, ablationType, max_num_examples = 5):
    # Create directories if they don't exist
    #ablation_dir = f"ablation_few"
    for ablation_dir in ["ablation_few", "ablation_cot"]:
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
            random.seed(i + e + 12345678)
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

            # Write graph to file
            graph_filename = f"data/ablation_few/input_graphs/{i}.graphml"
            nx.write_graphml(final_graph, graph_filename)

            # Write graph to file
            graph_filename = f"data/ablation_cot/input_graphs/{i}.graphml"
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

                few_cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_prompt, i, task, examples, examples_strs, prompt, solution)

"""
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

def generate_data_graph_type_preserve(n_graphs, n, graphType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_preserve/{graphType}/{str(n)}"
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_node",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_node",
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
        #graph = nx.erdos_renyi_graph(n, 0.5)
        random.seed(i)

        if graphType == "star":
            # Randomly select an integer from 0 to n-1
            random_node = random.randint(0, n-1)

            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            
            # Put random_node at the front of the iterable
            nodes.remove(random_node)
            nodes.insert(0, random_node)
            print("Nodes: ", nodes)

            # Create a star graph with the randomly selected node as the center
            graph = nx.star_graph(nodes)
            print("Star graph: ", graph.edges)
        elif graphType == "path":
            # Create an iterable from 0 to n-1
            nodes = list(range(n))
            # Shuffle the iterable
            random.shuffle(nodes)

            # Create a path graph with the shuffled nodes
            graph = nx.path_graph(nodes)
        else:
            print("Graph type not supported")
            return
        
        # Initialize a graph of size n with no edges
        new_graph = nx.Graph()
        new_graph.add_nodes_from(range(n))

        # Loop through the edges of the graph and add them to the new graph
        for edge in graph.edges:
            new_graph.add_edge(edge[0], edge[1])

        # Convert graph to string
        graph_str = graph_to_string_encoder(new_graph)
        print("Graph: ", graph_str)

        # Write graph to file
        graph_filename = f"data/{ablation_dir}/input_graphs/{i}.graphml"
        #print(f"Writing graph to file {graph_filename}")
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(graph_str)
        nx.write_graphml(new_graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_prompt = "A: "

        ablation_dir += "/"
        
        add_node_preserve(new_graph, graph_str, init_prompt, end_prompt, i, False, nodes, n, ablation_dir, graphType)

        #return

def generate_data_encoding(n_graphs, ablationType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_encoding"
    

    encoding_types = ["adj_list", "incidence", "coauthorship", "friendship", "social_network", "expert", "politician", "got", "sp"]

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

        # create a list of 20 strings of common names
        names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher"]
        south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red"]
        game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm"]

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

            ablation_dir += "/"

            # Graph augmentation tasks
            add_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            remove_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            add_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            remove_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_mod_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)

            # Graph property tasks
            node_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            edge_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            node_degree(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            edge_exists(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            connected_nodes(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)
            cycle(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType, graph_type=graph_type, encoding_dict=encoding_dict)

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
        add_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType)
        remove_edge(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType)
        add_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType)
        remove_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType)

        # Graph property tasks
        node_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)
        edge_count(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)
        node_degree(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)
        edge_exists(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)
        connected_nodes(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)
        cycle(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_count_prompt, i=i, ablation_dir=ablation_dir, ablationType=ablationType)

def generate_data_add_node_without_connecting(n_graphs, ablationType):
    # Create directories if they don't exist
    ablation_dir = f"ablation_add_node_without_connecting"
    
    os.makedirs(f"data/{ablation_dir}/input_graphs", exist_ok=True)

    # Empty the directory if it's not empty
    if os.listdir(f"data/{ablation_dir}/input_graphs"):
        for file_name in os.listdir(f"data/{ablation_dir}/input_graphs"):
            file_path = os.path.join(f"data/{ablation_dir}/input_graphs", file_name)
            os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/prompts", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/prompts/add_node", exist_ok=True)

    # Empty the directories if they are not empty
    prompt_directories = [
        f"data/{ablation_dir}/prompts/add_node",
    ]

    for directory in prompt_directories:
        if os.listdir(directory):
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)

    os.makedirs(f"data/{ablation_dir}/solutions", exist_ok=True)
    os.makedirs(f"data/{ablation_dir}/solutions/add_node", exist_ok=True)

    # Empty the directories if they are not empty
    solution_directories = [
        f"data/{ablation_dir}/solutions/add_node",
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

        # Write graph to file
        graph_filename = f"data/ablation_add_node_without_connecting/input_graphs/{i}.graphml"
        #with open(graph_filename, "w") as graph_file:
        #    graph_file.write(encoding_graph_str)
        nx.write_graphml(final_graph, graph_filename)

        init_prompt = "The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        end_matrix_prompt = "A: "
        end_count_prompt = "A: "
        end_yes_no_prompt = "A: "

        ablation_dir += "/"

        # Graph augmentation tasks
        add_node(graph=final_graph, graph_str=encoding_graph_str, init_prompt=init_prompt, end_prompt=end_matrix_prompt, i=i, part_of_chain=False, ablation_dir=ablation_dir, ablationType=ablationType)

def generate_chains_abductive(n_graphs):
    for i in range(n_graphs):
        random.seed(i)
        print(f"Generating prompt {i}")

        simpleCluster = SimpleCluster(num_clusters=2, cluster_sizes=[5, 5], connection_probs=[0.5, 0.5], connect_to_ego=[False, False])

        graph = simpleCluster.get_graph()
        print(graph)
        
        print(graph_to_string_encoder_abductive(graph, graph_type="social_network"))

        # Visualize the graph
        plt.figure(figsize=(8, 6))
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"Graph {i}")
        plt.savefig(f"/home/clc348/GraphGen/graph_{i}.png")
        plt.close()

def load_graph(json_file):
    with open(json_file, 'r') as f:
        edges = json.load(f)
    return edges

def build_graph(edges):
    G = nx.Graph()
    birth_years = {}
    death_years = {}
    print("Building graph...")
    for edge in tqdm.tqdm(edges):
        node1, node2, year = edge

        # Skip if year is None
        if year == None:
            continue

        # Skip if node1 == node2
        if node1 == node2:
            continue

        G.add_edge(node1, node2, year=year)
        
        for node in [node1, node2]:
            if node not in birth_years:
                birth_years[node] = year
            else:
                if year < birth_years[node]:
                    birth_years[node] = year
            if year <= 2020:
                if node not in death_years:
                    death_years[node] = year
                else:
                    if year > death_years[node]:
                        death_years[node] = year
    
    print("Graph built.")
    return G, birth_years, death_years

def extract_ego_networks(G):
    ego_networks = {}
    print()
    print("Extracting ego networks...")
    maximum_size = 20
    minimum_size = 7
    for node in tqdm.tqdm(list(G.nodes())[:100000]):
        ego_graph = nx.ego_graph(G, node)

        if len(ego_graph.nodes()) > maximum_size or len(ego_graph.nodes()) < minimum_size:
            continue

        ego_networks[node] = ego_graph
    print(f"{len(ego_networks)} ego networks of max {maximum_size} nodes extracted.")
    return ego_networks

def save_prompt_and_solution_final(story, ego_network, ego_network_count, bucket_size, total_triangles, overlapped_nodes, overlapped_edges):
    """
    # Node Count Question
    
    # Prompt File
    #story.append(f"Q: How many nodes are in the network?\nA: ")

    node_count = len(ego_network.nodes())

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/node_count/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/node_count", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: How many nodes are in the resulting graph?\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/node_count/solution.txt"
    with open(solution_file_path, 'w') as f:
        f.write(str(node_count))

    # Edge Count Question

    # Prompt File

    # Remove the last line from the story
    #story = story[:-1]

    #story.append(f"Q: How many edges are in the network?\nA: ")
    edge_count = len(ego_network.edges())
    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/edge_count/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/edge_count", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: How many edges are in the resulting graph?\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/edge_count/solution.txt"
    with open(solution_file_path, 'w') as f:
        f.write(str(edge_count))

    # Node Degree Question

    # Prompt File

    # Remove the last line from the story
    #story = story[:-1]

    node = random.choice(list(ego_network.nodes()))
    degree = ego_network.degree(node)
    #story.append(f"Q: How many papers has {node} written?\nA: ")

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/node_degree/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/node_degree", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: What is the degree of {node} in the resulting graph?\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/node_degree/solution.txt"
    with open(solution_file_path, 'w') as f:
        f.write(str(degree))

    # Connected Nodes Question

    # Prompt File

    # Remove the last line from the story
    #story = story[:-1]

    # Get all nodes that have at least one neighbor
    nodes_with_neighbors = [node for node in ego_network.nodes() if ego_network.degree(node) > 0]

    # Check if there are any nodes with neighbors
    if len(nodes_with_neighbors) == 0:
        # Sample a random node
        #print(f"No nodes with neighbors in ego network {ego_network_count}/{mod_count}.")
        node = random.choice(list(ego_network.nodes()))
    else:
        node = random.choice(nodes_with_neighbors)

    neighbors = list(ego_network.neighbors(node))
    #story.append(f"Q: List all authors that {node} has written a paper with.\nA: ")

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/connected_nodes/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/connected_nodes", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: List all nodes that {node} shares an edge with in the resulting graph.\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/connected_nodes/solution.txt"
    with open(solution_file_path, 'w') as f:
        if len(neighbors) == 0:
            f.write("None")
        elif len(neighbors) == 1:
            f.write(neighbors[0])
        else:
            f.write(", ".join(neighbors))
    """

    # Print Graph Question

    # Prompt File

    # Remove the last line from the story
    #story = story[:-1]

    #story.append(f"Q: What is the final resulting graph? Write a list of edges in the same format as the graph above.\nA: ")

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/print_graph/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/print_graph", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: What is the final resulting graph? Write a list of edges in the same format as the graph above.\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/print_graph/solution.txt"
    with open(solution_file_path, 'w') as f:
        for u, v, data in ego_network.edges(data=True):
            f.write(f"{u} is working on a project with {v}.\n")
    """

    # Isolated Nodes Question

    # Prompt File

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/isolated_nodes/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/isolated_nodes", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: List all isolated nodes in the resulting graph.\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/isolated_nodes/solution.txt"
    with open(solution_file_path, 'w') as f:
        isolated_nodes = [node for node in ego_network.nodes() if ego_network.degree(node) == 0]
        if len(isolated_nodes) == 0:
            f.write("None")
        elif len(isolated_nodes) == 1:
            f.write(isolated_nodes[0])
        else:
            f.write(", ".join(isolated_nodes))

    # Triangle Count Question

    # Prompt File

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/triangle_count/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/triangle_count", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: How many triangles were formed throughout the history of the graph?\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/triangle_count/solution.txt"
    with open(solution_file_path, 'w') as f:
        f.write(str(total_triangles))

    # Overlapped Nodes Question

    node1 = random.choice(list(overlapped_nodes.keys()))

    # sample another node that is not the same node as node1
    node2 = random.choice(list(overlapped_nodes.keys()))

    if len(overlapped_nodes.keys()) == 1:
        node2 = node1
    else:
        while node2 == node1:
            node2 = random.choice(list(overlapped_nodes.keys()))

    # Prompt File

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_nodes/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_nodes", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: Were {node1} and {node2} ever part of the group at the same time?\nA: "]))

    # Solution File
    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_nodes/solution.txt"
    with open(solution_file_path, 'w') as f:
        if node2 in overlapped_nodes[node1]:
            f.write("Yes")
        else:
            f.write("No")


    # Overlapped Edges Question

    edge1 = random.choice(list(overlapped_edges.keys()))

    # sample another edge that is not the same edge as edge1
    edge2 = random.choice(list(overlapped_edges.keys()))

    if len(overlapped_edges.keys()) == 1:
        edge2 = edge1
    else:
        while edge2 == edge1:
            edge2 = random.choice(list(overlapped_edges.keys()))

    # Prompt File

    prompt_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_edges/prompt.txt"
    # Create directory if it doesn't exist
    os.makedirs(f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_edges", exist_ok=True)
    with open(prompt_file_path, 'w') as f:
        f.write("\n".join(story + [f"Q: Was a project between {edge1[0]} and {edge1[1]} ever happening at the same time as a project between {edge2[0]} and {edge2[1]}?\nA: "]))

    # Solution File

    solution_file_path = f"data/coauth/{bucket_size}/{ego_network_count}/overlapped_edges/solution.txt"
    with open(solution_file_path, 'w') as f:
        if edge2 in overlapped_edges[edge1]:
            f.write("Yes")
        else:
            f.write("No")
    """

def update_and_save_final(ego_network, update_type, story, mod_count, node1, node2=None, year=None):
    if update_type == "add_edge":
        ego_network.add_edge(node1, node2, year=year)
        story.append(f"{mod_count}. {node1} starts a project with {node2}.")
    elif update_type == "remove_edge":
        ego_network.remove_edge(node1, node2)
        story.append(f"{mod_count}. The project between {node1} and {node2} comes to an end.")
    elif update_type == "add_node":
        ego_network.add_node(node1)
        story.append(f"{mod_count}. {node1} joins the group.")
    elif update_type == "remove_node":
        ego_network.remove_node(node1)
        story.append(f"{mod_count}. {node1} leaves the group.")

    return ego_network, story

def generate_ego_network_story_final(ego, graph, birth_years, death_years, ego_network_count, small_count, medium_count, large_count):
    edges_sorted = sorted(graph.edges(data=True), key=lambda x: x[2]['year'])
    window_size = 5
    
    if not edges_sorted:
        return "", 0, ""
    
    first_year = edges_sorted[0][2]['year']
    prev_year = edges_sorted[0][2]['year']

    final_year = edges_sorted[-1][2]['year']

    #print(f"first_year: {first_year}")
    #print(f"final_year: {final_year}")

    # first, find any authors who are born after the year of the first edge + window_size
    birthed_authors = [author for author in graph.nodes() if birth_years[author] > first_year + window_size]

    #for author in birthed_authors:
    #    print(f"author: {author}, birth_years[author]: {birth_years[author]}")

    # find authors who will die after the year of the first edge + window_size
    deathed_authors = [author for author in graph.nodes() if (death_years[author] > first_year + window_size) and (death_years[author] < final_year)]

    # if there are less than 2 authors who were born after the year of the first edge + 5, or less than 2 authors who will die after the year of the first edge + 5, return an empty string
    if len(birthed_authors) == 0 or len(deathed_authors) == 0:
        return "", 0, ""

    authors_to_write = [author for author in graph.nodes() if birth_years[author] <= first_year + window_size]

    #for author in deathed_authors:
    #    print(f"deathed_authors author: {author}, death_years[author]: {death_years[author]}")

    #story = [f"We are going to use a graph to model how members of a research group work together on projects. The research group members are the nodes, and there is an edge between two nodes if they are currently on a project. People can join and leave the group, and projects can begin and end, so this means that nodes and edges may be added to the graph or removed from it over time. When a person leaves the group, all their current projects come to an end. The following people are members of a research group: {", ".join(authors_to_write)}. At the beginning:"]

    beginning = True
    unseen_nodes = set(graph.nodes())
    seen_nodes = set()

    # initialize empty graph
    ego_graph = nx.Graph()

    count = 0
    add_edge_count = 0
    remove_edge_count = 0
    add_node_count = 0
    remove_node_count = 0

    # list of nodes to be removed from the ego network
    to_be_removed = []

    # list of edges to be removed from the ego network
    to_be_removed_edges = []

    initial_edge_count = 0

    # we will keep track of the triangles in the ego network after each edge is added
    unique_triangles = set()

    # we will keep track of all the nodes that have overlapped
    overlapped_nodes = {}

    # we will keep track of all the edges that have overlapped
    overlapped_edges = {}
    
    for u, v, data in edges_sorted:
        year = data['year']
        #print(f"year: {year}, u: {u}, v: {v}")

        #if year != prev_year:
            # remove nodes in to_be_removed from ego_graph
        #    for node in to_be_removed:
        #        ego_graph, story = update_and_save_final(ego_graph, "remove_node", story, count+1, node)
        #        remove_node_count += 1
        #        count += 1

        #    to_be_removed = []
            
        if (year > first_year + window_size) and beginning:
            beginning = False

            # count the number of triangles in ego_graph
            all_cliques = nx.enumerate_all_cliques(ego_graph)
            triangles = set(tuple(clique) for clique in all_cliques if len(clique) == 3)
            unique_triangles.update(triangles)

            # everytime a node is added into the network, we will keep track of the nodes that have overlapped
            for node in ego_graph.nodes():
                if node in overlapped_nodes:
                    overlapped_nodes[node].update(set(ego_graph.nodes()))
                else:
                    overlapped_nodes[node] = set(ego_graph.nodes())

            # every time an edge is added into the network, we will keep track of the edges that have overlapped
            for edge in ego_graph.edges():
                if edge in overlapped_edges:
                    overlapped_edges[edge].update(set(ego_graph.edges()))
                else:
                    overlapped_edges[edge] = set(ego_graph.edges())

            #print(f"story: {"\n".join(story)}")
            #print(f"unique_triangles: {unique_triangles}")
            #print(f"overlapped_nodes: {overlapped_nodes}")
            #print()
            #print(f"overlapped_edges: {overlapped_edges}")
            #sys.exit(1)

            if initial_edge_count < 5:
                return "", 0, ""

            to_be_seen_nodes = set()
            for unseen_neighbor in unseen_nodes:
                #if birth_years[unseen_neighbor] < year and birth_years[unseen_neighbor] < first_year + window_size:
                if birth_years[unseen_neighbor] <= first_year + window_size:
                    story.append(f"{unseen_neighbor} is not yet working on a project.")
                    ego_graph.add_node(unseen_neighbor)

                    to_be_seen_nodes.add(unseen_neighbor)

            unseen_nodes -= to_be_seen_nodes

            story.append("The following events then occur in order:")

        if beginning:
            ego_graph.add_edge(u, v, year=year)
            story.append(f"{u} is working on a project with {v}.")

            # Add nodes to seen_nodeså
            seen_nodes.add(u)
            seen_nodes.add(v)
            initial_edge_count += 1

            # Remove seen nodes from unseen_nodes
            unseen_nodes -= seen_nodes
            prev_year = year

            continue

            

        if year != prev_year:
            for unseen_neighbor in unseen_nodes:
                if birth_years[unseen_neighbor] > prev_year and birth_years[unseen_neighbor] <= year:
                    ego_graph, story = update_and_save_final(ego_graph, "add_node", story, count+1, unseen_neighbor)
                    add_node_count += 1
                    count += 1

                    # everytime a node is added into the network, we will keep track of the nodes that have overlapped
                    for node in ego_graph.nodes():
                        if node in overlapped_nodes:
                            overlapped_nodes[node].update(set(ego_graph.nodes()))
                        else:
                            overlapped_nodes[node] = set(ego_graph.nodes())
            
            for seen_neighbor in seen_nodes:
                # check if the seen neighbor died between the previous year and the current year
                if seen_neighbor in death_years.keys():
                    #if death_years[seen_neighbor] > prev_year and death_years[seen_neighbor] <= year:
                    if death_years[seen_neighbor] >= prev_year and death_years[seen_neighbor] < year:
                        if seen_neighbor not in to_be_removed:
                            to_be_removed.append(seen_neighbor)
        
            for prev_u, prev_v, prev_data in ego_graph.edges(data=True):
                if prev_data['year'] < year - window_size:
                    # place these edges in a to-be-removed list
                    to_be_removed_edges.append((prev_u, prev_v))

            for edge in to_be_removed_edges:
                ego_graph, story = update_and_save_final(ego_graph, "remove_edge", story, count+1, edge[0], edge[1])
                remove_edge_count += 1
                count += 1

            to_be_removed_edges = []

            # remove nodes in to_be_removed from ego_graph
            for node in to_be_removed:
                ego_graph, story = update_and_save_final(ego_graph, "remove_node", story, count+1, node)
                remove_node_count += 1
                count += 1

            to_be_removed = []

        ego_graph, story = update_and_save_final(ego_graph, "add_edge", story, count+1, u, v, year)
        add_edge_count += 1
        count += 1

        # count the number of triangles in ego_graph
        all_cliques = nx.enumerate_all_cliques(ego_graph)
        triangles = set(tuple(clique) for clique in all_cliques if len(clique) == 3)
        unique_triangles.update(triangles)

        # every time an edge is added into the network, we will keep track of the edges that have overlapped
        for edge in ego_graph.edges():
            if edge in overlapped_edges:
                overlapped_edges[edge].update(set(ego_graph.edges()))
            else:
                overlapped_edges[edge] = set(ego_graph.edges())

        # Add nodes to seen_nodes
        seen_nodes.add(u)
        seen_nodes.add(v)

        # Remove seen nodes from unseen_nodes
        unseen_nodes -= seen_nodes
        
        prev_year = year

    count += 1 + initial_edge_count

    # skip if we never need to slide the window
    if beginning:
        return "", 0, ""
    
    total_triangles = len(unique_triangles)

    if small_count == 250 and medium_count == 250 and large_count == 250:
        return "Done", 0, ""

    if 10 <= count and count <= 25:
        bucket_size = 'small'

        if small_count > 250:
            return "", 0, ""

        save_prompt_and_solution_final(story, ego_graph, small_count, 'small', total_triangles, overlapped_nodes, overlapped_edges)
        #print(f"small_count: {small_count}")
        #small_count += 1
        #print(f"small_count: {small_count}")
        #print(f"story: {"\n".join(story)}")
        #print(f"count: {count}")
        #print(f"add_edge_count: {add_edge_count}")
        #print(f"beginning: {beginning}")
        #sys.exit(1)
    elif 25 < count and count <= 50:
        bucket_size = 'medium'

        if medium_count > 250:
            return "", 0, ""
        
        save_prompt_and_solution_final(story, ego_graph, medium_count, 'medium', total_triangles, overlapped_nodes, overlapped_edges)
        #print(f"medium_count: {medium_count}")
        #medium_count += 1
        #print(f"medium_count: {medium_count}")
        #sys.exit(1)
    elif 50 < count and count <= 75:
        bucket_size = 'large'

        if large_count > 250:
            return "", 0, ""
        
        save_prompt_and_solution_final(story, ego_graph, large_count, 'large', total_triangles, overlapped_nodes, overlapped_edges)
        #print(f"large_count: {large_count}")
        #large_count += 1
        #print(f"large_count: {large_count}")
    else:
        # skip this ego network
        return "", 0, ""
    
    return "\n".join(story), count, bucket_size

def save_to_xml(ego_networks, birth_years, death_years, output_folder):
    print("Saving ego networks to XML...")
    ego_network_count = 0
    small_count, medium_count, large_count = 1, 1, 1
    size_dict = {}
    for ego, graph in tqdm.tqdm(ego_networks.items()):
        random.seed(ego_network_count)
        #root = ET.Element("EgoNetwork", name=ego)
        #edges_elem = ET.SubElement(root, "Edges")
        
        #for u, v, data in graph.edges(data=True):
        #    edge_elem = ET.SubElement(edges_elem, "Edge")
        #    ET.SubElement(edge_elem, "Node1").text = u
        #    ET.SubElement(edge_elem, "Node2").text = v
        #    ET.SubElement(edge_elem, "Year").text = str(data.get('year', 'Unknown'))
        
        #story_elem = ET.SubElement(root, "Story")
        story, count, bucket_size = generate_ego_network_story_final(ego, graph, birth_years, death_years, ego_network_count, small_count, medium_count, large_count)
        if story == "":
            #print(f"Skipping ego network {ego} due to low node birth/death events.")
            continue
        ego_network_count += 1

        if bucket_size == 'small':
            small_count += 1
        elif bucket_size == 'medium':
            medium_count += 1
        elif bucket_size == 'large':
            large_count += 1
        
        # increase size_dict[count] by 1
        if count not in size_dict:
            size_dict[count] = 1
        else:
            size_dict[count] += 1

        if story == "Done":
            break

        #if ego_network_count == 5:
        #    break
        #else:
        #    continue

    print(f"size_dict: {size_dict}")

    # plot histogram of sizes
    plt.bar(size_dict.keys(), size_dict.values())
    plt.xlabel("Size of Ego Network")
    plt.ylabel("Count")
    plt.title("Histogram of Ego Network Sizes")

    # save the plot
    plt.savefig("histogram.png")
    
    #tree = ET.ElementTree(root)
    #xml_filename = f"{output_folder}/{ego}_ego_network.xml"
    #tree.write(xml_filename)
    #print(f"Saved: {xml_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, help="number of graphs to generate")
    parser.add_argument("--experimentType", choices=["base", "chain", "p", "n", "d", "few_cot", "few_cot_chain", "encoding", "encoding_chain", "graph_type", "no_force", "preserve", "node_connect", "info", "encoding_chain_fc", "encoding_chain_graph_type", "encoding_no_print", "encoding_chain_graph_type_no_perform", "abductive", "real"], help="what type of graphs to generate for experiment")
    
    args = parser.parse_args()

    n_graphs = args.n_graphs

    experimentType = args.experimentType

    if experimentType == "base":
        print("Generating base prompts")
        generate_data(n_graphs)
    elif experimentType == "chain":
        print("Generating chain prompts")
        generate_chains_same(n_graphs)
    elif experimentType == "info":
        print("Generating info prompts")
        get_chain_info(n_graphs)
    elif experimentType == "encoding_chain_fc":
        print("Generating encoding chain prompts")
        generate_chains_encodings_fc(n_graphs)
    elif experimentType == "encoding_chain_graph_type":
        for graph_type in ["empty"]:
            print("Generating encoding chain graph type prompts")
            generate_chains_encodings_graph_types(n_graphs, graph_type)
    elif experimentType == "encoding_chain_graph_type_no_perform":
        for graph_type in ["empty", "complete", "star", "path"]:
            print("Generating encoding chain graph type prompts")
            generate_chains_encodings_graph_types_no_perform(n_graphs, graph_type)
    elif experimentType == "p":
        print("Density Ablation")
        generate_chains_encodings_p(n_graphs)
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
    #elif experimentType == "few": # few-shot
    #    generate_data_few(n_graphs, experimentType)
    #elif experimentType == "cot": # CoT
    #    generate_data_cot(n_graphs, experimentType)  
    elif experimentType == "few_cot": # few-shot
        generate_data_few_cot(n_graphs, experimentType)
    elif experimentType == "few_cot_chain":
        generate_chains_same_few_cot(n_graphs)
    elif experimentType == "encoding":
        generate_data_encoding(n_graphs, experimentType)
    elif experimentType == "encoding_no_print":
        generate_chains_encodings_no_print(n_graphs)
    elif experimentType == "encoding_chain":
        generate_chains_encodings(n_graphs)
    elif experimentType == "graph_type":
        for graphType in ["barabasi_albert", "star", "path"]:
            generate_data_graph_type(n_graphs, experimentType, graphType)
    elif experimentType == "no_force":
        generate_data_no_force(n_graphs, experimentType)
    elif experimentType == "preserve":
        for n in range(5, 16):
            print(f"Generating graphs for n={n}")
            for graphType in ["star", "path"]:
                generate_data_graph_type_preserve(10, n, graphType)
    elif experimentType == "node_connect":
        generate_data_add_node_without_connecting(n_graphs, experimentType)
    elif experimentType == "abductive":
        generate_chains_abductive(n_graphs)
    elif experimentType == "real":
        json_file = "dblp_coauthorship.json"  # Change as needed
        output_folder = "ego_networks"  # Ensure this folder exists
        edges = load_graph(json_file)
        G, birth_years, death_years = build_graph(edges)
        ego_networks = extract_ego_networks(G)
        save_to_xml(ego_networks, birth_years, death_years, output_folder)
    else:
        print("Please specify what type of prompts to generate")
        sys.exit(1)