
import os
import networkx as nx
import argparse
import random
import sys
import numpy as np
import re
import math

def few(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, prompt_type, examples, examples_strs, few, prompt, solution):
    num_examples = len(examples)
    full_example_prompt = ''

    print('Just entered few function')
    print(f"final_graph: {final_graph}")
    print(f"final_graph_str: {final_graph_str}")
    print(str(nx.adjacency_matrix(final_graph).todense()))
    
    if prompt_type == "add_edge":
        
        for n in range(num_examples):
            prompt_n, add_edge_graph, node_a, node_b = add_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

            # Save example input graph to file
            example_input_graph = examples[n]
        """
        #prompt1 = add_edge(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = add_edge(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        # ----------------------------
        # --- Add edge  ---
        # ----------------------------

        # Select two random nodes that are not connected
        unconnected_nodes = []
        for node_a in final_graph.nodes():
            for node_b in final_graph.nodes():
                if node_a != node_b and not final_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                    unconnected_nodes.append((node_a, node_b))
        #print(f"unconnected_nodes: {unconnected_nodes}")
        if len(unconnected_nodes) == 0:
            print(f"final_graph: {final_graph}")
            print(f"final_graph_str: {final_graph_str}")
            print(str(nx.adjacency_matrix(final_graph).todense()))
            sys.exit(1)
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]

        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        full_add_edge_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + add_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/add_edge/{num_examples}/prompt_{i}.txt"
        #print(f'prompt_filename: {prompt_filename}')
        #sys.exit(1)
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Create new graph with added edge
        add_edge_graph = final_graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_few/solutions/add_edge/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_edge_graph, solution_filename)
        """
        # Construct full prompt
        full_add_edge_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/add_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/add_edge/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_edge":
        #prompt1 = remove_edge(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = remove_edge(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n, remove_edge_graph, node_a, node_b = remove_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'
        """
        # ----------------------------
        # --- Remove edge  ---
        # ----------------------------

        # Select a random edge
        edge = random.choice(list(final_graph.edges()))

        remove_edge_prompt = f"Q: Remove the edge between node {edge[0]} and node {edge[1]}, and write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + remove_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/remove_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Create new graph with edge removed
        remove_edge_graph = final_graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_few/solutions/remove_edge/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_edge_graph, solution_filename)
        """

        # Construct full prompt
        full_remove_edge_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/remove_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/remove_edge/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "add_node":
        #prompt1 = add_node(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = add_node(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n, add_node_graph = add_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'
        """
        # ----------------------------
        # --- Add node  ---
        # ----------------------------

        number_of_nodes = final_graph.number_of_nodes()

        add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        full_add_node_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + add_node_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/add_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        # Create new graph with added node
        add_node_graph = final_graph.copy()
        add_node_graph.add_node(number_of_nodes)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_few/solutions/add_node/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_node_graph, solution_filename)
        """

        # Construct full prompt
        full_add_node_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/add_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/add_node/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_node":
        #prompt1 = remove_node(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = remove_node(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n, remove_node_graph, node = remove_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'
        """
        # ----------------------------
        # --- Remove node  ---
        # ----------------------------

        # Select a random node
        node = random.choice(list(final_graph.nodes()))

        remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        full_remove_node_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + remove_node_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/remove_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        # Create new graph with node removed
        remove_node_graph = final_graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_few/solutions/remove_node/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_node_graph, solution_filename)
        """

        # Construct full prompt
        full_remove_node_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/remove_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/remove_node/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "node_count":
        for n in range(num_examples):
            prompt_n = node_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'
        """
        # ----------------------------
        # --- Node count  ---
        # ----------------------------

        count = final_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n"
        full_node_count_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + node_count_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/node_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/node_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(count))
        """

        # Construct full prompt
        full_node_count_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/node_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/node_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "edge_count":
        for n in range(num_examples):
            prompt_n = edge_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Edge count  ---
        # ----------------------------
        """
        count = final_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"
        full_edge_count_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + edge_count_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/edge_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/edge_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(count))
        """

        # Construct full prompt
        full_edge_count_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/edge_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/edge_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "node_degree":
        for n in range(num_examples):
            prompt_n = node_degree(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Node degree  ---
        # ----------------------------
        """
        # Select a random node
        node = random.choice(list(final_graph.nodes()))
        degree = final_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"
        full_node_degree_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + node_degree_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/node_degree/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/node_degree/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(degree))
        """

        # Construct full prompt
        full_node_degree_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/node_degree/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/node_degree/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))
        
        return
    elif prompt_type == "edge_exists":
        for n in range(num_examples):
            prompt_n = edge_exists(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Edge exists  ---
        # ----------------------------
        """
        # Select two random nodes from the graph
        random_nodes = random.sample(list(final_graph.nodes()), 2)
        node_a, node_b = random_nodes

        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph?\n"
        full_edge_exists_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + edge_exists_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/edge_exists/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/edge_exists/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if final_graph.has_edge(node_a, node_b):
                solution_file.write("Yes")
            else:
                solution_file.write("No")
        """

        # Construct full prompt
        full_edge_exists_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/edge_exists/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/edge_exists/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "connected_nodes":
        for n in range(num_examples):
            prompt_n = connected_nodes(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Connected nodes  ---
        # ----------------------------
        """
        # Select one node from the graph that has at least one neighbor
        nodes_with_neighbors = [node for node in final_graph.nodes() if final_graph.degree[node] > 0]
        node = random.choice(nodes_with_neighbors)

        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n"
        full_connected_nodes_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + connected_nodes_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/connected_nodes/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_connected_nodes_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/connected_nodes/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            neighbor_list = sorted([node_b for node_b in final_graph.neighbors(node)])
            solution_file.write(str(neighbor_list))
        """

        # Construct full prompt
        full_connected_nodes_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/connected_nodes/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_connected_nodes_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/connected_nodes/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == 'cycle':
        for n in range(num_examples):
            prompt_n = cycle(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True, few=True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Cycle  ---
        # ----------------------------
        """
        # Create prompt string
        cycle_prompt = f"Q: Does the resulting graph contain a cycle?\n"
        full_cycle_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + cycle_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_few/prompts/cycle/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_cycle_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_few/solutions/cycle/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if nx.cycle_basis(final_graph):
                solution_file.write("Yes")
            else:
                solution_file.write("No")
        """

        # Construct full prompt
        full_cycle_prompt = full_example_prompt + prompt
        
        # Save full prompt
        prompt_filename = f"data/ablation_few/prompts/cycle/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_cycle_prompt)

        # Save solution
        solution_filename = f"data/ablation_few/solutions/cycle/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return

def cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, prompt_type, examples, examples_strs, prompt, solution):
    num_examples = len(examples)
    full_example_prompt = ''
    
    if prompt_type == "add_edge":
        
        for n in range(num_examples):
            prompt_n = add_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        #prompt1 = add_edge(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = add_edge(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        # ----------------------------
        # --- Add edge  ---
        # ----------------------------
        #print("Adding edge")
        # Select two random nodes that are not connected
        """
        unconnected_nodes = []
        for node_a in final_graph.nodes():
            for node_b in final_graph.nodes():
                if node_a != node_b and not final_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                    unconnected_nodes.append((node_a, node_b))
        #print(f"unconnected_nodes: {unconnected_nodes}")
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]

        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        full_add_edge_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + add_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/add_edge/{num_examples}/prompt_{i}.txt"
        #print(f'prompt_filename: {prompt_filename}')
        #sys.exit(1)
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Create new graph with added edge
        add_edge_graph = final_graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_cot/solutions/add_edge/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_edge_graph, solution_filename)
        """

        # Construct full prompt
        full_add_edge_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/add_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/add_edge/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_edge":
        #prompt1 = remove_edge(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = remove_edge(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n = remove_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Remove edge  ---
        # ----------------------------
        """
        # Select a random edge
        edge = random.choice(list(final_graph.edges()))

        remove_edge_prompt = f"Q: Remove the edge between node {edge[0]} and node {edge[1]}, and write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + remove_edge_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/remove_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Create new graph with edge removed
        remove_edge_graph = final_graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(remove_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_cot/solutions/remove_edge/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_edge_graph, solution_filename)
        """

        # Construct full prompt
        full_remove_edge_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/remove_edge/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/remove_edge/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "add_node":
        #prompt1 = add_node(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = add_node(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n = add_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Add node  ---
        # ----------------------------
        """
        number_of_nodes = final_graph.number_of_nodes()

        add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        full_add_node_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + add_node_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/add_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        # Create new graph with added node
        add_node_graph = final_graph.copy()
        add_node_graph.add_node(number_of_nodes)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_cot/solutions/add_node/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_node_graph, solution_filename)
        """

        # Construct full prompt
        full_add_node_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/add_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/add_node/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_node":
        #prompt1 = remove_node(examples[0], examples_strs[0], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #prompt2 = remove_node(examples[1], examples_strs[1], init_prompt, end_matrix_prompt, i, False, "", cot = True)
        #full_example_prompt = prompt1 + '\n' + prompt2
        #print(f'full_example_prompt: {full_example_prompt}')

        for n in range(num_examples):
            prompt_n = remove_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Remove node  ---
        # ----------------------------
        """
        # Select a random node
        node = random.choice(list(final_graph.nodes()))

        remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        full_remove_node_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + remove_node_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/remove_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        # Create new graph with node removed
        remove_node_graph = final_graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense())

        # Write new graph to file
        solution_filename = f"data/ablation_cot/solutions/remove_node/{num_examples}/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_node_graph, solution_filename)
        """

        # Construct full prompt
        full_remove_node_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/remove_node/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/remove_node/{num_examples}/solution_{i}.graphml"
        nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "node_count":
        for n in range(num_examples):
            prompt_n = node_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Node count  ---
        # ----------------------------
        """
        count = final_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n"
        full_node_count_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + node_count_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/node_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/node_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(count))
        """

        # Construct full prompt
        full_node_count_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/node_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/node_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "edge_count":
        for n in range(num_examples):
            prompt_n = edge_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Edge count  ---
        # ----------------------------
        """
        count = final_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"
        full_edge_count_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + edge_count_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/edge_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/edge_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(count))
        """

        # Construct full prompt
        full_edge_count_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/edge_count/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/edge_count/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "node_degree":
        for n in range(num_examples):
            prompt_n = node_degree(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Node degree  ---
        # ----------------------------
        """
        # Select a random node
        node = random.choice(list(final_graph.nodes()))
        degree = final_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"
        full_node_degree_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + node_degree_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/node_degree/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/node_degree/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(degree))
        """

        # Construct full prompt
        full_node_degree_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/node_degree/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/node_degree/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "edge_exists":
        for n in range(num_examples):
            prompt_n = edge_exists(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Edge exists  ---
        # ----------------------------
        """
        # Select two random nodes from the graph
        random_nodes = random.sample(list(final_graph.nodes()), 2)
        node_a, node_b = random_nodes

        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph?\n"
        full_edge_exists_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + edge_exists_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/edge_exists/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/edge_exists/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if final_graph.has_edge(node_a, node_b):
                solution_file.write("Yes")
            else:
                solution_file.write("No")
        """

        # Construct full prompt
        full_edge_exists_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/edge_exists/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/edge_exists/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == "connected_nodes":
        for n in range(num_examples):
            prompt_n = connected_nodes(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Connected nodes  ---
        # ----------------------------
        """
        # Select one node from the graph that has at least one neighbor
        nodes_with_neighbors = [node for node in final_graph.nodes() if final_graph.degree[node] > 0]
        node = random.choice(nodes_with_neighbors)

        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n"
        full_connected_nodes_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + connected_nodes_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/connected_nodes/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_connected_nodes_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/connected_nodes/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            neighbor_list = sorted([node_b for node_b in final_graph.neighbors(node)])
            solution_file.write(str(neighbor_list))
        """

        # Construct full prompt
        full_connected_nodes_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/connected_nodes/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_connected_nodes_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/connected_nodes/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
    elif prompt_type == 'cycle':
        for n in range(num_examples):
            prompt_n = cycle(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", cot = True)
            full_example_prompt += prompt_n + '\n'

        # ----------------------------
        # --- Cycle  ---
        # ----------------------------
        """
        # Create prompt string
        cycle_prompt = f"Q: Does the resulting graph contain a cycle?\n"
        full_cycle_prompt = full_example_prompt + init_prompt + final_graph_str + "\n" + cycle_prompt + end_matrix_prompt

        # Save prompt to file
        prompt_filename = f"data/ablation_cot/prompts/cycle/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_cycle_prompt)

        # Save solution to file
        solution_filename = f"data/ablation_cot/solutions/cycle/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if nx.cycle_basis(final_graph):
                solution_file.write("Yes")
            else:
                solution_file.write("No")
        """

        # Construct full prompt
        full_cycle_prompt = full_example_prompt + prompt

        # Save full prompt
        prompt_filename = f"data/ablation_cot/prompts/cycle/{num_examples}/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_cycle_prompt)

        # Save solution
        solution_filename = f"data/ablation_cot/solutions/cycle/{num_examples}/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))

        return
        

def add_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Add edge  ---
    # ----------------------------
    #print("Adding edge")
    # Select two random nodes that are not connected
    unconnected_nodes = []
    for node_a in graph.nodes():
        for node_b in graph.nodes():
            if node_a != node_b and not graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                unconnected_nodes.append((node_a, node_b))
    #print(f"unconnected_nodes: {unconnected_nodes}")
    node_a, node_b = random.sample(unconnected_nodes, 1)[0]

    if part_of_chain:
        if i == 1:
            add_edge_prompt = f"{i}: Add an edge between node {node_a} and node {node_b}.\n"
        else:
            add_edge_prompt = f"{i}: Add an edge between node {node_a} and node {node_b} in the resulting graph of operation {i-1}.\n"
        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)
        return add_edge_graph, add_edge_prompt, node_a, node_b
    elif cot:
        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        full_add_edge_prompt = init_prompt + graph_str + "\n" + add_edge_prompt + end_prompt

        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())
        if few:
            full_add_edge_prompt += new_graph_str
        else:
            full_add_edge_prompt += new_graph_str + f", because we replaced the 0 at row {node_a} and column {node_b} with a 1, and the 0 at row {node_b} and column {node_a} with a 1."
        #print(f'full_add_edge_prompt: {full_add_edge_prompt}')
        return full_add_edge_prompt, add_edge_graph, node_a, node_b
    else:
        #print(f'ablType: {ablationType}')
        #sys.exit(1)
        # Create prompt string
        if ablationType == "d": # directed
            add_edge_prompt = f"Q: Add an edge from node {node_a} to node {node_b}, and write the resulting adjacency matrix.\n"
        elif ablationType == "few": # few-shot
            example_prompt = f""
        elif ablationType == "no_force":
            add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        else:
            if graph_type == 'incidence':
                add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting incidence graph.\n"
            elif graph_type == 'adjacency':
                add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
            elif graph_type == 'coauthorship':
                add_edge_prompt = f"Q: Add an edge between authors {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting coauthorship graph.\n"
            elif graph_type == 'friendship':
                add_edge_prompt = f"Q: Add an edge between {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting friendship graph.\n"
            elif graph_type == 'social_network':
                add_edge_prompt = f"Q: Add an edge between {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting social network graph.\n"
        full_add_edge_prompt = init_prompt + graph_str + "\n" + add_edge_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/add_edge/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/add_edge/prompt_{i}.txt"
        #print(f'prompt_filename: {prompt_filename}')s
        #sys.exit(1)
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_edge_prompt)

        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        #new_graph_str = graph_to_string_encoder(add_edge_graph, graph_type, encoding_dict)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())

        # Write new graph to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/add_edge/solution_{i}.graphml" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/add_edge/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_edge_graph, solution_filename)

        return add_edge_graph, add_edge_prompt

def graph_to_string_encoder(graph, graph_type='adjacency', encoding_dict=None, error_analysis=None):
    # construct encoding_graph_str
    encoding_graph_str = ''

    for node in graph.nodes:
        # get the neighbors of the node
        neighbors = sorted(list(graph.neighbors(node)))

        if graph_type == "incidence":
            if len(neighbors) == 0:
                encoding_graph_str += f"Node {node} is not connected to any other node.\n"
            elif len(neighbors) == 1:
                encoding_graph_str += f"Node {node} is connected to node {neighbors[0]}.\n"
            else:
                # convert the neighbors list into a string separated by commas
                neighbors_str = ', '.join([str(n) for n in neighbors])
                encoding_graph_str += f"Node {node} is connected to nodes {neighbors_str}.\n"
        elif graph_type == "adjacency":
            encoding_graph_str = str(nx.adjacency_matrix(graph).todense())
        elif graph_type == "coauthorship":
            if len(neighbors) == 0:
                encoding_graph_str += f"{encoding_dict[int(node)]} has not co-authored any papers.\n"
            else:
                for neighbor in neighbors:
                    encoding_graph_str += f"{encoding_dict[int(node)]} and {encoding_dict[int(neighbor)]} wrote a paper together.\n"

        elif graph_type == "friendship":
            if len(neighbors) == 0:
                encoding_graph_str += f"{encoding_dict[int(node)]} has no friends.\n"
            else:
                for neighbor in neighbors:
                    encoding_graph_str += f"{encoding_dict[int(node)]} and {encoding_dict[int(neighbor)]} are friends.\n"

        elif graph_type == "social_network":
            if len(neighbors) == 0:
                encoding_graph_str += f"{encoding_dict[int(node)]} has no connections.\n"
            else:
                for neighbor in neighbors:
                    encoding_graph_str += f"{encoding_dict[int(node)]} and {encoding_dict[int(neighbor)]} are connected.\n"

    return encoding_graph_str

def remove_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Remove edge  ---
    # ----------------------------

    # Select a random edge
    edge = random.choice(list(graph.edges()))

    if part_of_chain:
        if i == 1:
            remove_edge_prompt = f"{i}: Remove the edge between node {edge[0]} and node {edge[1]}.\n"
        else:
            remove_edge_prompt = f"{i}: Remove the edge between node {edge[0]} and node {edge[1]} in the resulting graph of operation {i-1}.\n"
        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)
        return remove_edge_graph, remove_edge_prompt, edge[0], edge[1]
    elif cot:
        remove_edge_prompt = f"Q: Remove the edge between node {edge[0]} and node {edge[1]}, and write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = init_prompt + graph_str + "\n" + remove_edge_prompt + end_prompt

        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_edge_graph).todense())

        if few:
            full_remove_edge_prompt += new_graph_str
        else:
            full_remove_edge_prompt += new_graph_str + f", because we replaced the 1 at row {edge[0]} and column {edge[1]} with a 0, and the 1 at row {edge[1]} and column {edge[0]} with a 0."
        return full_remove_edge_prompt, remove_edge_graph, edge[0], edge[1]
    else:
        # Create prompt string
        node_a, node_b = edge
        if ablationType == "d": # directed
            remove_edge_prompt = f"Q: Remove the edge from node {node_a} to node {node_b}, and write the resulting adjacency matrix.\n"
        elif ablationType == "no_force":
            remove_edge_prompt = f"Q: Remove the edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        else:
            if graph_type == 'incidence':
                remove_edge_prompt = f"Q: Remove the edge between node {node_a} and node {node_b}, and write the resulting incidence graph.\n"
            elif graph_type == 'adjacency':
                remove_edge_prompt = f"Q: Remove the edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
            elif graph_type == 'coauthorship':
                remove_edge_prompt = f"Q: Remove the edge between authors {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting coauthorship graph.\n"
            elif graph_type == 'friendship':
                remove_edge_prompt = f"Q: Remove the edge between friends {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting friendship graph.\n"
            elif graph_type == 'social_network':
                remove_edge_prompt = f"Q: Remove the edge between {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]}, and write the resulting social network graph.\n"

        full_remove_edge_prompt = init_prompt + graph_str + "\n" + remove_edge_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/remove_edge/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/remove_edge/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_edge_prompt)

        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        #new_graph_str = graph_to_string_encoder(remove_edge_graph, graph_type, encoding_dict)

        # Write new graph to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/remove_edge/solution_{i}.graphml" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/remove_edge/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_edge_graph, solution_filename)

        return remove_edge_graph, remove_edge_prompt

def add_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Add node  ---
    # ----------------------------
    number_of_nodes = graph.number_of_nodes()

    if part_of_chain:
        if i == 1:
            add_node_prompt = f"{i}: Add a node to the graph.\n"
        else:
            add_node_prompt = f"{i}: Add a node to the resulting graph of operation {i-1}.\n"
        # Create new graph with added node
        add_node_graph = graph.copy()
        add_node_graph.add_node(number_of_nodes)
        return add_node_graph, add_node_prompt, number_of_nodes
    elif cot:
        add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        full_add_node_prompt = init_prompt + graph_str + "\n" + add_node_prompt + end_prompt

        # Create new graph with added node
        add_node_graph = graph.copy()
        add_node_graph.add_node(number_of_nodes)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense())

        if few:
            full_add_node_prompt += new_graph_str
        else:
            full_add_node_prompt += new_graph_str + f", because we appended a row of zeros and a column of zeros."

        return full_add_node_prompt, add_node_graph
    else:
        if ablationType == "no_force":
            add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        elif graph_type == 'incidence':
            add_node_prompt = f"Q: Add a node to the graph, and write the resulting incidence graph.\n"
        elif graph_type == 'adjacency':
            add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        elif graph_type == 'coauthorship':
            add_node_prompt = f"Q: Add a node called {encoding_dict[number_of_nodes]} to the graph, and write the resulting coauthorship graph.\n"
        elif graph_type == 'friendship':
            add_node_prompt = f"Q: Add a node called {encoding_dict[number_of_nodes]} to the graph, and write the resulting friendship graph.\n"
        elif graph_type == 'social_network':
            add_node_prompt = f"Q: Add a node called {encoding_dict[number_of_nodes]} to the graph, and write the resulting social network graph.\n"
        # Create prompt string
       # add_node_prompt = f"Q: Add a node to the graph. Only write the resulting adjacency matrix.\n"
        full_add_node_prompt = init_prompt + graph_str + "\n" + add_node_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/add_node/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/add_node/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        #print(f'Original graph: {graph_str}')

        # Add a node to the graph
        add_node_graph = graph.copy()
        #print(f'add_node_graph.nodes(): {add_node_graph.nodes()}')
        add_node_graph.add_node(number_of_nodes)
        #print(f'add_node_graph.nodes() after adding new node: {add_node_graph.nodes()}')

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense().astype(int))
        new_graph_str = graph_to_string_encoder(add_node_graph, graph_type, encoding_dict)

        #print(f'New graph after adding new node: {new_graph_str}')
        
        # Write new graph to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/add_node/solution_{i}.graphml" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/add_node/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(add_node_graph, solution_filename)

        return add_node_graph, add_node_prompt

def remove_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Remove node  ---
    # ----------------------------

    # Select a random node
    node = random.choice(list(graph.nodes()))

    if part_of_chain:
        if i == 1:
            remove_node_prompt = f"{i}: Remove node {node} from the graph.\n"
        else:
            remove_node_prompt = f"{i}: Remove node {node} from the resulting graph of operation {i-1}.\n"
        # Create new graph with node removed
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)
        return remove_node_graph, remove_node_prompt, node
    elif cot:
        remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        full_remove_node_prompt = init_prompt + graph_str + "\n" + remove_node_prompt + end_prompt

        # Create new graph with node removed
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense())

        if few:
            full_remove_node_prompt += new_graph_str
        else:
            full_remove_node_prompt += new_graph_str + f", because we removed all of node {node}'s edges, and removed row {node} and column {node}."

        return full_remove_node_prompt, remove_node_graph, node
    else:
        if ablationType == "no_force":
            remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        elif graph_type == 'incidence':
            remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting incidence graph.\n"
        elif graph_type == 'adjacency':
            remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        elif graph_type == 'coauthorship':
            remove_node_prompt = f"Q: Remove node {encoding_dict[int(node)]} from the graph, and write the resulting coauthorship graph.\n"
        elif graph_type == 'friendship':
            remove_node_prompt = f"Q: Remove node {encoding_dict[int(node)]} from the graph, and write the resulting friendship graph.\n"
        elif graph_type == 'social_network':
            remove_node_prompt = f"Q: Remove node {encoding_dict[int(node)]} from the graph, and write the resulting social network graph.\n"

        # Create prompt string
        #remove_node_prompt = f"Q: Remove node {node} from the graph. Only write the resulting adjacency matrix.\n"
        full_remove_node_prompt = init_prompt + graph_str + "\n" + remove_node_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/remove_node/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/remove_node/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        #print(f'Original graph: {graph_str}')

        # Remove node from the graph
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        #new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense().astype(int))
        #new_graph_str = graph_to_string_encoder(remove_node_graph, graph_type, encoding_dict)

        #print(f'New graph: {new_graph_str}')

        # Write new graph to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/remove_node/solution_{i}.graphml" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/remove_node/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(remove_node_graph, solution_filename)

        return remove_node_graph, remove_node_prompt

def chain(graph, graph_str, augment_tasks, static_tasks, init_prompt, end_prompt, i, final_task, chain_length):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    chain_graph = graph.copy()
    #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    print('-------------------')
    print(f'Graph before chain:{graph_str}')
    # Create prompt string
    chain_prompt = f"Q: Perform the following operations on the graph:\n"
    full_chain_prompt = init_prompt + graph_str + "\n" + chain_prompt
    
    tasks = []

    # Sample n_tasks tasks from the augment_tasks list
    for task_num in range(chain_length):
        # Sample a single task from the augment_tasks list
        task = random.choice(augment_tasks)
        print(f'Task initially chosen: {task}')
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        print(f'new_graph_str: {new_graph_str}')
        print()

        # For each node in the graph, create a copy of the graph and remove that node from the copy
        for node in chain_graph.nodes():
            # Create a copy of the graph
            temp_graph = chain_graph.copy()
            # Remove the node from the copy
            temp_graph.remove_node(node)
            # If the graph has 0 edges after removing the node, we cannot remove a node, so we need to sample a new task
            if temp_graph.number_of_edges() == 0 and task == "remove_node":
                print(f'Graph is disconnected after removing node {node}, so we need to sample a new task.')
                task = random.choice(["add_edge", "add_node"])
                break

        # If the graph has only one edge or one node, we cannot remove an edge or a node, respectively, so we need to sample a new task
        while (chain_graph.number_of_edges() == 1 and task in ["remove_node", "remove_edge"]) or (chain_graph.number_of_nodes() == 1 and task in ["remove_node", "remove_edge"]):
            print(f'Graph has only one edge or one node, so we need to sample a new task.')
            task = random.choice(["add_edge", "add_node"])

        print(f'Task {task} works!')
        print(f'Augmenting graph...')
        if task == "add_edge":
            chain_graph, prompt_to_append = add_edge(chain_graph, graph_str, "", "", task_num+1, True)
        elif task == "remove_edge":
            chain_graph, prompt_to_append = remove_edge(chain_graph, graph_str, "", "", task_num+1, True)
        elif task == "add_node":
            chain_graph, prompt_to_append = add_node(chain_graph, graph_str, "", "", task_num+1, True)
        elif task == "remove_node":
            chain_graph, prompt_to_append = remove_node(chain_graph, graph_str, "", "", task_num+1, True)

        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        print(f'Graph after augmenting via task {task}:{new_graph_str}')

        full_chain_prompt += prompt_to_append

    
    """
    for task in tasks:
        if task == "add_edge":
            chain_graph, prompt_to_append = add_edge(chain_graph, graph_str, "", "", task_num, True)
        elif task == "remove_edge":
            chain_graph, prompt_to_append = remove_edge(chain_graph, graph_str, "", "", task_num, True)
        elif task == "add_node":
            chain_graph, prompt_to_append = add_node(chain_graph, graph_str, "", "", task_num, True)
        elif task == "remove_node":
            chain_graph, prompt_to_append = remove_node(chain_graph, graph_str, "", "", task_num, True)

        full_chain_prompt += prompt_to_append
        task_num += 1
    """
    
    print('Picked all tasks!')
    print('Final task:', final_task)
    new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    print(f'Graph before final task:{new_graph_str}')

    if final_task == "node_count":
        node_count = chain_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
        full_chain_prompt += node_count_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_node_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_node_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_count))

        print('Solution:', node_count)
    elif final_task == "edge_count":
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
        full_chain_prompt += edge_count_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_edge_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_edge_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(edge_count))

        print('Solution:', edge_count)
    elif final_task == "node_degree":
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
        full_chain_prompt += node_degree_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_node_degree/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_node_degree/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_degree))

        print('Solution:', node_degree)
    elif final_task == "edge_exists":
        # Select two random nodes from the graph
        random_nodes = random.sample(list(chain_graph.nodes()), 2)
        node_a, node_b = random_nodes

        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph? Only write 'Yes' or 'No'.\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += edge_exists_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_edge_exists/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_edge_exists/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if chain_graph.has_edge(node_a, node_b):
                solution = "Yes"
                solution_file.write("Yes")
            else:
                solution_file.write("No")
                solution = "No"

        print('Solution:', solution)

    elif final_task == "connected_nodes":
        # Select one node from the graph that has at least one neighbor
        nodes_with_neighbors = [node for node in chain_graph.nodes() if chain_graph.degree[node] > 0]
        node = random.choice(nodes_with_neighbors)

        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += connected_nodes_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_connected_nodes/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_connected_nodes/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            connected_nodes = sorted([node_b for node_b in chain_graph.neighbors(node)])
            solution_file.write(str(connected_nodes))

        print('Solution:', connected_nodes)
    elif final_task == "cycle":
        # Create prompt string
        cycle_prompt = f"Q: Does the resulting graph contain a cycle?\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += cycle_prompt + end_prompt

        print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_cycle/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_cycle/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            try:
                nx.find_cycle(chain_graph)
                solution_file.write("Yes")
                solution = "Yes"
            except nx.NetworkXNoCycle:
                solution_file.write("No")
                solution = "No"

        print('Solution:', solution)
    elif final_task == "print_adjacency_matrix":
        full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_print/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        print('Final prompt to be saved:', full_chain_prompt)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        # Write new graph to file
        solution_filename = f"data/solutions/chain_{chain_length}_print/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(chain_graph, solution_filename)

        print('Solution:', new_graph_str)
    else:
        print("Final task not recognized. Exiting.")
        sys.exit(1)

    return chain_graph

def chain_same(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, max_chain_length):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    chain_graph = graph.copy()
    #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    #print('-------------------')
    #print(f'Graph before chain:{graph_str}')
    # Create prompt string
    chain_prompt = f"Q: Perform the following operations on the graph:\n"
    full_chain_prompt = init_prompt + graph_str + "\n" + chain_prompt
    
    tasks = []

    #node = random.choice(list(chain_graph.nodes()))
    #node_degree = chain_graph.degree[node]

    # Sample n_tasks tasks from the augment_tasks list
    for task_num in range(1, max_chain_length+1):
        # Sample a single task from the augment_tasks list
        #print(f'Task initially chosen: {task}')
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print(f'new_graph_str: {new_graph_str}')
        #print()

        # Save graph to file
        graph_filename = f"data/chains_same/{final_task}/{task}/{task_num}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        """

        # For each node in the graph, create a copy of the graph and remove that node from the copy
        for node in chain_graph.nodes():
            # Create a copy of the graph
            temp_graph = chain_graph.copy()
            # Remove the node from the copy
            temp_graph.remove_node(node)
            # If the graph has 0 edges after removing the node, we cannot remove a node, so we need to sample a new task
            if temp_graph.number_of_edges() == 0 and task == "remove_node":
                print(f'Graph is disconnected after removing node {node}, so we need to sample a new task.')
                task = random.choice(["add_edge", "add_node"])
                break
        

        # If the graph has only one edge or one node, we cannot remove an edge or a node, respectively, so we need to sample a new task
        while (chain_graph.number_of_edges() == 1 and task in ["remove_node", "remove_edge"]) or (chain_graph.number_of_nodes() == 1 and task in ["remove_node", "remove_edge"]):
            print(f'Graph has only one edge or one node, so we need to sample a new task.')
            task = random.choice(["add_edge", "add_node"])
        """

        #print(f'Task {task} works!')
        #print(f'Augmenting graph...')
        if task == "add_edge":
            chain_graph, prompt_to_append, _, _ = add_edge(chain_graph, graph_str, "", "", task_num, True)
        elif task == "remove_edge":
            chain_graph, prompt_to_append, _, _ = remove_edge(chain_graph, graph_str, "", "", task_num, True)
        elif task == "add_node":
            chain_graph, prompt_to_append, _ = add_node(chain_graph, graph_str, "", "", task_num, True)
        elif task == "remove_node":
            chain_graph, prompt_to_append, _ = remove_node(chain_graph, graph_str, "", "", task_num, True)

        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print(f'Graph after augmenting via task {task}:{new_graph_str}')

        full_chain_prompt += prompt_to_append
    
        #print('Picked all tasks!')
        #print('Final task:', final_task)
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print(f'Graph before final task:{new_graph_str}')

        if final_task == "node_count":
            node_count = chain_graph.number_of_nodes()
            # Create prompt string
            node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
            final_chain_prompt = full_chain_prompt + node_count_prompt + end_prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same/{final_task}/{task}/{task_num}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(final_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same/{final_task}/{task}/{task_num}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(node_count))

            #print('Solution:', node_count)
        elif final_task == "edge_count":
            edge_count = chain_graph.number_of_edges()
            # Create prompt string
            edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
            final_chain_prompt = full_chain_prompt + edge_count_prompt + end_prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same/{final_task}/{task}/{task_num}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(final_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same/{final_task}/{task}/{task_num}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(edge_count))

            #print('Solution:', edge_count)
        elif final_task == "node_degree":
            # Select a random node
            node = random.choice(list(chain_graph.nodes()))
            node_degree = chain_graph.degree[node]

            # Create prompt string
            node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
            final_chain_prompt = full_chain_prompt + node_degree_prompt + end_prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same/{final_task}/{task}/{task_num}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(final_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same/{final_task}/{task}/{task_num}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(node_degree))

            #print('Solution:', node_degree)
        elif final_task == "edge_exists":
            # Select two random nodes from the graph
            random_nodes = random.sample(list(chain_graph.nodes()), 2)
            node_a, node_b = random_nodes

            edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph?\n" # TODO: In the resulting graph,... instead?
            full_chain_prompt += edge_exists_prompt + end_prompt

            print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/prompts/chains_same/{final_task}/{task}/{chain_length}_edge_exists/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/solutions/chains_same/{final_task}/{task}/{chain_length}_edge_exists/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                if chain_graph.has_edge(node_a, node_b):
                    solution = "Yes"
                    solution_file.write("Yes")
                else:
                    solution_file.write("No")
                    solution = "No"

            print('Solution:', solution)

        elif final_task == "connected_nodes":
            # Select one node from the graph that has at least one neighbor
            nodes_with_neighbors = [node for node in chain_graph.nodes() if chain_graph.degree[node] > 0]
            node = random.choice(nodes_with_neighbors)

            # Create prompt string
            connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n" # TODO: In the resulting graph,... instead?
            full_chain_prompt += connected_nodes_prompt + end_prompt

            print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/prompts/chains_same/{task}/{chain_length}_connected_nodes/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/solutions/chains_same/{task}/{chain_length}_connected_nodes/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                connected_nodes = sorted([node_b for node_b in chain_graph.neighbors(node)])
                solution_file.write(str(connected_nodes))

            print('Solution:', connected_nodes)
        elif final_task == "cycle":
            # Create prompt string
            cycle_prompt = f"Q: Does the resulting graph contain a cycle?\n" # TODO: In the resulting graph,... instead?
            full_chain_prompt += cycle_prompt + end_prompt

            print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/prompts/chains_same/{task}/{chain_length}_cycle/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/solutions/chains_same/{task}/{chain_length}_cycle/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                try:
                    nx.find_cycle(chain_graph)
                    solution_file.write("Yes")
                    solution = "Yes"
                except nx.NetworkXNoCycle:
                    solution_file.write("No")
                    solution = "No"

            print('Solution:', solution)
        elif final_task == "print_adjacency_matrix":
            final_chain_prompt = full_chain_prompt + "What is the resulting adjacency matrix?\n" + end_prompt

            # Save prompt to file
            prompt_filename = f"data/chains_same/{final_task}/{task}/{task_num}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(final_chain_prompt)

            #print('Final prompt to be saved:', full_chain_prompt)

            # Convert graph to string
            new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

            # Write new graph to file
            solution_filename = f"data/chains_same/{final_task}/{task}/{task_num}/solutions/solution_{i}.graphml"
            #with open(solution_filename, "w") as solution_file:
            #    solution_file.write(new_graph_str)
            nx.write_graphml(chain_graph, solution_filename)

            #print('Solution:', new_graph_str)
        else:
            print("Final task not recognized. Exiting.")
            sys.exit(1)

    return chain_graph

def chain_same_example(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, few = False):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    chain_graph = graph.copy()
    #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    #print('-------------------')
    #print(f'Graph before chain:{graph_str}')
    # Create prompt string
    chain_prompt = f"Q: Perform the following operations on the graph:\n"
    full_chain_prompt = init_prompt + graph_str + "\n" + chain_prompt

    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()
    
    tasks = []
    involved_nodes = []

    for task_num in range(chain_length):

        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        if task == "add_edge":
            chain_graph, prompt_to_append, node_a, node_b = add_edge(chain_graph, graph_str, "", "", task_num+1, True)
            involved_nodes.append([node_a, node_b])
        elif task == "remove_edge":
            chain_graph, prompt_to_append, node_a, node_b = remove_edge(chain_graph, graph_str, "", "", task_num+1, True)
            involved_nodes.append([node_a, node_b])
        elif task == "add_node":
            chain_graph, prompt_to_append, node_a = add_node(chain_graph, graph_str, "", "", task_num+1, True)
            involved_nodes.append([node_a])
        elif task == "remove_node":
            chain_graph, prompt_to_append, node_a = remove_node(chain_graph, graph_str, "", "", task_num+1, True)
            involved_nodes.append([node_a])

        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        full_chain_prompt += prompt_to_append
    
    new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

    if final_task == "node_count":
        node_count = chain_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
        full_chain_prompt += node_count_prompt + end_prompt + f"{node_count}"

        if few:
            task_str = ""
        else:
            if task == "add_edge":
                edge = "edges" if chain_length > 1 else "edge"
                task_str = f", because after adding {chain_length} {edge} to the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
            elif task == "remove_edge":
                edge = "edges" if chain_length > 1 else "edge"
                task_str = f", because after removing {chain_length} {edge} from the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
            elif task == "add_node":
                node = "nodes" if chain_length > 1 else "node"
                task_str = f", because after adding {chain_length} {node} to the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
            elif task == "remove_node":
                node = "nodes" if chain_length > 1 else "node"
                task_str = f", because after removing {chain_length} {node} from the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
            else:
                pass

        full_chain_prompt += task_str + '\n'

    elif final_task == "edge_count":
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
        full_chain_prompt += edge_count_prompt + end_prompt + f"{edge_count}"

        if few:
            task_str = ""
        else:

            if task == "add_edge":
                edge = "edges" if chain_length > 1 else "edge"
                task_str = f", because after adding {chain_length} {edge} to the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
            elif task == "remove_edge":
                edge = "edges" if chain_length > 1 else "edge"
                task_str = f", because after removing {chain_length} {edge} from the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
            elif task == "add_node":
                node = "nodes" if chain_length > 1 else "node"
                task_str = f", because after adding {chain_length} {node} to the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
            elif task == "remove_node":
                node = "nodes" if chain_length > 1 else "node"
                task_str = f", because after removing {chain_length} {node} from the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
            else:
                pass

        full_chain_prompt += task_str + '\n'
        
    elif final_task == "node_degree":
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
        full_chain_prompt += node_degree_prompt + end_prompt + f"{node_degree}"

        if few:
            task_str = ""
        else:

            if task == "add_edge":
                edge = "edges" if chain_length > 1 else "edge"

                involved_nodes_str = ""
                for nodes in involved_nodes:
                    involved_nodes_str += f'the edge between nodes {nodes[0]} and {nodes[1]},'

                task_str = f", because after adding {involved_nodes_str} to the original graph where node {node} had {graph.degree[node]} neighbors, node {node} has {node_degree} neighbors in the resulting graph."
            elif task == "remove_edge":
                edge = "edges" if chain_length > 1 else "edge"

                involved_nodes_str = ""
                for nodes in involved_nodes:
                    involved_nodes_str += f'the edge between nodes {nodes[0]} and {nodes[1]},'

                task_str = f", because after removing {involved_nodes_str} from the original graph where node {node} had {graph.degree[node]} neighbors, node {node} has {node_degree} neighbors in the resulting graph."
            elif task == "add_node":
                #node = "nodes" if chain_length > 1 else "node"

                involved_nodes_str = ""
                for nodes in involved_nodes:
                    involved_nodes_str += f'node {nodes[0]},'

                node_in_original_graph = True
                try:
                    deg = graph.degree[node]
                except:
                    node_in_original_graph = False

                if node_in_original_graph:
                    task_str = f", because after adding {involved_nodes_str} to the original graph where node {node} had {graph.degree[node]} neighbors, node {node} has {node_degree} neighbors in the resulting graph."
                else:
                    task_str = f", because after adding {involved_nodes_str} to the original graph, node {node} has {node_degree} neighbors in the resulting graph."
            elif task == "remove_node":
                #node = "nodes" if chain_length > 1 else "node"

                involved_nodes_str = ""
                for nodes in involved_nodes:
                    involved_nodes_str += f'node {nodes[0]},'

                task_str = f", because after removing {involved_nodes_str} from the original graph where node {node} had {graph.degree[node]} neighbors, node {node} has {node_degree} neighbors in the resulting graph."
            else:
                pass

        full_chain_prompt += task_str + '\n'
        
    elif final_task == "print_adjacency_matrix":
        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt + f"{new_graph_str}"

        if few:
            task_str = ""
        else:

            if task == "add_edge":
                involved_nodes_str = ""
                for nodes in involved_nodes:
                    if len(involved_nodes) == 1:
                        involved_nodes_str += f'the 0 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 1'
                    elif nodes == involved_nodes[-1]:
                        involved_nodes_str += f'and the 0 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 1'
                    else:
                        involved_nodes_str += f'the 0 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 1, '

                task_str = f", because after changing {involved_nodes_str} in the original graph, the resulting graph is as shown above."
            elif task == "remove_edge":
                involved_nodes_str = ""
                for nodes in involved_nodes:
                    if len(involved_nodes) == 1:
                        involved_nodes_str += f'the 1 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 0'
                    elif nodes == involved_nodes[-1]:
                        involved_nodes_str += f'and the 1 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 0'
                    else:
                        involved_nodes_str += f'the 1 in entries ({nodes[0]},{nodes[1]}) and ({nodes[1]},{nodes[0]}) to a 0, '

                task_str = f", because after changing the {involved_nodes_str} in the original graph, the resulting graph is as shown above."
            elif task == "add_node":
                involved_nodes_str = ""
                for nodes in involved_nodes:
                    if len(involved_nodes) == 1:
                        involved_nodes_str += f'the row and column of zeros corresponding to node {nodes[0]} to the adjacency matrix,'
                    elif nodes == involved_nodes[-1]:
                        involved_nodes_str += f'and the row and column of zeros corresponding to node {nodes[0]} to the adjacency matrix,'
                    else:
                        involved_nodes_str += f'the row and column of zeros corresponding to node {nodes[0]} to the adjacency matrix, '

                task_str = f", because after adding {involved_nodes_str} the resulting graph is as shown above."
            elif task == "remove_node":
                involved_nodes_str = ""
                for nodes in involved_nodes:
                    if len(involved_nodes) == 1:
                        involved_nodes_str += f'the row and column corresponding to node {nodes[0]} from the adjacency matrix,'
                    elif nodes == involved_nodes[-1]:
                        involved_nodes_str += f'and the row and column corresponding to node {nodes[0]} from the adjacency matrix,'
                    else:
                        involved_nodes_str += f'the row and column corresponding to node {nodes[0]} from the adjacency matrix, '

                task_str = f", because after removing {involved_nodes_str} the resulting graph is as shown above."

        full_chain_prompt += task_str + '\n'
    else:
        print("Final task not recognized. Exiting.")
        sys.exit(1)

    return full_chain_prompt

def chain_same_few(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    example_chain_prompt = ''

    # Extract prompts
    prompt_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/prompts"
    prompt_filename = f"prompt_{i}.txt"

    with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
        prompt = prompt_file.read()

    # Extract solutions
    solution_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/solutions"
    if final_task == "print_adjacency_matrix":
        solution_filename = f"solution_{i}.graphml"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = nx.read_graphml(solution_file)
    else:
        solution_filename = f"solution_{i}.txt"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = solution_file.read()

    for n in range(len(examples)):
        example_chain_prompt += chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, True)

        chain_graph = graph.copy()
        #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print('-------------------')
        #print(f'Graph before chain:{graph_str}')
        # Create prompt string
        #chain_prompt = f"Q: Perform the following operations on the graph:\n"
        #full_chain_prompt = example_chain_prompt + init_prompt + graph_str + "\n" + chain_prompt
    
        tasks = []

        #n = len(examples)
        """
        # Sample n_tasks tasks from the augment_tasks list
        for task_num in range(chain_length):
            new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
            
            if task == "add_edge":
                chain_graph, prompt_to_append, _, _ = add_edge(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "remove_edge":
                chain_graph, prompt_to_append, _, _ = remove_edge(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "add_node":
                chain_graph, prompt_to_append, _ = add_node(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "remove_node":
                chain_graph, prompt_to_append, _ = remove_node(chain_graph, graph_str, "", "", task_num+1, True)

            new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
            #print(f'Graph after augmenting via task {task}:{new_graph_str}')

            full_chain_prompt += prompt_to_append
        """
        
        #print('Picked all tasks!')
        #print('Final task:', final_task)
        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print(f'Graph before final task:{new_graph_str}')

        graph_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        if final_task == "node_count":
            #node_count = chain_graph.number_of_nodes()
            # Create prompt string
            #node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_count)
        elif final_task == "edge_count":
            #edge_count = chain_graph.number_of_edges()
            # Create prompt string
            #edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
            #full_chain_prompt += edge_count_prompt + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', edge_count)
        elif final_task == "node_degree":
            # Select a random node
            #node = random.choice(list(chain_graph.nodes()))
            #node_degree = chain_graph.degree[node]

            # Create prompt string
            #node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
            #full_chain_prompt += node_degree_prompt + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_degree)
        elif final_task == "print_adjacency_matrix":
            #full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Write new graph to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.graphml"
            #with open(solution_filename, "w") as solution_file:
            #    solution_file.write(new_graph_str)
            nx.write_graphml(solution, solution_filename)

            #print('Solution:', new_graph_str)
        else:
            print("Final task not recognized. Exiting.")
            sys.exit(1)

    return chain_graph

def chain_same_cot(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    example_chain_prompt = ''

    # Extract prompts
    prompt_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/prompts"
    prompt_filename = f"prompt_{i}.txt"

    with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
        prompt = prompt_file.read()

    # Extract solutions
    solution_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/solutions"
    if final_task == "print_adjacency_matrix":
        solution_filename = f"solution_{i}.graphml"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = nx.read_graphml(solution_file)
    else:
        solution_filename = f"solution_{i}.txt"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = solution_file.read()

    for n in range(len(examples)):
        #example_chain_prompt += chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, True)
        example_chain_prompt += chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length)

        chain_graph = graph.copy()
        #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print('-------------------')
        #print(f'Graph before chain:{graph_str}')
        # Create prompt string
        #chain_prompt = f"Q: Perform the following operations on the graph:\n"
        #full_chain_prompt = example_chain_prompt + init_prompt + graph_str + "\n" + chain_prompt
    
        tasks = []

        #n = len(examples)
        """
        # Sample n_tasks tasks from the augment_tasks list
        for task_num in range(chain_length):
            new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
            
            if task == "add_edge":
                chain_graph, prompt_to_append, _, _ = add_edge(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "remove_edge":
                chain_graph, prompt_to_append, _, _ = remove_edge(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "add_node":
                chain_graph, prompt_to_append, _ = add_node(chain_graph, graph_str, "", "", task_num+1, True)
            elif task == "remove_node":
                chain_graph, prompt_to_append, _ = remove_node(chain_graph, graph_str, "", "", task_num+1, True)

            new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
            #print(f'Graph after augmenting via task {task}:{new_graph_str}')

            full_chain_prompt += prompt_to_append
        """
        
        #print('Picked all tasks!')
        #print('Final task:', final_task)
        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print(f'Graph before final task:{new_graph_str}')

        graph_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        if final_task == "node_count":
            #node_count = chain_graph.number_of_nodes()
            # Create prompt string
            #node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_count)
        elif final_task == "edge_count":
            #edge_count = chain_graph.number_of_edges()
            # Create prompt string
            #edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
            #full_chain_prompt += edge_count_prompt + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', edge_count)
        elif final_task == "node_degree":
            # Select a random node
            #node = random.choice(list(chain_graph.nodes()))
            #node_degree = chain_graph.degree[node]

            # Create prompt string
            #node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
            #full_chain_prompt += node_degree_prompt + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Save solution to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_degree)
        elif final_task == "print_adjacency_matrix":
            #full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt

            full_chain_prompt = example_chain_prompt + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt)

            # Write new graph to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.graphml"
            #with open(solution_filename, "w") as solution_file:
            #    solution_file.write(new_graph_str)
            nx.write_graphml(solution, solution_filename)

            #print('Solution:', new_graph_str)
        else:
            print("Final task not recognized. Exiting.")
            sys.exit(1)

    return chain_graph

"""
def chain_same_cot(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    example_chain_prompt = ''

    for n in range(len(examples)):
        example_chain_prompt += chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length)

    chain_graph = graph.copy()
    #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    #print('-------------------')
    #print(f'Graph before chain:{graph_str}')
    # Create prompt string
    #chain_prompt = f"Q: Perform the following operations on the graph:\n"
    #full_chain_prompt = example_chain_prompt + init_prompt + graph_str + "\n" + chain_prompt
    
    tasks = []

    # Extract prompts
    prompt_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/prompts"
    prompt_filename = f"prompt_{i}.txt"

    with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
        prompt = prompt_file.read()

    # Extract solutions
    solution_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/solutions"
    if final_task == "print_adjacency_matrix":
        solution_filename = f"solution_{i}.graphml"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = nx.read_graphml(solution_file)
    else:
        solution_filename = f"solution_{i}.txt"

        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            solution = solution_file.read()

    n = len(examples)


    
    #print('Picked all tasks!')
    #print('Final task:', final_task)
    #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
    #print(f'Graph before final task:{new_graph_str}')

    if final_task == "node_count":


        full_chain_prompt = example_chain_prompt + prompt

        # Save prompt to file
        prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))
    elif final_task == "edge_count":

        full_chain_prompt = example_chain_prompt + prompt

        # Save prompt to file
        prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))
        
    elif final_task == "node_degree":


        full_chain_prompt = example_chain_prompt + prompt

        # Save prompt to file
        prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(solution))
    elif final_task == "print_adjacency_matrix":


        full_chain_prompt = example_chain_prompt + prompt

        #print('Final prompt to be saved:', full_chain_prompt)

        # Save prompt to file
        prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Write new graph to file
        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n}/solutions/solution_{i}.graphml"
        #with open(solution_filename, "w") as solution_file:
        #    solution_file.write(new_graph_str)
        nx.write_graphml(solution, solution_filename)
    else:
        print("Final task not recognized. Exiting.")
        sys.exit(1)

    return chain_graph
"""
def node_count(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Node count  ---
    # ----------------------------

    node_count = graph.number_of_nodes()
    if cot:
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n"
        full_node_count_prompt = init_prompt + graph_str + "\n" + node_count_prompt + end_prompt

        if few:
            full_node_count_prompt += f"{node_count}"
        else:
            full_node_count_prompt += f"{node_count}, because the adjacency matrix has {node_count} rows."

        return full_node_count_prompt
    else:
        # Create prompt string
        if graph_type == 'incidence':
            node_count_prompt = f"Q: How many nodes are in this graph?\n"
        elif graph_type == 'adjacency':
            node_count_prompt = f"Q: How many nodes are in this graph?\n"
        elif graph_type == 'coauthorship':
            node_count_prompt = f"Q: How many people are in this graph?\n"
        elif graph_type == 'friendship':
            node_count_prompt = f"Q: How many people are in this graph?\n"
        elif graph_type == 'social_network':
            node_count_prompt = f"Q: How many people are in this graph?\n"

        full_node_count_prompt = init_prompt + graph_str + "\n" + node_count_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/node_count/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/node_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_count_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/node_count/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/node_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_count))

def edge_count(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Edge count  ---
    # ----------------------------

    edge_count = graph.number_of_edges()
    if cot:
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"
        full_edge_count_prompt = init_prompt + graph_str + "\n" + edge_count_prompt + end_prompt

        if few:
            full_edge_count_prompt += f"{edge_count}"
        else:
            full_edge_count_prompt += f"{edge_count}, because the adjacency matrix has {edge_count*2} non-zero entries, and because the graph is undirected, there are {edge_count*2} // 2 = {edge_count} edges."

        return full_edge_count_prompt
    else:
        # Create prompt string
        if graph_type == 'incidence':
            edge_count_prompt = f"Q: How many edges are in this graph?\n" # TODO: in this undirected graph instead?
        elif graph_type == 'adjacency':
            edge_count_prompt = f"Q: How many edges are in this graph?\n"
        elif graph_type == 'coauthorship':
            edge_count_prompt = f"Q: How many coauthorships are in this graph?\n"
        elif graph_type == 'friendship':
            edge_count_prompt = f"Q: How many friendships are in this graph?\n"
        elif graph_type == 'social_network':
            edge_count_prompt = f"Q: How many connections are in this graph?\n"
        
        full_edge_count_prompt = init_prompt + graph_str + "\n" + edge_count_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/edge_count/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/edge_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_count_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/edge_count/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/edge_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if graph_type == 'adjacency' or graph_type == 'incidence':
                solution_file.write(str(edge_count))
            else:
                solution_file.write(str(int(edge_count * 2)))

def node_degree(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Node degree  ---
    # ----------------------------

    # Select a random node
    node = random.choice(list(graph.nodes()))
    node_degree = graph.degree[node]

    if cot:
        # Create prompt string
       
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"
        full_node_degree_prompt = init_prompt + graph_str + "\n" + node_degree_prompt + end_prompt

        if few:
            full_node_degree_prompt += f"{node_degree}"
        else:
            full_node_degree_prompt += f"{node_degree}, because the adjacency matrix has {node_degree} non-zero entries in row {node}."

        return full_node_degree_prompt, node
    else:
        if graph_type == 'incidence':
            node_degree_prompt = f"Q: How many neighbors does node {node} have?\n" # TODO: How many neighbors does node {node} have?/What is the degree of node {node}?/How many edges are connected to node {node}?
        elif graph_type == 'adjacency':
            node_degree_prompt = f"Q: How many neighbors does node {node} have?\n" # TODO: How many neighbors does node {node} have?/What is the degree of node {node}?/How many edges are connected to node {node}?
        elif graph_type == 'coauthorship':
            node_degree_prompt = f"Q: How many papers does {encoding_dict[int(node)]} have?\n"
        elif graph_type == 'friendship':
            node_degree_prompt = f"Q: How many friends does {encoding_dict[int(node)]} have?\n"
        elif graph_type == 'social_network':
            node_degree_prompt = f"Q: How many connections does {encoding_dict[int(node)]} have?\n"

        # Create prompt string
        full_node_degree_prompt = init_prompt + graph_str + "\n" + node_degree_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/node_degree/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/node_degree/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_node_degree_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/node_degree/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/node_degree/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_degree))

def edge_exists(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Edge exists  ---
    # ----------------------------

    # Select two random nodes from the graph
    random_nodes = random.sample(list(graph.nodes()), 2)
    node_a, node_b = random_nodes

    if cot:
        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph?\n"
        full_edge_exists_prompt = init_prompt + graph_str + "\n" + edge_exists_prompt + end_prompt

        if graph.has_edge(node_a, node_b):
            if few:
                full_edge_exists_prompt += f"Yes"
            else:
                full_edge_exists_prompt += f"Yes, because the adjacency matrix has a 1 in entries {node_a},{node_b} and {node_b},{node_a}."
        else:
            if few:
                full_edge_exists_prompt += f"No"
            else:
                full_edge_exists_prompt += f"No, because the adjacency matrix has a 0 in entries {node_a},{node_b} and {node_b},{node_a}."

        return full_edge_exists_prompt, node_a, node_b
    else:
        if ablationType == "no_force":
            edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}?\n"
        elif graph_type == 'incidence':
            edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}?\n"
        elif graph_type == 'adjacency':
            edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}?\n"
        elif graph_type == 'coauthorship':
            edge_exists_prompt = f"Q: Have {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]} written a paper together?\n"
        elif graph_type == 'friendship':
            edge_exists_prompt = f"Q: Are {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]} friends?\n"
        elif graph_type == 'social_network':
            edge_exists_prompt = f"Q: Are {encoding_dict[int(node_a)]} and {encoding_dict[int(node_b)]} connected?\n"
        
        full_edge_exists_prompt = init_prompt + graph_str + "\n" + edge_exists_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/edge_exists/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/edge_exists/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_edge_exists_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/edge_exists/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/edge_exists/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if graph.has_edge(node_a, node_b):
                solution_file.write("Yes")
            else:
                solution_file.write("No")

def connected_nodes(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Connected nodes  ---
    # ----------------------------

    # Select one node from the graph that has at least one neighbor
    nodes_with_neighbors = [node for node in graph.nodes() if graph.degree[node] > 0]
    node = random.choice(nodes_with_neighbors)

    if cot:
        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n"
        full_connected_nodes_prompt = init_prompt + graph_str + "\n" + connected_nodes_prompt + end_prompt

        connected_nodes = sorted([node_b for node_b in graph.neighbors(node)])
        if len(connected_nodes) == 0:
            if few:
                full_connected_nodes_prompt += f"{connected_nodes}"
            else:
                full_connected_nodes_prompt += f"{connected_nodes}, because the adjacency matrix has a row of zeros corresponding to node {node}."
            
        else:
            neighbors_str = ''
            for node_b in connected_nodes:
                if node_b == connected_nodes[-1]:
                    neighbors_str += f"and ({node},{node_b})."
                else:
                    neighbors_str += f"({node},{node_b}), "
            if few:
                full_connected_nodes_prompt += f"{connected_nodes}"
            else:
                full_connected_nodes_prompt += f"{connected_nodes}, because the adjacency matrix has a 1 in entries {neighbors_str}"

        return full_connected_nodes_prompt, node
    else:
        if graph_type == 'incidence':
            connected_nodes_prompt = f"Q: List all neighbors of node {node}.\n" #in ascending order, and surround your final answer in brackets, like this: [answer]
        elif graph_type == 'adjacency':
            connected_nodes_prompt = f"Q: List all neighbors of node {node}.\n"
        elif graph_type == 'coauthorship':
            connected_nodes_prompt = f"Q: List all coauthors of {encoding_dict[int(node)]}.\n"
        elif graph_type == 'friendship':
            connected_nodes_prompt = f"Q: List all friends of {encoding_dict[int(node)]}.\n"
        elif graph_type == 'social_network':
            connected_nodes_prompt = f"Q: List all connections of {encoding_dict[int(node)]}.\n"
        # Create prompt string
        full_connected_nodes_prompt = init_prompt + graph_str + "\n" + connected_nodes_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/connected_nodes/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/connected_nodes/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_connected_nodes_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/connected_nodes/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/connected_nodes/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if graph_type == 'adjacency':
                connected_nodes = sorted([node_b for node_b in graph.neighbors(node)])
            else:
                #print(f'Node: {node}')
                #print(f'Neighbors: {[node_b for node_b in graph.neighbors(node)]}')
                #print(f'Encoding dict: {encoding_dict}')
                connected_nodes = sorted([encoding_dict[int(node_b)] for node_b in graph.neighbors(node)])
            solution_file.write(str(connected_nodes))

def cycle(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, cot = False, graph_type = 'adjacency', encoding_dict = None, few = False):
    # ----------------------------
    # --- Cycle  ---
    # ----------------------------

    if cot:
        # Create prompt string
        cycle_prompt = f"Q: Does the resulting graph contain a cycle?\n"
        full_cycle_prompt = init_prompt + graph_str + "\n" + cycle_prompt + end_prompt

        try:
            cycle = nx.find_cycle(graph)
            cycle_str = ''
            for pair in cycle:
                if pair == cycle[-1]:
                    cycle_str += f"and node {pair[0]} is connected to node {pair[1]}."
                else:
                    cycle_str += f"node {pair[0]} is connected to node {pair[1]}, "
            if few:
                full_cycle_prompt += "Yes"
            else:
                full_cycle_prompt += "Yes, because " + cycle_str
        except nx.NetworkXNoCycle:
            if few:
                full_cycle_prompt += "No"
            else:
                full_cycle_prompt += "No, because the adjacency matrix does not have a cycle."

        return full_cycle_prompt
    else:
        # Create prompt string
        if ablationType == 'no_force':
            cycle_prompt = f"Q: Does this graph contain a cycle?\n"
        else:
            cycle_prompt = f"Q: Does this graph contain a cycle?\n"
        full_cycle_prompt = init_prompt + graph_str + "\n" + cycle_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/{ablation_dir}prompts/{graph_type}/cycle/prompt_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}prompts/cycle/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_cycle_prompt)

        # Save solution to file
        solution_filename = f"data/{ablation_dir}solutions/{graph_type}/cycle/solution_{i}.txt" if graph_type != 'adjacency' else f"data/{ablation_dir}solutions/cycle/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            try:
                nx.find_cycle(graph)
                solution_file.write("Yes")
            except nx.NetworkXNoCycle:
                solution_file.write("No")
        