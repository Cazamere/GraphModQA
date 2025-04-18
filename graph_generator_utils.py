
import os
import networkx as nx
import argparse
import random
import sys
import numpy as np
import re
import math
import json
import os

def few_cot(final_graph, final_graph_str, augment_tasks, init_prompt, end_matrix_prompt, i, prompt_type, examples, examples_strs, prompt, solution):
    num_examples = len(examples)
    full_example_prompt_few = ''
    full_example_prompt_cot = ''

    #print('Just entered few function')
    #print(f"final_graph: {final_graph}")
    #print(f"final_graph_str: {final_graph_str}")
    #print(str(nx.adjacency_matrix(final_graph).todense()))
    
    if prompt_type == "add_edge":
        
        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = add_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_add_edge_prompt_few = full_example_prompt_few + prompt
            full_add_edge_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/add_edge/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_add_edge_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/add_edge/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/add_edge/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_add_edge_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/add_edge/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_edge":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = remove_edge(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)

            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_remove_edge_prompt_few = full_example_prompt_few + prompt
            full_remove_edge_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/remove_edge/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_remove_edge_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/remove_edge/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/remove_edge/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_remove_edge_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/remove_edge/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "add_node":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = add_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)

            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_add_node_prompt_few = full_example_prompt_few + prompt
            full_add_node_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/add_node/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_add_node_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/add_node/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/add_node/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_add_node_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/add_node/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "remove_node":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = remove_node(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_remove_node_prompt_few = full_example_prompt_few + prompt
            full_remove_node_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/remove_node/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_remove_node_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/remove_node/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/remove_node/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_remove_node_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/remove_node/{n+1}/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

        return
    elif prompt_type == "node_count":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = node_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_node_count_prompt_few = full_example_prompt_few + prompt
            full_node_count_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/node_count/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_node_count_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/node_count/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))
            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/node_count/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_node_count_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/node_count/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

        return
    elif prompt_type == "edge_count":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = edge_count(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_edge_count_prompt_few = full_example_prompt_few + prompt
            full_edge_count_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/edge_count/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_edge_count_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/edge_count/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/edge_count/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_edge_count_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/edge_count/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

        return
    elif prompt_type == "node_degree":

        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = node_degree(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_node_degree_prompt_few = full_example_prompt_few + prompt
            full_node_degree_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/node_degree/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_node_degree_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/node_degree/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/node_degree/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_node_degree_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/node_degree/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

        return
    elif prompt_type == "edge_exists":
       
        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = edge_exists(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_edge_exists_prompt_few = full_example_prompt_few + prompt
            full_edge_exists_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/edge_exists/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_edge_exists_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/edge_exists/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/edge_exists/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_edge_exists_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/edge_exists/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

        return
    elif prompt_type == "connected_nodes":
        
        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = connected_nodes(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_connected_nodes_prompt_few = full_example_prompt_few + prompt
            full_connected_nodes_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/connected_nodes/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_connected_nodes_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/connected_nodes/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/connected_nodes/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_connected_nodes_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/connected_nodes/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

        return
    elif prompt_type == 'cycle':
        
        for n in range(num_examples):
            prompt_few_n, prompt_cot_n = cycle(examples[n], examples_strs[n], init_prompt, end_matrix_prompt, i, False, "", in_context = True)
            full_example_prompt_few += prompt_few_n + '\n'
            full_example_prompt_cot += prompt_cot_n + '\n'

            # Construct full prompt
            full_cycle_prompt_few = full_example_prompt_few + prompt
            full_cycle_prompt_cot = full_example_prompt_cot + prompt

            # Save full few prompt
            prompt_filename = f"data/ablation_few/prompts/cycle/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_cycle_prompt_few)

            # Save few solution
            solution_filename = f"data/ablation_few/solutions/cycle/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

            # Save full cot prompt
            prompt_filename = f"data/ablation_cot/prompts/cycle/{n+1}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_cycle_prompt_cot)

            # Save cot solution
            solution_filename = f"data/ablation_cot/solutions/cycle/{n+1}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

        return

"""
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
"""

def add_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
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
    elif in_context:
        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}, and write the resulting adjacency matrix.\n"
        full_add_edge_prompt = init_prompt + graph_str + "\n" + add_edge_prompt + end_prompt

        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_edge_graph).todense())
        #if few:
        #    full_add_edge_prompt += new_graph_str
        #else:
        #    full_add_edge_prompt += new_graph_str + f", because we replaced the 0 at row {node_a} and column {node_b} with a 1, and the 0 at row {node_b} and column {node_a} with a 1."
        #print(f'full_add_edge_prompt: {full_add_edge_prompt}')
        full_add_edge_prompt_few = full_add_edge_prompt + new_graph_str
        full_add_edge_prompt_cot = full_add_edge_prompt + new_graph_str + f", because we replaced the 0 at row {node_a} and column {node_b} with a 1, and the 0 at row {node_b} and column {node_a} with a 1."
        return full_add_edge_prompt_few, full_add_edge_prompt_cot
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

def remove_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
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
    elif in_context:
        remove_edge_prompt = f"Q: Remove the edge between node {edge[0]} and node {edge[1]}, and write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = init_prompt + graph_str + "\n" + remove_edge_prompt + end_prompt

        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_edge_graph).todense())

        #if few:
        #    full_remove_edge_prompt += new_graph_str
        #else:
        #    full_remove_edge_prompt += new_graph_str + f", because we replaced the 1 at row {edge[0]} and column {edge[1]} with a 0, and the 1 at row {edge[1]} and column {edge[0]} with a 0."
        #return full_remove_edge_prompt

        full_remove_edge_prompt_few = full_remove_edge_prompt + new_graph_str
        full_remove_edge_prompt_cot = full_remove_edge_prompt + new_graph_str + f", because we replaced the 1 at row {edge[0]} and column {edge[1]} with a 0, and the 1 at row {edge[1]} and column {edge[0]} with a 0."

        return full_remove_edge_prompt_few, full_remove_edge_prompt_cot
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

def add_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
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
    elif in_context:
        add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        full_add_node_prompt = init_prompt + graph_str + "\n" + add_node_prompt + end_prompt

        # Create new graph with added node
        add_node_graph = graph.copy()
        add_node_graph.add_node(number_of_nodes)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense())

        #if few:
        #    full_add_node_prompt += new_graph_str
        #else:
        #    full_add_node_prompt += new_graph_str + f", because we appended a row of zeros and a column of zeros."

        #return full_add_node_prompt

        full_add_node_prompt_few = full_add_node_prompt + new_graph_str
        full_add_node_prompt_cot = full_add_node_prompt + new_graph_str + f", because we appended a row of zeros and a column of zeros."

        return full_add_node_prompt_few, full_add_node_prompt_cot
    else:
        if ablationType == "no_force":
            add_node_prompt = f"Q: Add a node to the graph, and write the resulting adjacency matrix.\n"
        elif ablationType == "node_connect":
            add_node_prompt = f"Q: Add a node to the graph without connecting the new node to existing nodes, and write the resulting adjacency matrix.\n"
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

def add_node_preserve(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, nodes, n, ablation_dir = "", graphType='star'):
    # ----------------------------
    # --- Add node  ---
    # ----------------------------
    number_of_nodes = graph.number_of_nodes()

    add_node_prompt = f"Q: Add a node to the graph while preserving the underlying graph structure, and write the resulting adjacency matrix.\n"

    full_add_node_prompt = init_prompt + graph_str + "\n" + add_node_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/{ablation_dir}prompts/add_node/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_add_node_prompt)

    # Add a node to the graph
    add_node_graph = graph.copy()
    #print(f'add_node_graph.nodes(): {add_node_graph.nodes()}')
    add_node_graph.add_node(number_of_nodes)
    
    if graphType == 'star':
        solution_graph = add_node_graph.copy()
        solution_graph.add_edge(number_of_nodes, nodes[0])

        # Write solution to file
        solution_filename = f"data/{ablation_dir}solutions/add_node/solution_{i}.graphml"
        nx.write_graphml(solution_graph, solution_filename)
    elif graphType == 'path':
        solution_1_graph = add_node_graph.copy()
        solution_1_graph.add_edge(number_of_nodes, nodes[0])

        solution_2_graph = add_node_graph.copy()
        solution_2_graph.add_edge(number_of_nodes, nodes[-1])

        # Write first solution to file
        solution_filename = f"data/{ablation_dir}solutions/add_node/solution_{i}_0.graphml"
        nx.write_graphml(solution_1_graph, solution_filename)

        # Write second solution to file
        solution_filename = f"data/{ablation_dir}solutions/add_node/solution_{i}_1.graphml"
        nx.write_graphml(solution_2_graph, solution_filename)

    return add_node_graph, add_node_prompt

def remove_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
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
    elif in_context:
        remove_node_prompt = f"Q: Remove node {node} from the graph, and write the resulting adjacency matrix.\n"
        full_remove_node_prompt = init_prompt + graph_str + "\n" + remove_node_prompt + end_prompt

        # Create new graph with node removed
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense())

        #if few:
        #    full_remove_node_prompt += new_graph_str
        #else:
        #    full_remove_node_prompt += new_graph_str + f", because we removed all of node {node}'s edges, and removed row {node} and column {node}."

        #return full_remove_node_prompt

        full_remove_node_prompt_few = full_remove_node_prompt + new_graph_str
        full_remove_node_prompt_cot = full_remove_node_prompt + new_graph_str + f", because we removed all of node {node}'s edges, and removed row {node} and column {node}."

        return full_remove_node_prompt_few, full_remove_node_prompt_cot
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

def chain_same(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_tasks, max_chain_length):
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

        #if final_task == "node_count":
        node_count = chain_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
        final_nc_chain_prompt = full_chain_prompt + node_count_prompt + end_prompt

        # Save input graph to new file
        graph_filename = f"data/chains_same/node_count/{task}/{task_num}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        # Save prompt to file
        prompt_filename = f"data/chains_same/node_count/{task}/{task_num}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(final_nc_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same/node_count/{task}/{task_num}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_count))

            #print('Solution:', node_count)
        #elif final_task == "edge_count":
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
        final_ec_chain_prompt = full_chain_prompt + edge_count_prompt + end_prompt

        # Save input graph to new file
        graph_filename = f"data/chains_same/edge_count/{task}/{task_num}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        # Save prompt to file
        prompt_filename = f"data/chains_same/edge_count/{task}/{task_num}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(final_ec_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same/edge_count/{task}/{task_num}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(edge_count))

        #print('Solution:', edge_count)
        #elif final_task == "node_degree":
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
        final_nd_chain_prompt = full_chain_prompt + node_degree_prompt + end_prompt

        # Save input graph to new file
        graph_filename = f"data/chains_same/node_degree/{task}/{task_num}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        # Save prompt to file
        prompt_filename = f"data/chains_same/node_degree/{task}/{task_num}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(final_nd_chain_prompt)

        # Save solution to file
        solution_filename = f"data/chains_same/node_degree/{task}/{task_num}/solutions/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_degree))

        #print('Solution:', node_degree)
        """
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
        """
        final_print_chain_prompt = full_chain_prompt + "What is the resulting adjacency matrix?\n" + end_prompt

        # Save input graph to new file
        graph_filename = f"data/chains_same/print_adjacency_matrix/{task}/{task_num}/input_graphs/{i}.graphml"
        nx.write_graphml(graph, graph_filename)

        # Save prompt to file
        prompt_filename = f"data/chains_same/print_adjacency_matrix/{task}/{task_num}/prompts/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(final_print_chain_prompt)

        # Save solution graph to file
        solution_filename = f"data/chains_same/print_adjacency_matrix/{task}/{task_num}/solutions/solution_{i}.graphml"
        nx.write_graphml(chain_graph, solution_filename)

            #print('Solution:', new_graph_str)

    return chain_graph

def chain_same_example(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, max_chain_length):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    chain_graph = graph.copy()

    # Create prompt string
    chain_prompt = f"Q: Perform the following operations on the graph:\n"
    full_chain_prompt = init_prompt + graph_str + "\n" + chain_prompt

    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()
    
    tasks = []
    involved_nodes = []

    lengths_to_prompt_few_dict = {}
    lengths_to_prompt_cot_dict = {}

    chain_prompt = ''

    for task_num in range(1, max_chain_length+1):

        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        if task == "add_edge":
            chain_graph, prompt_to_append, node_a, node_b = add_edge(chain_graph, graph_str, "", "", task_num, True)
            involved_nodes.append([node_a, node_b])
        elif task == "remove_edge":
            chain_graph, prompt_to_append, node_a, node_b = remove_edge(chain_graph, graph_str, "", "", task_num, True)
            involved_nodes.append([node_a, node_b])
        elif task == "add_node":
            chain_graph, prompt_to_append, node_a = add_node(chain_graph, graph_str, "", "", task_num, True)
            involved_nodes.append([node_a])
        elif task == "remove_node":
            chain_graph, prompt_to_append, node_a = remove_node(chain_graph, graph_str, "", "", task_num, True)
            involved_nodes.append([node_a])

        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        chain_prompt += prompt_to_append

        final_chain_prompt = full_chain_prompt + chain_prompt
    
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        task_to_prompt_few_dict = {}
        task_to_prompt_cot_dict = {}

        # node count
        node_count = chain_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
        final_node_count_chain_prompt = final_chain_prompt + node_count_prompt + end_prompt + f"{node_count}"

        if task == "add_edge":
            edge = "edges" if task_num > 1 else "edge"
            task_str = f", because after adding {task_num} {edge} to the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
        elif task == "remove_edge":
            edge = "edges" if task_num > 1 else "edge"
            task_str = f", because after removing {task_num} {edge} from the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
        elif task == "add_node":
            node = "nodes" if task_num > 1 else "node"
            task_str = f", because after adding {task_num} {node} to the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
        elif task == "remove_node":
            node = "nodes" if task_num > 1 else "node"
            task_str = f", because after removing {task_num} {node} from the original graph that had {original_node_count} nodes, the resulting graph has {node_count} nodes."
        else:
            pass

        #full_chain_prompt += task_str + '\n'
        final_node_count_chain_prompt_few = final_node_count_chain_prompt + '\n'
        final_node_count_chain_prompt_cot = final_node_count_chain_prompt + task_str + '\n'

        task_to_prompt_few_dict['node_count'] = final_node_count_chain_prompt_few
        task_to_prompt_cot_dict['node_count'] = final_node_count_chain_prompt_cot

        # edge count
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
        final_edge_count_chain_prompt = final_chain_prompt + edge_count_prompt + end_prompt + f"{edge_count}"


        if task == "add_edge":
            edge = "edges" if task_num > 1 else "edge"
            task_str = f", because after adding {task_num} {edge} to the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
        elif task == "remove_edge":
            edge = "edges" if task_num > 1 else "edge"
            task_str = f", because after removing {task_num} {edge} from the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
        elif task == "add_node":
            node = "nodes" if task_num > 1 else "node"
            task_str = f", because after adding {task_num} {node} to the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
        elif task == "remove_node":
            node = "nodes" if task_num > 1 else "node"
            task_str = f", because after removing {task_num} {node} from the original graph that had {original_edge_count} edges, the resulting graph has {edge_count} edges."
        else:
            pass

        final_edge_count_chain_prompt_few = final_edge_count_chain_prompt + '\n'
        final_edge_count_chain_prompt_cot = final_edge_count_chain_prompt + task_str + '\n'

        task_to_prompt_few_dict['edge_count'] = final_edge_count_chain_prompt_few
        task_to_prompt_cot_dict['edge_count'] = final_edge_count_chain_prompt_cot
        
        # node degree
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
        final_node_count_chain_prompt = final_chain_prompt + node_degree_prompt + end_prompt + f"{node_degree}"

        if task == "add_edge":
            edge = "edges" if task_num > 1 else "edge"

            involved_nodes_str = ""
            for nodes in involved_nodes:
                involved_nodes_str += f'the edge between nodes {nodes[0]} and {nodes[1]},'

            task_str = f", because after adding {involved_nodes_str} to the original graph where node {node} had {graph.degree[node]} neighbors, node {node} has {node_degree} neighbors in the resulting graph."
        elif task == "remove_edge":
            edge = "edges" if task_num > 1 else "edge"

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

        final_node_count_chain_prompt_few = final_node_count_chain_prompt + '\n'
        final_node_count_chain_prompt_cot = final_node_count_chain_prompt + task_str + '\n'

        task_to_prompt_few_dict['node_degree'] = final_node_count_chain_prompt_few
        task_to_prompt_cot_dict['node_degree'] = final_node_count_chain_prompt_cot
        
        # print_adjacency_matrix
        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        final_print_chain_prompt = final_chain_prompt + f"What is the resulting adjacency matrix?\n" + end_prompt + f"{new_graph_str}"

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

            task_str = f", because after changing {involved_nodes_str} in the original graph, the resulting graph is as shown above."
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

        final_print_chain_prompt_few = final_print_chain_prompt + '\n'
        final_print_chain_prompt_cot = final_print_chain_prompt + task_str + '\n'

        task_to_prompt_few_dict['print_adjacency_matrix'] = final_print_chain_prompt_few
        task_to_prompt_cot_dict['print_adjacency_matrix'] = final_print_chain_prompt_cot

        lengths_to_prompt_few_dict[task_num] = task_to_prompt_few_dict
        lengths_to_prompt_cot_dict[task_num] = task_to_prompt_cot_dict

    return lengths_to_prompt_few_dict, lengths_to_prompt_cot_dict

"""
def chain_same_example(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, max_chain_length):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    chain_graph = graph.copy()

    # Create prompt string
    chain_prompt = f"Q: Perform the following operations on the graph:\n"
    full_chain_prompt = init_prompt + graph_str + "\n" + chain_prompt

    original_node_count = graph.number_of_nodes()
    original_edge_count = graph.number_of_edges()
    
    tasks = []
    involved_nodes = []

    for task_num in range(max_chain_length):

        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

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

        #new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        full_chain_prompt += prompt_to_append
    
    new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

    if final_task == "node_count":
        node_count = chain_graph.number_of_nodes()
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n" # TODO: in the resulting graph instead?
        full_chain_prompt += node_count_prompt + end_prompt + f"{node_count}"

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

        #full_chain_prompt += task_str + '\n'
        full_chain_prompt_few = full_chain_prompt + '\n'
        full_chain_prompt_cot = full_chain_prompt + task_str + '\n'

    elif final_task == "edge_count":
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
        full_chain_prompt += edge_count_prompt + end_prompt + f"{edge_count}"


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

        full_chain_prompt_few = full_chain_prompt + '\n'
        full_chain_prompt_cot = full_chain_prompt + task_str + '\n'
        
    elif final_task == "node_degree":
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"  # TODO: In the resulting graph,... instead?
        full_chain_prompt += node_degree_prompt + end_prompt + f"{node_degree}"


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

        full_chain_prompt_few = full_chain_prompt + '\n'
        full_chain_prompt_cot = full_chain_prompt + task_str + '\n'
        
    elif final_task == "print_adjacency_matrix":
        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt + f"{new_graph_str}"

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

        full_chain_prompt_few = full_chain_prompt + '\n'
        full_chain_prompt_cot = full_chain_prompt + task_str + '\n'
    else:
        print("Final task not recognized. Exiting.")
        sys.exit(1)

    return full_chain_prompt_few, full_chain_prompt_cot
"""
"""
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
"""

def chain_same_few_cot(task, static_tasks, init_prompt, end_prompt, i, final_tasks, max_chain_length, examples, examples_strs):

    example_to_chain_length_few_dict = {}
    example_to_chain_length_cot_dict = {}

    for n in range(len(examples)):
        lengths_to_prompt_few_dict, lengths_to_prompt_cot_dict = chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, max_chain_length)

        example_to_chain_length_few_dict[n+1] = lengths_to_prompt_few_dict
        example_to_chain_length_cot_dict[n+1] = lengths_to_prompt_cot_dict

        for chain_length in range(1, max_chain_length+1):
            for final_task in final_tasks:
                # build our final prompt
                ex_prompt = ''
                for ex in range(1, n+1):

                    # Extract input graph from input file
                    input_dir = f"data/chains_same/{final_task}/{task}/{chain_length}/input_graphs/"
                    input_filename = f"{i}.graphml"

                    # Read input graph
                    with open(os.path.join(input_dir, input_filename), "r") as input_file:
                        graph = nx.read_graphml(input_file)

                    # Save input graph to new file
                    graph_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{ex}/input_graphs/{i}.graphml"
                    nx.write_graphml(graph, graph_filename)

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

                    ex_prompt += example_to_chain_length_cot_dict[ex][chain_length][final_task]

                    # Save few prompt to file
                    prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{ex}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(ex_prompt + prompt)

                    # Save cot prompt to file
                    prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{ex}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(ex_prompt + prompt)

                    if final_task in ["node_count", "edge_count", "node_degree"]:
                        # Save few solution to file
                        solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{ex}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))

                        # Save cot solution to file
                        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{ex}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))
                    elif final_task == "print_adjacency_matrix":
                        # Save few solution to file
                        solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{ex}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)

                        # Save cot solution to file
                        solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{ex}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)
                    else:
                        print("Final task not recognized. Exiting.")
                        sys.exit(1)

    return

"""
def chain_same_few_cot(graph, graph_str, task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length, examples, examples_strs):
    # ----------------------------
    # --- Chain  ---
    # ----------------------------

    example_chain_prompt_few = ''
    example_chain_prompt_cot = ''

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
        example_chain_prompt_few_n, example_chain_prompt_cot_n = chain_same_example(examples[n], examples_strs[n], task, static_tasks, init_prompt, end_prompt, i, final_task, chain_length)
        example_chain_prompt_few += example_chain_prompt_few_n
        example_chain_prompt_cot += example_chain_prompt_cot_n

        # for 

        chain_graph = graph.copy()
        #graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))
        #print('-------------------')
        #print(f'Graph before chain:{graph_str}')
        # Create prompt string
        #chain_prompt = f"Q: Perform the following operations on the graph:\n"
        #full_chain_prompt = example_chain_prompt + init_prompt + graph_str + "\n" + chain_prompt
    
        tasks = []

        #n = len(examples)
        
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
            full_chain_prompt_few = example_chain_prompt_few + prompt
            full_chain_prompt_cot = example_chain_prompt_cot + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save few prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_few)

            # Save few solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            # Save cot prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_cot)

            # Save cot solution to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_count)
        elif final_task == "edge_count":
            #edge_count = chain_graph.number_of_edges()
            # Create prompt string
            #edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"  # TODO: in the resulting graph instead?
            #full_chain_prompt += edge_count_prompt + end_prompt

            #full_chain_prompt = example_chain_prompt + prompt

            full_chain_prompt_few = example_chain_prompt_few + prompt
            full_chain_prompt_cot = example_chain_prompt_cot + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save few prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_few)

            # Save few solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            # Save cot prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_cot)

            # Save cot solution to file
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

            #full_chain_prompt = example_chain_prompt + prompt

            full_chain_prompt_few = example_chain_prompt_few + prompt
            full_chain_prompt_cot = example_chain_prompt_cot + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save few prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_few)

            # Save few solution to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            # Save cot prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_cot)

            # Save cot solution to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(str(solution))

            #print('Solution:', node_degree)
        elif final_task == "print_adjacency_matrix":
            #full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt

            #full_chain_prompt = example_chain_prompt + prompt

            full_chain_prompt_few = example_chain_prompt_few + prompt
            full_chain_prompt_cot = example_chain_prompt_cot + prompt

            #print('Final prompt to be saved:', full_chain_prompt)

            # Save few prompt to file
            prompt_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_few)

            # Write new few graph to file
            solution_filename = f"data/chains_same_few/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            # Save cot prompt to file
            prompt_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/prompts/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_chain_prompt_cot)

            # Write new cot graph to file
            solution_filename = f"data/chains_same_cot/{final_task}/{task}/{chain_length}/{n+1}/solutions/solution_{i}.graphml"
            nx.write_graphml(solution, solution_filename)

            #print('Solution:', new_graph_str)
        else:
            print("Final task not recognized. Exiting.")
            sys.exit(1)

    return chain_graph
"""
"""
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

def graph_to_string_encoder_abductive(graph, graph_type='adjacency_matrix', encoding_dict=None, error_analysis=None):

    # create a list of 50 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul", "Matthew", "Laura", "Kevin", "Amy", "Jason", "Michelle", "Brian", "Angela", "Johnathan", "Stephanie", "Andrew", "Rebecca", "Steven", "Nicole", "Timothy", "Kimberly", "Anthony", "Christine", "Jonathan", "Melissa", "Ryan", "Amanda", "Jeremy", "Samantha", "Benjamin", "Emily"] 
    
    node_to_name = {}

    # map from node ID to name depending on the encoding
    for n in range(50):
        node_to_name[n] = {}
        node_to_name[n]["social_network"] = names[n]

    if graph_type == "adjacency_matrix" or graph_type == "adjacency":
        return str(nx.adjacency_matrix(graph).todense()) + "\n"
    else:
        # construct encoding_graph_str
        encoding_graph_str = ''

        for node in graph.nodes:
            # get the neighbors of the node
            neighbors = sorted(list(graph.neighbors(node)))

            if graph_type == "incident":
                if len(neighbors) == 1:
                    encoding_graph_str += f"Node {node} is connected to node {neighbors[0]}.\n"
                elif len(neighbors) > 1:
                    # convert the neighbors list into a string separated by commas
                    neighbors_str = ', '.join([str(n) for n in neighbors])
                    encoding_graph_str += f"Node {node} is connected to nodes {neighbors_str}.\n"
            #elif graph_type == "coauthorship":
            #    for neighbor in neighbors:
            #        encoding_graph_str += f"{node_to_name[int(node)]["coauthorship"]} and {node_to_name[int(neighbor)]["coauthorship"]} wrote a paper together.\n"
            elif graph_type == "friendship":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["friendship"]} and {node_to_name[int(neighbor)]["friendship"]} are friends.\n"
            elif graph_type == "social_network":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["social_network"]} and {node_to_name[int(neighbor)]["social_network"]} are connected.\n"
            elif graph_type == "expert":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["expert"]} -> {node_to_name[int(neighbor)]["expert"]}\n"
            elif graph_type == "politician":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["politician"]} and {node_to_name[int(neighbor)]["politician"]} are connected.\n"
            elif graph_type == "got":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["got"]} and {node_to_name[int(neighbor)]["got"]} are friends.\n"
            elif graph_type == "sp":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["sp"]} and {node_to_name[int(neighbor)]["sp"]} are friends.\n"
            elif graph_type == "adjacency_list":
                for neighbor in neighbors:
                    encoding_graph_str += f"({node}, {neighbor})\n"

        return encoding_graph_str

"""
def graph_to_string_encoder(graph, graph_type='adjacency_matrix', encoding_dict=None, error_analysis=None):

    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

    if graph_type == "adjacency_matrix" or graph_type == "adjacency":
        return str(nx.adjacency_matrix(graph).todense()) + "\n"
    else:
        # construct encoding_graph_str
        encoding_graph_str = ''

        for node in graph.nodes:
            # get the neighbors of the node
            neighbors = sorted(list(graph.neighbors(node)))

            if graph_type == "incident":
                if len(neighbors) == 1:
                    encoding_graph_str += f"Node {node} is connected to node {neighbors[0]}.\n"
                elif len(neighbors) > 1:
                    # convert the neighbors list into a string separated by commas
                    neighbors_str = ', '.join([str(n) for n in neighbors])
                    encoding_graph_str += f"Node {node} is connected to nodes {neighbors_str}.\n"
            elif graph_type == "coauthorship":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["coauthorship"]} and {node_to_name[int(neighbor)]["coauthorship"]} wrote a paper together.\n"
            elif graph_type == "friendship":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["friendship"]} and {node_to_name[int(neighbor)]["friendship"]} are friends.\n"
            elif graph_type == "social_network":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["social_network"]} and {node_to_name[int(neighbor)]["social_network"]} are connected.\n"
            elif graph_type == "expert":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["expert"]} -> {node_to_name[int(neighbor)]["expert"]}\n"
            elif graph_type == "politician":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["politician"]} and {node_to_name[int(neighbor)]["politician"]} are connected.\n"
            elif graph_type == "got":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["got"]} and {node_to_name[int(neighbor)]["got"]} are friends.\n"
            elif graph_type == "sp":
                for neighbor in neighbors:
                    encoding_graph_str += f"{node_to_name[int(node)]["sp"]} and {node_to_name[int(neighbor)]["sp"]} are friends.\n"
            elif graph_type == "adjacency_list":
                for neighbor in neighbors:
                    encoding_graph_str += f"({node}, {neighbor})\n"

        return encoding_graph_str

def graph_to_prompts(graph, i):
    encodings = ["adjacency_matrix", "adjacency_list", "incident", "coauthorship", "friendship", "social_network", "expert", "politician", "got", "sp"]
    
    # create a list of 20 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # map from node ID to name depending on the encoding
    for n in nodes:
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

    def graph_to_init_prompt(nodes, encoding, node_to_name):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"G describes an undirected graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            return f"G describes an undirected co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given an undirected graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        
    node_for_node_degree = random.choice(nodes)
    
    random_nodes = random.sample(list(graph.nodes()), 2)
    node_a, node_b = random_nodes

    # Select one node from the graph that has at least one neighbor
    nodes_with_neighbors = [node for node in graph.nodes() if graph.degree[node] > 0]
    node_for_connected_nodes = random.choice(nodes_with_neighbors)

    relevant_nodes = [node_for_node_degree, node_a, node_b, node_for_connected_nodes]

    def question_prompt(task, encoding, relevant_nodes):
        if task == "node_count":
            return f"Q: How many nodes are in this graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many undirected edges are in this graph?\nA: "
        elif task == "node_degree":
            if encoding == "adjacency_list" or encoding == "adjacency_matrix":
                return f"Q: How many neighbors does node {relevant_nodes[0]} have?\nA: "
            elif encoding == "incident":
                return f"Q: How many neighbors does node {relevant_nodes[0]} have?\nA: "
            elif encoding == "coauthorship":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['coauthorship']} have?\nA: "
            elif encoding == "friendship":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['friendship']} have?\nA: "
            elif encoding == "social_network":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['social_network']} have?\nA: "
            elif encoding == "expert":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['expert']} have?\nA: "
            elif encoding == "politician":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['politician']} have?\nA: "
            elif encoding == "got":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['got']} have?\nA: "
            elif encoding == "sp":
                return f"Q: How many neighbors does node {node_to_name[relevant_nodes[0]]['sp']} have?\nA: "
        elif task == "edge_exists":
            if encoding == "adjacency_list" or encoding == "adjacency_matrix":
                return f"Q: Is node {relevant_nodes[1]} connected to node {relevant_nodes[2]}?\nA: "
            elif encoding == "incident":
                return f"Q: Is node {relevant_nodes[1]} connected to node {relevant_nodes[2]}?\nA: "
            elif encoding == "coauthorship":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['coauthorship']} connected to node {node_to_name[relevant_nodes[2]]['coauthorship']}?\nA: "
            elif encoding == "friendship":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['friendship']} connected to node {node_to_name[relevant_nodes[2]]['friendship']}?\nA: "
            elif encoding == "social_network":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['social_network']} connected to node {node_to_name[relevant_nodes[2]]['social_network']}?\nA: "
            elif encoding == "expert":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['expert']} connected to node {node_to_name[relevant_nodes[2]]['expert']}?\nA: "
            elif encoding == "politician":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['politician']} connected to node {node_to_name[relevant_nodes[2]]['politician']}?\nA: "
            elif encoding == "got":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['got']} connected to node {node_to_name[relevant_nodes[2]]['got']}?\nA: "
            elif encoding == "sp":
                return f"Q: Is node {node_to_name[relevant_nodes[1]]['sp']} connected to node {node_to_name[relevant_nodes[2]]['sp']}?\nA: "
        elif task == "connected_nodes":
            if encoding == "adjacency_list" or encoding == "adjacency_matrix" or encoding == "incident":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['adjacency_list']}.\nA: "
            elif encoding == "coauthorship":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['coauthorship']}.\nA: "
            elif encoding == "friendship":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['friendship']}.\nA: "
            elif encoding == "social_network":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['social_network']}.\nA: "
            elif encoding == "expert":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['expert']}.\nA: "
            elif encoding == "politician":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['politician']}.\nA: "
            elif encoding == "got":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['got']}.\nA: "
            elif encoding == "sp":
                return f"Q: List all neighbors of node {node_to_name[int(relevant_nodes[-1])]['sp']}.\nA: "
        elif task == "cycle":
            return f"Q: Does the graph contain a cycle?\nA: "
        
    def graph_and_task_to_solution(graph, task, encoding, relevant_nodes):
        if task == "node_count":
            return str(graph.number_of_nodes())
        elif task == "edge_count":
            return str(graph.number_of_edges())
        elif task == "node_degree":
            return str(graph.degree[relevant_nodes[0]])
        elif task == "edge_exists":
            return "Yes" if graph.has_edge(relevant_nodes[1], relevant_nodes[2]) else "No"
        elif task == "connected_nodes":
            if encoding == "adjacency_list" or encoding == "adjacency_matrix":
                return str(sorted([int(node_b) for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "incident":
                return str(sorted([int(node_b) for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "coauthorship":
                return str(sorted([node_to_name[int(node_b)]["coauthorship"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "friendship":
                return str(sorted([node_to_name[int(node_b)]["friendship"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "social_network":
                return str(sorted([node_to_name[int(node_b)]["social_network"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "expert":
                return str(sorted([node_to_name[int(node_b)]["expert"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "politician":
                return str(sorted([node_to_name[int(node_b)]["politician"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "got":
                return str(sorted([node_to_name[int(node_b)]["got"] for node_b in graph.neighbors(relevant_nodes[-1])]))
            elif encoding == "sp":
                return str(sorted([node_to_name[int(node_b)]["sp"] for node_b in graph.neighbors(relevant_nodes[-1])]))
        elif task == "cycle":
            try:
                nx.find_cycle(graph)
                return "Yes"
            except nx.NetworkXNoCycle:
                return "No"
        
    # generate all prompts
    for task in ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "cycle"]:
        for encoding in encodings:
            print(f"Generating prompts for task {task} and encoding {encoding}...")
            # construct example prompts for few + cot

            # construct init_prompt
            init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name)

            # construct graph string
            graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

            # construct question prompt
            question_prompt_str = question_prompt(task, encoding, relevant_nodes)

            # construct solution
            solution = graph_and_task_to_solution(graph, task, encoding, relevant_nodes)

            # construct full prompt
            full_prompt = init_prompt + graph_string + question_prompt_str # + solution + cot

            # Write graph to file
            graph_filename = f"data/{encoding}/input_graphs/{i}.graphml"
            nx.write_graphml(graph, graph_filename)

            # Write prompt to file
            prompt_filename = f"data/{encoding}/prompts/{task}/prompt_{i}.txt"
            with open(prompt_filename, "w") as prompt_file:
                prompt_file.write(full_prompt)

            # Write solution to file
            solution_filename = f"data/{encoding}/solutions/{task}/solution_{i}.txt"
            with open(solution_filename, "w") as solution_file:
                solution_file.write(solution)

def graph_to_prompts_chain(graph, i, max_chain_length):
    encodings = ["adjacency_matrix"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["print_graph"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

        remove_node_mapping[n] = {0: n}
        mix_node_mapping[n] = {0: n}

    all_nodes = list(range(25))
    all_nodes_remove = list(range(25))
    all_nodes_mix = list(range(25))

    def graph_to_init_prompt(nodes, encoding, node_to_name):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"G describes a graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            return f"G describes a co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes a friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given a graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        else:
            print("Encoding not recognized. Exiting.")
            sys.exit(1)

    #add_edge = lambda graph: nx.add_edge(graph, random.choice(list(graph.nodes())), random.choice(list(graph.nodes())))
    add_edge_graph = graph.copy()
    remove_edge_graph = graph.copy()
    add_node_graph = graph.copy()
    remove_node_graph = graph.copy()
    mix_graph = graph.copy()

    relavent_nodes_mod = {
        "add_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "add_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "mix": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
    }

    relavent_nodes_final = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        relavent_nodes_final[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            relavent_nodes_final[modification][chain_length] = {}
            for final_task in final_tasks:
                relavent_nodes_final[modification][chain_length][final_task] = 0

    solutions = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        solutions[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            solutions[modification][chain_length] = {}
            for task in final_tasks:
                solutions[modification][chain_length][task] = 0

    for chain_length in range(1, max_chain_length+1):
        # -------------------------- #
        # Save relevant modification nodes

        # get "add edge" nodes
        unconnected_nodes = []
        for node_a in add_edge_graph.nodes():
            for node_b in add_edge_graph.nodes():
                if node_a != node_b and not add_edge_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                    unconnected_nodes.append((node_a, node_b))
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]
        #add_edge_nodes.append([node_a, node_b])
        relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]

        # get "add edge" solutions
        add_edge_graph.add_edge(node_a, node_b)

        # get "remove edge" nodes
        edge = random.choice(list(remove_edge_graph.edges()))
        #remove_edge_nodes.append(edge)
        relavent_nodes_mod["remove_edge"][chain_length] = edge
        remove_edge_graph.remove_edge(edge[0], edge[1])

        # get "add node" node
        add_node_graph.add_node(max(add_node_graph.nodes()) + 1)
        #add_node_nodes.append(max(add_node_graph.nodes()))
        relavent_nodes_mod["add_node"][chain_length] = max(add_node_graph.nodes())

        # get "remove node" node
        node_for_removal = random.choice(list(remove_node_graph.nodes()))
        #remove_node_nodes.append(node_for_removal)
        relavent_nodes_mod["remove_node"][chain_length] = node_for_removal
        remove_node_graph.remove_node(node_for_removal)
        #print(f"Remove Node for removal: {node_for_removal}")

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #graph = nx.relabel_nodes(graph, mapping)
        #print(f"Mapping: {mapping}")
        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")

        for node in remove_node_graph.nodes():
            #print(f"Node {node} is being renumbered to {mapping[node]}")
            remove_node_mapping[node][chain_length] = mapping[node]

        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v}")

        # Renumber the nodes to be consecutive
        #mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #remove_node_graph = nx.relabel_nodes(remove_node_graph, mapping)

        # get "mix" task
        mix_task = random.choice(modifications[:-1])

        # check if remove edge works
        if mix_task == "remove_edge":
            # if there are no more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == 0:
                mix_task = random.choice(["add_edge", "add_node", "remove_node"]) 

        #print(f"Mix task: {mix_task}")        

        if mix_task == "add_edge":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in mix_graph.nodes():
                for node_b in mix_graph.nodes():
                    if node_a != node_b and not mix_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            mix_nodes = [node_a, node_b]
            mix_graph.add_edge(node_a, node_b)
        elif mix_task == "remove_edge":
            edge = random.choice(list(mix_graph.edges()))
            mix_nodes = edge
            mix_graph.remove_edge(edge[0], edge[1])
        elif mix_task == "add_node":
            #print(f"Nodes before addition: {mix_graph.nodes()}")
            mix_graph.add_node(max(mix_graph.nodes()) + 1)
            #print(f"Node for addition: {max(mix_graph.nodes())}")
            #print(f"Node for addition + 1: {max(mix_graph.nodes())+1}")
            #print(f"Nodes after addition: {mix_graph.nodes()}")
            mix_nodes = max(mix_graph.nodes())
        elif mix_task == "remove_node":
            node_for_removal = random.choice(list(mix_graph.nodes()))
            #print(f"Mix Node for removal: {node_for_removal}")
            mix_nodes = node_for_removal
            mix_graph.remove_node(node_for_removal)

        #for node, v in mix_node_mapping.items():
        #    print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")
            

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(mix_graph.nodes))}
        for node in mix_graph:
            #print(f"Node {node} is being renumbered to {mapping[node]} at chain length {chain_length}")
            mix_node_mapping[node][chain_length] = mapping[node]

        #for node, v in mix_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v} at chain length {chain_length}")

        #print()

        relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

        # -------------------------- #
        # Save relevant final nodes

        # Save releveant nodes for final task on our add edge graph
        relavent_nodes_final["add_edge"][chain_length]["node_degree"] = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["edge_exists"] = random.sample(list(add_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_edge_graph.nodes() if add_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove edge graph
        relavent_nodes_final["remove_edge"][chain_length]["node_degree"] = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["edge_exists"] = random.sample(list(remove_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #print(graph_to_string_encoder(remove_edge_graph))
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our add node graph
        relavent_nodes_final["add_node"][chain_length]["node_degree"] = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["edge_exists"] = random.sample(list(add_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_node_graph.nodes() if add_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove node graph
        relavent_nodes_final["remove_node"][chain_length]["node_degree"] = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["edge_exists"] = random.sample(list(remove_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_node_graph.nodes() if remove_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our mix graph
        relavent_nodes_final["mix"][chain_length]["node_degree"] = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["edge_exists"] = random.sample(list(mix_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in mix_graph.nodes() if mix_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["connected_nodes"] = node_deg

        # -------------------------- #
        # Save solutions

        # Save solutions for add edge graph
        solutions["add_edge"][chain_length]["node_count"] = str(add_edge_graph.number_of_nodes())
        solutions["add_edge"][chain_length]["edge_count"] = str(add_edge_graph.number_of_edges())
        solutions["add_edge"][chain_length]["node_degree"] = str(add_edge_graph.degree[relavent_nodes_final["add_edge"][chain_length]["node_degree"]])
        solutions["add_edge"][chain_length]["edge_exists"] = "Yes" if add_edge_graph.has_edge(relavent_nodes_final["add_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["add_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_edge_graph.neighbors(relavent_nodes_final["add_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_edge_graph)
            solutions["add_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_edge"][chain_length]["cycle"] = "No"
        solutions["add_edge"][chain_length]["print_graph"] = add_edge_graph.copy()

        # Save solutions for remove edge graph
        solutions["remove_edge"][chain_length]["node_count"] = str(remove_edge_graph.number_of_nodes())
        solutions["remove_edge"][chain_length]["edge_count"] = str(remove_edge_graph.number_of_edges())
        solutions["remove_edge"][chain_length]["node_degree"] = str(remove_edge_graph.degree[relavent_nodes_final["remove_edge"][chain_length]["node_degree"]])
        solutions["remove_edge"][chain_length]["edge_exists"] = "Yes" if remove_edge_graph.has_edge(relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["remove_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_edge_graph.neighbors(relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_edge_graph)
            solutions["remove_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_edge"][chain_length]["cycle"] = "No"     
        solutions["remove_edge"][chain_length]["print_graph"] = remove_edge_graph.copy()

        # Save solutions for add node graph
        solutions["add_node"][chain_length]["node_count"] = str(add_node_graph.number_of_nodes())
        solutions["add_node"][chain_length]["edge_count"] = str(add_node_graph.number_of_edges())
        solutions["add_node"][chain_length]["node_degree"] = str(add_node_graph.degree[relavent_nodes_final["add_node"][chain_length]["node_degree"]])
        solutions["add_node"][chain_length]["edge_exists"] = "Yes" if add_node_graph.has_edge(relavent_nodes_final["add_node"][chain_length]["edge_exists"][0], relavent_nodes_final["add_node"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_node_graph.neighbors(relavent_nodes_final["add_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_node_graph)
            solutions["add_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_node"][chain_length]["cycle"] = "No"   
        solutions["add_node"][chain_length]["print_graph"] = add_node_graph.copy()

        # Save solutions for remove node graph TODO: fix this
        solutions["remove_node"][chain_length]["node_count"] = str(remove_node_graph.number_of_nodes())
        solutions["remove_node"][chain_length]["edge_count"] = str(remove_node_graph.number_of_edges())
        solutions["remove_node"][chain_length]["node_degree"] = str(remove_node_graph.degree[relavent_nodes_final["remove_node"][chain_length]["node_degree"]])
        solutions["remove_node"][chain_length]["edge_exists"] = "Yes" if remove_node_graph.has_edge(relavent_nodes_final["remove_node"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_node"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([remove_node_mapping[int(node_b)][chain_length] for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_node_graph)
            solutions["remove_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_node"][chain_length]["cycle"] = "No"    
        solutions["remove_node"][chain_length]["print_graph"] = remove_node_graph.copy()

        # Save solutions for mix graph TODO: fix this
        solutions["mix"][chain_length]["node_count"] = str(mix_graph.number_of_nodes())
        solutions["mix"][chain_length]["edge_count"] = str(mix_graph.number_of_edges())
        solutions["mix"][chain_length]["node_degree"] = str(mix_graph.degree[relavent_nodes_final["mix"][chain_length]["node_degree"]])
        solutions["mix"][chain_length]["edge_exists"] = "Yes" if mix_graph.has_edge(relavent_nodes_final["mix"][chain_length]["edge_exists"][0], relavent_nodes_final["mix"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([mix_node_mapping[int(node_b)][chain_length] for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(mix_graph)
            solutions["mix"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["mix"][chain_length]["cycle"] = "No"
        solutions["mix"][chain_length]["print_graph"] = mix_graph.copy()

    def modification_prompt(encoding, modification, chain_num, relevant_nodes_mod):
        #print(f"Generating modification prompt for encoding {encoding}, modification {modification}, chain length {chain_length}...")
        modification_prompt = "Perform the following operations on the graph:\n"
        for mod_number in range(1, chain_num+1):
            node = relevant_nodes_mod[modification][mod_number]
            #print(f"relevant_nodes_mod[{modification}][{mod_number}]: {node}")
            if modification == "add_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "add_node":
                if encoding == "adjacency_matrix":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                elif encoding in ["adjacency_list", "incident"]:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                else:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_node":
                if mod_number == 1:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the graph.\n"
                else:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
            elif modification == "mix":
                mod = node[0]
                mix_node = node[1]
                if mod == "add_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "add_node":
                    if encoding == "adjacency_matrix":
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                    elif encoding in ["adjacency_list", "incident"]:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                    else:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_node":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                        else:   
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the graph.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
        return modification_prompt
    
    def question_prompt(task, encoding, modification, chain_num, relevant_nodes_final):
        node = relevant_nodes_final[modification][chain_num][task]
        if task == "node_count":
            return f"Q: How many nodes are in the final resulting graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many edges are in the final resulting graph?\nA: "
        elif task == "node_degree":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    #print(f"In question remove, node: {node}, gets mapped to {remove_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {remove_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                elif modification == "mix":
                    #print(f"In question mix, node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {mix_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                else:
                    return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
            else:
                return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
        elif task == "edge_exists":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: Is node {remove_node_mapping[node[0]][chain_num]} connected to node {remove_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                elif modification == "mix":
                    return f"Q: Is node {mix_node_mapping[node[0]][chain_num]} connected to node {mix_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                else:
                    return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
            else:
                return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
        elif task == "connected_nodes":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: List all neighbors of node {remove_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                elif modification == "mix":
                    #print(f"Node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: List all neighbors of node {mix_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                else:
                    return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
            else:
                return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
        elif task == "cycle":
            return f"Q: Does the final resulting graph contain a cycle?\nA: "
        elif task == "print_graph":
            if encoding == "adjacency_matrix":
                return f"Q: What is the final resulting adjacency matrix? For each operation, write out the entire resulting adjacency matrix. Write out the entire final resulting adjacency matrix. \nA: "
            else:
                return f"Q: What is the final resulting graph? Present the graph in the same structure as above, and write out the entire resulting graph.\nA: "
        else:
            print("Task not recognized. Exiting.")
            sys.exit(1)
   
    # generate all prompts
    for task in final_tasks:
        for modification in modifications:
            for encoding in encodings:
                for chain_length in range(1, max_chain_length+1):
                    #if chain_length == 5 and encoding == "adjacency_matrix" and modification == "remove_node" and task == "edge_count":
                    print(f"Generating prompts for task {task}, modification {modification}, encoding {encoding}, and chain length {chain_length}...")
                    # construct example prompts for few + cot

                    # construct init_prompt
                    init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name)

                    # construct graph string
                    graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

                    # construct modification prompt
                    modification_prompt_str = modification_prompt(encoding, modification, chain_length, relavent_nodes_mod)

                    # construct question prompt
                    question_prompt_str = question_prompt(task, encoding, modification, chain_length, relavent_nodes_final)

                    # construct solution
                    solution = solutions[modification][chain_length][task]

                    if task == "connected_nodes" and encoding == "adjacency_matrix":
                        if modification == "remove_node":
                            solution = [remove_node_mapping[node][chain_length] for node in solution]
                        elif modification == "mix":
                            solution = [mix_node_mapping[node][chain_length] for node in solution]
                    # construct full prompt
                    full_prompt = init_prompt + graph_string + modification_prompt_str + question_prompt_str # + solution + cot

                    #print(f"Full prompt: {full_prompt}")
                    #print(f"Solution: {solution}")

                    # Write graph to file
                    graph_filename = f"data/{encoding}_chain_big/{task}/{modification}/{chain_length}/input_graphs/{i}.graphml"
                    nx.write_graphml(graph, graph_filename)

                    # Write prompt to file
                    prompt_filename = f"data/{encoding}_chain_big/{task}/{modification}/{chain_length}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(full_prompt)

                    if task == "print_graph":
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big/{task}/{modification}/{chain_length}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)
                    elif task == "connected_nodes":
                        # Use node_to_name on list
                        solution = [node_to_name[node][encoding] for node in solution]

                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))
                    else:
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(solution)

def graph_to_prompts_chain_no_print(graph, i, max_chain_length, density=None, size=None):
    encodings = ["adjacency_matrix"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["triangle", "isolated"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

        remove_node_mapping[n] = {0: n}
        mix_node_mapping[n] = {0: n}

    all_nodes = list(range(25))
    all_nodes_remove = list(range(25))
    all_nodes_mix = list(range(25))

    def graph_to_init_prompt(nodes, encoding, node_to_name):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"G describes a graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            return f"G describes a co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes a friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given a graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        else:
            print("Encoding not recognized. Exiting.")
            sys.exit(1)

    #add_edge = lambda graph: nx.add_edge(graph, random.choice(list(graph.nodes())), random.choice(list(graph.nodes())))
    add_edge_graph = graph.copy()
    remove_edge_graph = graph.copy()
    add_node_graph = graph.copy()
    remove_node_graph = graph.copy()
    mix_graph = graph.copy()

    relavent_nodes_mod = {
        "add_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "add_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "mix": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
    }

    relavent_nodes_final = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        relavent_nodes_final[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            relavent_nodes_final[modification][chain_length] = {}
            for final_task in final_tasks:
                relavent_nodes_final[modification][chain_length][final_task] = 0

    solutions = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        solutions[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            solutions[modification][chain_length] = {}
            for task in final_tasks:
                solutions[modification][chain_length][task] = 0

    # we will keep track of the triangles in the network after each edge is added
    add_unique_triangles = set()
    mix_unique_triangles = set()

    for chain_length in range(1, max_chain_length+1):
        # -------------------------- #
        # Save relevant modification nodes

        # get "add edge" nodes
        unconnected_nodes = []
        for node_a in add_edge_graph.nodes():
            for node_b in add_edge_graph.nodes():
                if node_a != node_b and not add_edge_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                    unconnected_nodes.append((node_a, node_b))
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]
        #add_edge_nodes.append([node_a, node_b])
        relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]

        # get "add edge" solutions
        add_edge_graph.add_edge(node_a, node_b)

        # get "remove edge" nodes
        edge = random.choice(list(remove_edge_graph.edges()))
        #remove_edge_nodes.append(edge)
        relavent_nodes_mod["remove_edge"][chain_length] = edge
        remove_edge_graph.remove_edge(edge[0], edge[1])

        # get "add node" node
        add_node_graph.add_node(max(add_node_graph.nodes()) + 1)
        #add_node_nodes.append(max(add_node_graph.nodes()))
        relavent_nodes_mod["add_node"][chain_length] = max(add_node_graph.nodes())

        # get "remove node" node
        node_for_removal = random.choice(list(remove_node_graph.nodes()))
        #remove_node_nodes.append(node_for_removal)
        relavent_nodes_mod["remove_node"][chain_length] = node_for_removal
        remove_node_graph.remove_node(node_for_removal)
        #print(f"Remove Node for removal: {node_for_removal}")

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #graph = nx.relabel_nodes(graph, mapping)
        #print(f"Mapping: {mapping}")
        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")

        for node in remove_node_graph.nodes():
            #print(f"Node {node} is being renumbered to {mapping[node]}")
            remove_node_mapping[node][chain_length] = mapping[node]

        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v}")

        # Renumber the nodes to be consecutive
        #mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #remove_node_graph = nx.relabel_nodes(remove_node_graph, mapping)

        # get "mix" task
        mix_task = random.choice(modifications[:-1])

        # check if remove edge works
        if mix_task == "remove_edge":
            # if there are no more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == 0:
                mix_task = random.choice(["add_edge", "add_node", "remove_node"]) 

        #print(f"Mix task: {mix_task}")        

        if mix_task == "add_edge":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in mix_graph.nodes():
                for node_b in mix_graph.nodes():
                    if node_a != node_b and not mix_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            mix_nodes = [node_a, node_b]
            mix_graph.add_edge(node_a, node_b)
        elif mix_task == "remove_edge":
            edge = random.choice(list(mix_graph.edges()))
            mix_nodes = edge
            mix_graph.remove_edge(edge[0], edge[1])
        elif mix_task == "add_node":
            #print(f"Nodes before addition: {mix_graph.nodes()}")
            mix_graph.add_node(max(mix_graph.nodes()) + 1)
            #print(f"Node for addition: {max(mix_graph.nodes())}")
            #print(f"Node for addition + 1: {max(mix_graph.nodes())+1}")
            #print(f"Nodes after addition: {mix_graph.nodes()}")
            mix_nodes = max(mix_graph.nodes())
        elif mix_task == "remove_node":
            node_for_removal = random.choice(list(mix_graph.nodes()))
            #print(f"Mix Node for removal: {node_for_removal}")
            mix_nodes = node_for_removal
            mix_graph.remove_node(node_for_removal)

        #for node, v in mix_node_mapping.items():
        #    print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")
            

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(mix_graph.nodes))}
        for node in mix_graph:
            #print(f"Node {node} is being renumbered to {mapping[node]} at chain length {chain_length}")
            mix_node_mapping[node][chain_length] = mapping[node]

        #for node, v in mix_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v} at chain length {chain_length}")

        #print()

        relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

        # -------------------------- #
        # Save relevant final nodes

        # Save releveant nodes for final task on our add edge graph
        relavent_nodes_final["add_edge"][chain_length]["node_degree"] = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["edge_exists"] = random.sample(list(add_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_edge_graph.nodes() if add_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove edge graph
        relavent_nodes_final["remove_edge"][chain_length]["node_degree"] = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["edge_exists"] = random.sample(list(remove_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #print(graph_to_string_encoder(remove_edge_graph))
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our add node graph
        relavent_nodes_final["add_node"][chain_length]["node_degree"] = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["edge_exists"] = random.sample(list(add_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_node_graph.nodes() if add_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove node graph
        relavent_nodes_final["remove_node"][chain_length]["node_degree"] = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["edge_exists"] = random.sample(list(remove_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_node_graph.nodes() if remove_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our mix graph
        relavent_nodes_final["mix"][chain_length]["node_degree"] = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["edge_exists"] = random.sample(list(mix_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in mix_graph.nodes() if mix_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["connected_nodes"] = node_deg

        # -------------------------- #
        # Save solutions

        # Save solutions for add edge graph
        solutions["add_edge"][chain_length]["node_count"] = str(add_edge_graph.number_of_nodes())
        solutions["add_edge"][chain_length]["edge_count"] = str(add_edge_graph.number_of_edges())
        solutions["add_edge"][chain_length]["node_degree"] = str(add_edge_graph.degree[relavent_nodes_final["add_edge"][chain_length]["node_degree"]])
        solutions["add_edge"][chain_length]["edge_exists"] = "Yes" if add_edge_graph.has_edge(relavent_nodes_final["add_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["add_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_edge_graph.neighbors(relavent_nodes_final["add_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_edge_graph)
            solutions["add_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_edge"][chain_length]["cycle"] = "No"
        solutions["add_edge"][chain_length]["print_graph"] = add_edge_graph.copy()

        # count the number of triangles in ego_graph
        all_cliques = nx.enumerate_all_cliques(add_edge_graph)
        triangles = set(tuple(clique) for clique in all_cliques if len(clique) == 3)
        add_unique_triangles.update(triangles)
        add_total_triangles = len(add_unique_triangles)
        solutions["add_edge"][chain_length]["triangle"] = str(add_total_triangles)

        add_edge_isolated_nodes = [node for node in add_edge_graph.nodes() if add_edge_graph.degree(node) == 0]
        if len(add_edge_isolated_nodes) == 0:
            solutions["add_edge"][chain_length]["isolated"] = "None"
        elif len(add_edge_isolated_nodes) == 1:
            solutions["add_edge"][chain_length]["isolated"] = str(add_edge_isolated_nodes[0])
        else:
            solutions["add_edge"][chain_length]["isolated"] = ", ".join([str(node) for node in add_edge_isolated_nodes])

        # Save solutions for remove edge graph
        solutions["remove_edge"][chain_length]["node_count"] = str(remove_edge_graph.number_of_nodes())
        solutions["remove_edge"][chain_length]["edge_count"] = str(remove_edge_graph.number_of_edges())
        solutions["remove_edge"][chain_length]["node_degree"] = str(remove_edge_graph.degree[relavent_nodes_final["remove_edge"][chain_length]["node_degree"]])
        solutions["remove_edge"][chain_length]["edge_exists"] = "Yes" if remove_edge_graph.has_edge(relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["remove_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_edge_graph.neighbors(relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_edge_graph)
            solutions["remove_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_edge"][chain_length]["cycle"] = "No"     
        solutions["remove_edge"][chain_length]["print_graph"] = remove_edge_graph.copy()

        remove_edge_isolated_nodes = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree(node) == 0]
        if len(remove_edge_isolated_nodes) == 0:
            solutions["remove_edge"][chain_length]["isolated"] = "None"
        elif len(remove_edge_isolated_nodes) == 1:
            solutions["remove_edge"][chain_length]["isolated"] = str(remove_edge_isolated_nodes[0])
        else:
            solutions["remove_edge"][chain_length]["isolated"] = ", ".join([str(node) for node in remove_edge_isolated_nodes])

        # Save solutions for add node graph
        solutions["add_node"][chain_length]["node_count"] = str(add_node_graph.number_of_nodes())
        solutions["add_node"][chain_length]["edge_count"] = str(add_node_graph.number_of_edges())
        solutions["add_node"][chain_length]["node_degree"] = str(add_node_graph.degree[relavent_nodes_final["add_node"][chain_length]["node_degree"]])
        solutions["add_node"][chain_length]["edge_exists"] = "Yes" if add_node_graph.has_edge(relavent_nodes_final["add_node"][chain_length]["edge_exists"][0], relavent_nodes_final["add_node"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_node_graph.neighbors(relavent_nodes_final["add_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_node_graph)
            solutions["add_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_node"][chain_length]["cycle"] = "No"   
        solutions["add_node"][chain_length]["print_graph"] = add_node_graph.copy()

        add_node_isolated_nodes = [node for node in add_node_graph.nodes() if add_node_graph.degree(node) == 0]
        if len(add_node_isolated_nodes) == 0:
            solutions["add_node"][chain_length]["isolated"] = "None"
        elif len(add_node_isolated_nodes) == 1:
            solutions["add_node"][chain_length]["isolated"] = str(add_node_isolated_nodes[0])
        else:
            solutions["add_node"][chain_length]["isolated"] = ", ".join([str(node) for node in add_node_isolated_nodes])

        # Save solutions for remove node graph TODO: fix this
        solutions["remove_node"][chain_length]["node_count"] = str(remove_node_graph.number_of_nodes())
        solutions["remove_node"][chain_length]["edge_count"] = str(remove_node_graph.number_of_edges())
        solutions["remove_node"][chain_length]["node_degree"] = str(remove_node_graph.degree[relavent_nodes_final["remove_node"][chain_length]["node_degree"]])
        solutions["remove_node"][chain_length]["edge_exists"] = "Yes" if remove_node_graph.has_edge(relavent_nodes_final["remove_node"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_node"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([remove_node_mapping[int(node_b)][chain_length] for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_node_graph)
            solutions["remove_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_node"][chain_length]["cycle"] = "No"    
        solutions["remove_node"][chain_length]["print_graph"] = remove_node_graph.copy()

        remove_node_isolated_nodes = [node for node in remove_node_graph.nodes() if remove_node_graph.degree(node) == 0]
        if len(remove_node_isolated_nodes) == 0:
            solutions["remove_node"][chain_length]["isolated"] = "None"
        elif len(remove_node_isolated_nodes) == 1:
            solutions["remove_node"][chain_length]["isolated"] = str(remove_node_isolated_nodes[0])
        else:
            solutions["remove_node"][chain_length]["isolated"] = ", ".join([str(node) for node in remove_node_isolated_nodes])

        # Save solutions for mix graph TODO: fix this
        solutions["mix"][chain_length]["node_count"] = str(mix_graph.number_of_nodes())
        solutions["mix"][chain_length]["edge_count"] = str(mix_graph.number_of_edges())
        solutions["mix"][chain_length]["node_degree"] = str(mix_graph.degree[relavent_nodes_final["mix"][chain_length]["node_degree"]])
        solutions["mix"][chain_length]["edge_exists"] = "Yes" if mix_graph.has_edge(relavent_nodes_final["mix"][chain_length]["edge_exists"][0], relavent_nodes_final["mix"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([mix_node_mapping[int(node_b)][chain_length] for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(mix_graph)
            solutions["mix"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["mix"][chain_length]["cycle"] = "No"
        solutions["mix"][chain_length]["print_graph"] = mix_graph.copy()

        # count the number of triangles in ego_graph
        all_cliques = nx.enumerate_all_cliques(mix_graph)
        triangles = set(tuple(clique) for clique in all_cliques if len(clique) == 3)
        mix_unique_triangles.update(triangles)
        mix_total_triangles = len(mix_unique_triangles)
        solutions["mix"][chain_length]["triangle"] = str(mix_total_triangles)

        mix_isolated_nodes = [node for node in mix_graph.nodes() if mix_graph.degree(node) == 0]
        if len(mix_isolated_nodes) == 0:
            solutions["mix"][chain_length]["isolated"] = "None"
        elif len(mix_isolated_nodes) == 1:
            solutions["mix"][chain_length]["isolated"] = str(mix_isolated_nodes[0])
        else:
            solutions["mix"][chain_length]["isolated"] = ", ".join([str(node) for node in mix_isolated_nodes])

    def modification_prompt(encoding, modification, chain_num, relevant_nodes_mod):
        #print(f"Generating modification prompt for encoding {encoding}, modification {modification}, chain length {chain_length}...")
        modification_prompt = "Perform the following operations on the graph:\n"
        for mod_number in range(1, chain_num+1):
            node = relevant_nodes_mod[modification][mod_number]
            #print(f"relevant_nodes_mod[{modification}][{mod_number}]: {node}")
            if modification == "add_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "add_node":
                if encoding == "adjacency_matrix":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                elif encoding in ["adjacency_list", "incident"]:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                else:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_node":
                if mod_number == 1:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the graph.\n"
                else:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
            elif modification == "mix":
                mod = node[0]
                mix_node = node[1]
                if mod == "add_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "add_node":
                    if encoding == "adjacency_matrix":
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                    elif encoding in ["adjacency_list", "incident"]:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                    else:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_node":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                        else:   
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the graph.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
        return modification_prompt
    
    def question_prompt(task, encoding, modification, chain_num, relevant_nodes_final):
        node = relevant_nodes_final[modification][chain_num][task]
        if task == "node_count":
            return f"Q: How many nodes are in the final resulting graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many edges are in the final resulting graph?\nA: "
        elif task == "node_degree":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    #print(f"In question remove, node: {node}, gets mapped to {remove_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {remove_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                elif modification == "mix":
                    #print(f"In question mix, node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {mix_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                else:
                    return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
            else:
                return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
        elif task == "edge_exists":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: Is node {remove_node_mapping[node[0]][chain_num]} connected to node {remove_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                elif modification == "mix":
                    return f"Q: Is node {mix_node_mapping[node[0]][chain_num]} connected to node {mix_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                else:
                    return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
            else:
                return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
        elif task == "connected_nodes":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: List all neighbors of node {remove_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                elif modification == "mix":
                    #print(f"Node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: List all neighbors of node {mix_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                else:
                    return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
            else:
                return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
        elif task == "cycle":
            return f"Q: Does the final resulting graph contain a cycle?\nA: "
        elif task == "print_graph":
            if encoding == "adjacency_matrix":
                return f"Q: What is the final resulting adjacency matrix? Write out the entire final resulting adjacency matrix. \nA: "
            else:
                return f"Q: What is the final resulting graph? Present the graph in the same structure as above, and write out the entire resulting graph.\nA: "
        elif task == "triangle":
            return f"Q: How many total triangles were formed throughout the history of the graph, including the triangles present in the original unmodified graph?\nA: "
        elif task == "isolated":
            return f"Q: List all isolated nodes in the resulting graph.\nA: "
        else:
            print("Task not recognized. Exiting.")
            sys.exit(1)
   
    # generate all prompts
    for task in final_tasks:
        for modification in modifications:
            for encoding in encodings:
                for chain_length in range(1, max_chain_length+1):
                    #if chain_length == 5 and encoding == "adjacency_matrix" and modification == "remove_node" and task == "edge_count":
                    print(f"Generating prompts for task {task}, modification {modification}, encoding {encoding}, and chain length {chain_length}...")
                    # construct example prompts for few + cot

                    # construct init_prompt
                    init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name)

                    # construct graph string
                    graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

                    # construct modification prompt
                    modification_prompt_str = modification_prompt(encoding, modification, chain_length, relavent_nodes_mod)

                    # construct question prompt
                    question_prompt_str = question_prompt(task, encoding, modification, chain_length, relavent_nodes_final)

                    # construct solution
                    solution = solutions[modification][chain_length][task]

                    if (task in ["connected_nodes"]) and encoding == "adjacency_matrix":
                        if modification == "remove_node":
                            solution = [remove_node_mapping[node][chain_length] for node in solution]
                        elif modification == "mix":
                            solution = [mix_node_mapping[node][chain_length] for node in solution]
                    # construct full prompt
                    full_prompt = init_prompt + graph_string + modification_prompt_str + question_prompt_str # + solution + cot

                    #print(f"Full prompt: {full_prompt}")
                    #print(f"Solution: {solution}")

                    # Write graph to file
                    if density and size:
                        graph_filename = f"data/{encoding}_chain_p/{task}/{modification}/{chain_length}/{density}/{size}/input_graphs/{i}.graphml"
                    else:
                        graph_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/input_graphs/{i}.graphml"
                    nx.write_graphml(graph, graph_filename)

                    # Write prompt to file
                    if density and size:
                        prompt_filename = f"data/{encoding}_chain_p/{task}/{modification}/{chain_length}/{density}/{size}/prompts/prompt_{i}.txt"
                    else:
                        prompt_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(full_prompt)

                    if task == "print_graph":
                        # Write solution to file
                        if density and size:
                            solution_filename = f"data/{encoding}_chain_p/{task}/{modification}/{chain_length}/{density}/{size}/solutions/solution_{i}.graphml"
                        else:
                            solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)
                    elif task == "connected_nodes":
                        # Use node_to_name on list
                        solution = [node_to_name[node][encoding] for node in solution]

                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))
                    else:
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))

def graph_to_prompts_chain_graph_types(graph, graph_type, i, max_chain_length):
    encodings = ["adjacency_matrix", "incident", "coauthorship"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    if graph_type == "empty":
        modifications = ["add_edge", "add_node", "remove_node", "mix"]
    elif graph_type == "complete":
        modifications = ["remove_edge", "add_node", "remove_node", "mix"]
    else:
        modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "print_graph"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

        remove_node_mapping[n] = {0: n}
        mix_node_mapping[n] = {0: n}

    all_nodes = list(range(25))
    all_nodes_remove = list(range(25))
    all_nodes_mix = list(range(25))

    def graph_to_init_prompt(nodes, encoding, node_to_name, graph_type):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            if graph_type == "empty":
                return f"G describes an undirected graph among {nodes_str}.\nInitially, there are no edges in the graph.\n"
            else:
                return f"G describes an undirected graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            if graph_type == "empty":
                return f"G describes an undirected co-authorship graph among {nodes_str}.\nInitially, there are no edges in the graph.\n"
            else:
                return f"G describes an undirected co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given an undirected graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        else:
            print("Encoding not recognized. Exiting.")
            sys.exit(1)

    #add_edge = lambda graph: nx.add_edge(graph, random.choice(list(graph.nodes())), random.choice(list(graph.nodes())))
    add_edge_graph = graph.copy()
    remove_edge_graph = graph.copy()
    add_node_graph = graph.copy()
    remove_node_graph = graph.copy()
    mix_graph = graph.copy()

    relavent_nodes_mod = {
        "add_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "add_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "mix": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
    }

    relavent_nodes_final = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        relavent_nodes_final[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            relavent_nodes_final[modification][chain_length] = {}
            for final_task in final_tasks:
                relavent_nodes_final[modification][chain_length][final_task] = 0

    solutions = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        solutions[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            solutions[modification][chain_length] = {}
            for task in final_tasks:
                solutions[modification][chain_length][task] = 0

    for chain_length in range(1, max_chain_length+1):
        # -------------------------- #
        # Save relevant modification nodes
        if graph_type != "complete":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in add_edge_graph.nodes():
                for node_b in add_edge_graph.nodes():
                    if node_a != node_b and not add_edge_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            #add_edge_nodes.append([node_a, node_b])
            relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]

            # get "add edge" solutions
            add_edge_graph.add_edge(node_a, node_b)

        if graph_type != "empty":
            # get "remove edge" nodes
            edge = random.choice(list(remove_edge_graph.edges()))
            #remove_edge_nodes.append(edge)
            relavent_nodes_mod["remove_edge"][chain_length] = edge
            remove_edge_graph.remove_edge(edge[0], edge[1])

        # get "add node" node
        add_node_graph.add_node(max(add_node_graph.nodes()) + 1)
        #add_node_nodes.append(max(add_node_graph.nodes()))
        relavent_nodes_mod["add_node"][chain_length] = max(add_node_graph.nodes())

        # get "remove node" node
        node_for_removal = random.choice(list(remove_node_graph.nodes()))
        #remove_node_nodes.append(node_for_removal)
        relavent_nodes_mod["remove_node"][chain_length] = node_for_removal
        remove_node_graph.remove_node(node_for_removal)
        #print(f"Remove Node for removal: {node_for_removal}")

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #graph = nx.relabel_nodes(graph, mapping)
        #print(f"Mapping: {mapping}")
        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")

        for node in remove_node_graph.nodes():
            #print(f"Node {node} is being renumbered to {mapping[node]}")
            remove_node_mapping[node][chain_length] = mapping[node]

        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v}")

        # Renumber the nodes to be consecutive
        #mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #remove_node_graph = nx.relabel_nodes(remove_node_graph, mapping)

        # get "mix" task
        mix_task = random.choice(modifications[:-1])

        # check if remove edge works
        if mix_task == "remove_edge":
            # if there are no more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == 0:
                mix_task = random.choice(["add_edge", "add_node", "remove_node"]) 
        if mix_task == "add_edge":
            # if all the nodes are connected and we can't add any more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == len(mix_graph.nodes) * (len(mix_graph.nodes) - 1) / 2:
                mix_task = random.choice(["remove_edge", "add_node", "remove_node"])

        #print(f"Mix task: {mix_task}")        

        if mix_task == "add_edge":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in mix_graph.nodes():
                for node_b in mix_graph.nodes():
                    if node_a != node_b and not mix_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            mix_nodes = [node_a, node_b]
            mix_graph.add_edge(node_a, node_b)
        elif mix_task == "remove_edge":
            edge = random.choice(list(mix_graph.edges()))
            mix_nodes = edge
            mix_graph.remove_edge(edge[0], edge[1])
        elif mix_task == "add_node":
            #print(f"Nodes before addition: {mix_graph.nodes()}")
            mix_graph.add_node(max(mix_graph.nodes()) + 1)
            #print(f"Node for addition: {max(mix_graph.nodes())}")
            #print(f"Node for addition + 1: {max(mix_graph.nodes())+1}")
            #print(f"Nodes after addition: {mix_graph.nodes()}")
            mix_nodes = max(mix_graph.nodes())
        elif mix_task == "remove_node":
            node_for_removal = random.choice(list(mix_graph.nodes()))
            #print(f"Mix Node for removal: {node_for_removal}")
            mix_nodes = node_for_removal
            mix_graph.remove_node(node_for_removal)

        #for node, v in mix_node_mapping.items():
        #    print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")
            

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(mix_graph.nodes))}
        for node in mix_graph:
            #print(f"Node {node} is being renumbered to {mapping[node]} at chain length {chain_length}")
            mix_node_mapping[node][chain_length] = mapping[node]

        #for node, v in mix_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v} at chain length {chain_length}")

        #print()

        relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

        # -------------------------- #
        # Save relevant final nodes
        if graph_type != "complete":
            # Save releveant nodes for final task on our add edge graph
            relavent_nodes_final["add_edge"][chain_length]["node_degree"] = random.choice(list(add_edge_graph.nodes()))
            relavent_nodes_final["add_edge"][chain_length]["edge_exists"] = random.sample(list(add_edge_graph.nodes()), 2)
            #nodes_with_neighbors = [node for node in add_edge_graph.nodes() if add_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
            #node_deg = random.choice(nodes_with_neighbors)
            node_deg = random.choice(list(add_edge_graph.nodes()))
            relavent_nodes_final["add_edge"][chain_length]["connected_nodes"] = node_deg

        if graph_type != "empty":
            # Save releveant nodes for final task on our remove edge graph
            relavent_nodes_final["remove_edge"][chain_length]["node_degree"] = random.choice(list(remove_edge_graph.nodes()))
            relavent_nodes_final["remove_edge"][chain_length]["edge_exists"] = random.sample(list(remove_edge_graph.nodes()), 2)
            #nodes_with_neighbors = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
            #print(graph_to_string_encoder(remove_edge_graph))
            #node_deg = random.choice(nodes_with_neighbors)
            node_deg = random.choice(list(remove_edge_graph.nodes()))
            relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our add node graph
        relavent_nodes_final["add_node"][chain_length]["node_degree"] = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["edge_exists"] = random.sample(list(add_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_node_graph.nodes() if add_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove node graph
        relavent_nodes_final["remove_node"][chain_length]["node_degree"] = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["edge_exists"] = random.sample(list(remove_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_node_graph.nodes() if remove_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our mix graph
        relavent_nodes_final["mix"][chain_length]["node_degree"] = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["edge_exists"] = random.sample(list(mix_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in mix_graph.nodes() if mix_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["connected_nodes"] = node_deg

        # -------------------------- #
        # Save solutions

        if graph_type != "complete":
            # Save solutions for add edge graph
            solutions["add_edge"][chain_length]["node_count"] = str(add_edge_graph.number_of_nodes())
            solutions["add_edge"][chain_length]["edge_count"] = str(add_edge_graph.number_of_edges())
            solutions["add_edge"][chain_length]["node_degree"] = str(add_edge_graph.degree[relavent_nodes_final["add_edge"][chain_length]["node_degree"]])
            solutions["add_edge"][chain_length]["edge_exists"] = "Yes" if add_edge_graph.has_edge(relavent_nodes_final["add_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["add_edge"][chain_length]["edge_exists"][1]) else "No"
            solutions["add_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_edge_graph.neighbors(relavent_nodes_final["add_edge"][chain_length]["connected_nodes"])])
            try:
                nx.find_cycle(add_edge_graph)
                solutions["add_edge"][chain_length]["cycle"] = "Yes"
            except nx.NetworkXNoCycle:
                solutions["add_edge"][chain_length]["cycle"] = "No"
            solutions["add_edge"][chain_length]["print_graph"] = add_edge_graph.copy()

        if graph_type != "empty":
            # Save solutions for remove edge graph
            solutions["remove_edge"][chain_length]["node_count"] = str(remove_edge_graph.number_of_nodes())
            solutions["remove_edge"][chain_length]["edge_count"] = str(remove_edge_graph.number_of_edges())
            solutions["remove_edge"][chain_length]["node_degree"] = str(remove_edge_graph.degree[relavent_nodes_final["remove_edge"][chain_length]["node_degree"]])
            solutions["remove_edge"][chain_length]["edge_exists"] = "Yes" if remove_edge_graph.has_edge(relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][1]) else "No"
            solutions["remove_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_edge_graph.neighbors(relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"])])
            try:
                nx.find_cycle(remove_edge_graph)
                solutions["remove_edge"][chain_length]["cycle"] = "Yes"
            except nx.NetworkXNoCycle:
                solutions["remove_edge"][chain_length]["cycle"] = "No"     
            solutions["remove_edge"][chain_length]["print_graph"] = remove_edge_graph.copy()

        # Save solutions for add node graph
        solutions["add_node"][chain_length]["node_count"] = str(add_node_graph.number_of_nodes())
        solutions["add_node"][chain_length]["edge_count"] = str(add_node_graph.number_of_edges())
        solutions["add_node"][chain_length]["node_degree"] = str(add_node_graph.degree[relavent_nodes_final["add_node"][chain_length]["node_degree"]])
        solutions["add_node"][chain_length]["edge_exists"] = "Yes" if add_node_graph.has_edge(relavent_nodes_final["add_node"][chain_length]["edge_exists"][0], relavent_nodes_final["add_node"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_node_graph.neighbors(relavent_nodes_final["add_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_node_graph)
            solutions["add_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_node"][chain_length]["cycle"] = "No"   
        solutions["add_node"][chain_length]["print_graph"] = add_node_graph.copy()

        # Save solutions for remove node graph TODO: fix this
        solutions["remove_node"][chain_length]["node_count"] = str(remove_node_graph.number_of_nodes())
        solutions["remove_node"][chain_length]["edge_count"] = str(remove_node_graph.number_of_edges())
        solutions["remove_node"][chain_length]["node_degree"] = str(remove_node_graph.degree[relavent_nodes_final["remove_node"][chain_length]["node_degree"]])
        solutions["remove_node"][chain_length]["edge_exists"] = "Yes" if remove_node_graph.has_edge(relavent_nodes_final["remove_node"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_node"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([remove_node_mapping[int(node_b)][chain_length] for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_node_graph)
            solutions["remove_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_node"][chain_length]["cycle"] = "No"    
        solutions["remove_node"][chain_length]["print_graph"] = remove_node_graph.copy()

        # Save solutions for mix graph TODO: fix this
        solutions["mix"][chain_length]["node_count"] = str(mix_graph.number_of_nodes())
        solutions["mix"][chain_length]["edge_count"] = str(mix_graph.number_of_edges())
        solutions["mix"][chain_length]["node_degree"] = str(mix_graph.degree[relavent_nodes_final["mix"][chain_length]["node_degree"]])
        solutions["mix"][chain_length]["edge_exists"] = "Yes" if mix_graph.has_edge(relavent_nodes_final["mix"][chain_length]["edge_exists"][0], relavent_nodes_final["mix"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([mix_node_mapping[int(node_b)][chain_length] for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(mix_graph)
            solutions["mix"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["mix"][chain_length]["cycle"] = "No"
        solutions["mix"][chain_length]["print_graph"] = mix_graph.copy()

    def modification_prompt(encoding, modification, chain_num, relevant_nodes_mod):
        #print(f"Generating modification prompt for encoding {encoding}, modification {modification}, chain length {chain_length}...")
        modification_prompt = "Perform the following operations on the graph:\n"
        for mod_number in range(1, chain_num+1):
            node = relevant_nodes_mod[modification][mod_number]
            #print(f"relevant_nodes_mod[{modification}][{mod_number}]: {node}")
            if modification == "add_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "add_node":
                if encoding == "adjacency_matrix":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                elif encoding in ["adjacency_list", "incident"]:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                else:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_node":
                if mod_number == 1:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the graph.\n"
                else:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
            elif modification == "mix":
                mod = node[0]
                mix_node = node[1]
                if mod == "add_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "add_node":
                    if encoding == "adjacency_matrix":
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                    elif encoding in ["adjacency_list", "incident"]:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                    else:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_node":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                        else:   
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the graph.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
        return modification_prompt
    
    def question_prompt(task, encoding, modification, chain_num, relevant_nodes_final):
        node = relevant_nodes_final[modification][chain_num][task]
        if task == "node_count":
            return f"Q: How many nodes are in the final resulting graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many edges are in the final resulting graph?\nA: "
        elif task == "node_degree":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    #print(f"In question remove, node: {node}, gets mapped to {remove_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {remove_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                elif modification == "mix":
                    #print(f"In question mix, node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {mix_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                else:
                    return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
            else:
                return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
        elif task == "edge_exists":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: Is node {remove_node_mapping[node[0]][chain_num]} connected to node {remove_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                elif modification == "mix":
                    return f"Q: Is node {mix_node_mapping[node[0]][chain_num]} connected to node {mix_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                else:
                    return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
            else:
                return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
        elif task == "connected_nodes":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: List all neighbors of node {remove_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                elif modification == "mix":
                    #print(f"Node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: List all neighbors of node {mix_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                else:
                    return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
            else:
                return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
        elif task == "cycle":
            return f"Q: Does the final resulting graph contain a cycle?\nA: "
        elif task == "print_graph":
            if encoding == "adjacency_matrix":
                return f"Q: What is the final resulting adjacency matrix? Write out the entire final resulting adjacency matrix. \nA: "
            else:
                return f"Q: What is the final resulting graph? Present the graph in the same structure as above, and write out the entire resulting graph.\nA: "
        else:
            print("Task not recognized. Exiting.")
            sys.exit(1)
   
    # generate all prompts
    for task in final_tasks:
        for modification in modifications:
            for encoding in encodings:
                for chain_length in range(1, max_chain_length+1):
                    #if chain_length == 5 and encoding == "adjacency_matrix" and modification == "remove_node" and task == "edge_count":
                    print(f"Generating prompts for task {task}, modification {modification}, encoding {encoding}, and chain length {chain_length}...")
                    # construct example prompts for few + cot

                    # construct init_prompt
                    init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name, graph_type)

                    graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

                    # construct modification prompt
                    modification_prompt_str = modification_prompt(encoding, modification, chain_length, relavent_nodes_mod)

                    # construct question prompt
                    question_prompt_str = question_prompt(task, encoding, modification, chain_length, relavent_nodes_final)

                    # construct solution
                    solution = solutions[modification][chain_length][task]

                    if task == "connected_nodes" and encoding == "adjacency_matrix":
                        if modification == "remove_node":
                            solution = [remove_node_mapping[node][chain_length] for node in solution]
                        elif modification == "mix":
                            solution = [mix_node_mapping[node][chain_length] for node in solution]
                    # construct full prompt
                    full_prompt = init_prompt + graph_string + modification_prompt_str + question_prompt_str # + solution + cot

                    #print(f"Full prompt: {full_prompt}")
                    #print(f"Solution: {solution}")

                    # Write graph to file
                    graph_filename = f"data/{encoding}_chain_big_{graph_type}/{task}/{modification}/{chain_length}/input_graphs/{i}.graphml"
                    nx.write_graphml(graph, graph_filename)

                    # Write prompt to file
                    prompt_filename = f"data/{encoding}_chain_big_{graph_type}/{task}/{modification}/{chain_length}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(full_prompt)

                    if task == "print_graph":
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}/{task}/{modification}/{chain_length}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)
                    elif task == "connected_nodes":
                        # Use node_to_name on list
                        solution = [node_to_name[node][encoding] for node in solution]

                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))
                    else:
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(solution)

def graph_to_prompts_chain_graph_types_no_perform(graph, graph_type, i, max_chain_length):
    encodings = ["adjacency_matrix", "incident", "coauthorship"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    if graph_type == "empty":
        modifications = ["add_edge", "add_node", "remove_node", "mix"]
    elif graph_type == "complete":
        modifications = ["remove_edge", "add_node", "remove_node", "mix"]
    else:
        modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "print_graph"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

        remove_node_mapping[n] = {0: n}
        mix_node_mapping[n] = {0: n}

    all_nodes = list(range(25))
    all_nodes_remove = list(range(25))
    all_nodes_mix = list(range(25))

    def graph_to_init_prompt(nodes, encoding, node_to_name, graph_type):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            if graph_type == "empty":
                return f"G describes an undirected graph among {nodes_str}.\nInitially, there are no edges in the graph.\n"
            else:
                return f"G describes an undirected graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            if graph_type == "empty":
                return f"G describes an undirected co-authorship graph among {nodes_str}.\nInitially, there are no edges in the graph.\n"
            else:
                return f"G describes an undirected co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given an undirected graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes an undirected social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes an undirected friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        else:
            print("Encoding not recognized. Exiting.")
            sys.exit(1)

    #add_edge = lambda graph: nx.add_edge(graph, random.choice(list(graph.nodes())), random.choice(list(graph.nodes())))
    add_edge_graph = graph.copy()
    remove_edge_graph = graph.copy()
    add_node_graph = graph.copy()
    remove_node_graph = graph.copy()
    mix_graph = graph.copy()

    relavent_nodes_mod = {
        "add_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "add_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "mix": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
    }

    relavent_nodes_final = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        relavent_nodes_final[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            relavent_nodes_final[modification][chain_length] = {}
            for final_task in final_tasks:
                relavent_nodes_final[modification][chain_length][final_task] = 0

    solutions = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        solutions[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            solutions[modification][chain_length] = {}
            for task in final_tasks:
                solutions[modification][chain_length][task] = 0

    for chain_length in range(1, max_chain_length+1):
        # -------------------------- #
        # Save relevant modification nodes
        if graph_type != "complete":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in add_edge_graph.nodes():
                for node_b in add_edge_graph.nodes():
                    if node_a != node_b and not add_edge_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            #add_edge_nodes.append([node_a, node_b])
            relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]

            # get "add edge" solutions
            add_edge_graph.add_edge(node_a, node_b)

        if graph_type != "empty":
            # get "remove edge" nodes
            edge = random.choice(list(remove_edge_graph.edges()))
            #remove_edge_nodes.append(edge)
            relavent_nodes_mod["remove_edge"][chain_length] = edge
            remove_edge_graph.remove_edge(edge[0], edge[1])

        # get "add node" node
        add_node_graph.add_node(max(add_node_graph.nodes()) + 1)
        #add_node_nodes.append(max(add_node_graph.nodes()))
        relavent_nodes_mod["add_node"][chain_length] = max(add_node_graph.nodes())

        # get "remove node" node
        node_for_removal = random.choice(list(remove_node_graph.nodes()))
        #remove_node_nodes.append(node_for_removal)
        relavent_nodes_mod["remove_node"][chain_length] = node_for_removal
        remove_node_graph.remove_node(node_for_removal)
        #print(f"Remove Node for removal: {node_for_removal}")

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #graph = nx.relabel_nodes(graph, mapping)
        #print(f"Mapping: {mapping}")
        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")

        for node in remove_node_graph.nodes():
            #print(f"Node {node} is being renumbered to {mapping[node]}")
            remove_node_mapping[node][chain_length] = mapping[node]

        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v}")

        # Renumber the nodes to be consecutive
        #mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #remove_node_graph = nx.relabel_nodes(remove_node_graph, mapping)

        # get "mix" task
        mix_task = random.choice(modifications[:-1])

        # check if remove edge works
        if mix_task == "remove_edge":
            # if there are no more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == 0:
                mix_task = random.choice(["add_edge", "add_node", "remove_node"]) 
        if mix_task == "add_edge":
            # if all the nodes are connected and we can't add any more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == len(mix_graph.nodes) * (len(mix_graph.nodes) - 1) / 2:
                mix_task = random.choice(["remove_edge", "add_node", "remove_node"])

        #print(f"Mix task: {mix_task}")        

        if mix_task == "add_edge":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in mix_graph.nodes():
                for node_b in mix_graph.nodes():
                    if node_a != node_b and not mix_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            mix_nodes = [node_a, node_b]
            mix_graph.add_edge(node_a, node_b)
        elif mix_task == "remove_edge":
            edge = random.choice(list(mix_graph.edges()))
            mix_nodes = edge
            mix_graph.remove_edge(edge[0], edge[1])
        elif mix_task == "add_node":
            #print(f"Nodes before addition: {mix_graph.nodes()}")
            mix_graph.add_node(max(mix_graph.nodes()) + 1)
            #print(f"Node for addition: {max(mix_graph.nodes())}")
            #print(f"Node for addition + 1: {max(mix_graph.nodes())+1}")
            #print(f"Nodes after addition: {mix_graph.nodes()}")
            mix_nodes = max(mix_graph.nodes())
        elif mix_task == "remove_node":
            node_for_removal = random.choice(list(mix_graph.nodes()))
            #print(f"Mix Node for removal: {node_for_removal}")
            mix_nodes = node_for_removal
            mix_graph.remove_node(node_for_removal)

        #for node, v in mix_node_mapping.items():
        #    print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")
            

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(mix_graph.nodes))}
        for node in mix_graph:
            #print(f"Node {node} is being renumbered to {mapping[node]} at chain length {chain_length}")
            mix_node_mapping[node][chain_length] = mapping[node]

        #for node, v in mix_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v} at chain length {chain_length}")

        #print()

        relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

        # -------------------------- #
        # Save relevant final nodes
        if graph_type != "complete":
            # Save releveant nodes for final task on our add edge graph
            relavent_nodes_final["add_edge"][chain_length]["node_degree"] = random.choice(list(add_edge_graph.nodes()))
            relavent_nodes_final["add_edge"][chain_length]["edge_exists"] = random.sample(list(add_edge_graph.nodes()), 2)
            #nodes_with_neighbors = [node for node in add_edge_graph.nodes() if add_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
            #node_deg = random.choice(nodes_with_neighbors)
            node_deg = random.choice(list(add_edge_graph.nodes()))
            relavent_nodes_final["add_edge"][chain_length]["connected_nodes"] = node_deg

        if graph_type != "empty":
            # Save releveant nodes for final task on our remove edge graph
            relavent_nodes_final["remove_edge"][chain_length]["node_degree"] = random.choice(list(remove_edge_graph.nodes()))
            relavent_nodes_final["remove_edge"][chain_length]["edge_exists"] = random.sample(list(remove_edge_graph.nodes()), 2)
            #nodes_with_neighbors = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
            #print(graph_to_string_encoder(remove_edge_graph))
            #node_deg = random.choice(nodes_with_neighbors)
            node_deg = random.choice(list(remove_edge_graph.nodes()))
            relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our add node graph
        relavent_nodes_final["add_node"][chain_length]["node_degree"] = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["edge_exists"] = random.sample(list(add_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_node_graph.nodes() if add_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove node graph
        relavent_nodes_final["remove_node"][chain_length]["node_degree"] = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["edge_exists"] = random.sample(list(remove_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_node_graph.nodes() if remove_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our mix graph
        relavent_nodes_final["mix"][chain_length]["node_degree"] = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["edge_exists"] = random.sample(list(mix_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in mix_graph.nodes() if mix_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["connected_nodes"] = node_deg

        # -------------------------- #
        # Save solutions

        if graph_type != "complete":
            # Save solutions for add edge graph
            solutions["add_edge"][chain_length]["node_count"] = str(add_edge_graph.number_of_nodes())
            solutions["add_edge"][chain_length]["edge_count"] = str(add_edge_graph.number_of_edges())
            solutions["add_edge"][chain_length]["node_degree"] = str(add_edge_graph.degree[relavent_nodes_final["add_edge"][chain_length]["node_degree"]])
            solutions["add_edge"][chain_length]["edge_exists"] = "Yes" if add_edge_graph.has_edge(relavent_nodes_final["add_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["add_edge"][chain_length]["edge_exists"][1]) else "No"
            solutions["add_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_edge_graph.neighbors(relavent_nodes_final["add_edge"][chain_length]["connected_nodes"])])
            try:
                nx.find_cycle(add_edge_graph)
                solutions["add_edge"][chain_length]["cycle"] = "Yes"
            except nx.NetworkXNoCycle:
                solutions["add_edge"][chain_length]["cycle"] = "No"
            solutions["add_edge"][chain_length]["print_graph"] = add_edge_graph.copy()

        if graph_type != "empty":
            # Save solutions for remove edge graph
            solutions["remove_edge"][chain_length]["node_count"] = str(remove_edge_graph.number_of_nodes())
            solutions["remove_edge"][chain_length]["edge_count"] = str(remove_edge_graph.number_of_edges())
            solutions["remove_edge"][chain_length]["node_degree"] = str(remove_edge_graph.degree[relavent_nodes_final["remove_edge"][chain_length]["node_degree"]])
            solutions["remove_edge"][chain_length]["edge_exists"] = "Yes" if remove_edge_graph.has_edge(relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][1]) else "No"
            solutions["remove_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_edge_graph.neighbors(relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"])])
            try:
                nx.find_cycle(remove_edge_graph)
                solutions["remove_edge"][chain_length]["cycle"] = "Yes"
            except nx.NetworkXNoCycle:
                solutions["remove_edge"][chain_length]["cycle"] = "No"     
            solutions["remove_edge"][chain_length]["print_graph"] = remove_edge_graph.copy()

        # Save solutions for add node graph
        solutions["add_node"][chain_length]["node_count"] = str(add_node_graph.number_of_nodes())
        solutions["add_node"][chain_length]["edge_count"] = str(add_node_graph.number_of_edges())
        solutions["add_node"][chain_length]["node_degree"] = str(add_node_graph.degree[relavent_nodes_final["add_node"][chain_length]["node_degree"]])
        solutions["add_node"][chain_length]["edge_exists"] = "Yes" if add_node_graph.has_edge(relavent_nodes_final["add_node"][chain_length]["edge_exists"][0], relavent_nodes_final["add_node"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_node_graph.neighbors(relavent_nodes_final["add_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_node_graph)
            solutions["add_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_node"][chain_length]["cycle"] = "No"   
        solutions["add_node"][chain_length]["print_graph"] = add_node_graph.copy()

        # Save solutions for remove node graph TODO: fix this
        solutions["remove_node"][chain_length]["node_count"] = str(remove_node_graph.number_of_nodes())
        solutions["remove_node"][chain_length]["edge_count"] = str(remove_node_graph.number_of_edges())
        solutions["remove_node"][chain_length]["node_degree"] = str(remove_node_graph.degree[relavent_nodes_final["remove_node"][chain_length]["node_degree"]])
        solutions["remove_node"][chain_length]["edge_exists"] = "Yes" if remove_node_graph.has_edge(relavent_nodes_final["remove_node"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_node"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([remove_node_mapping[int(node_b)][chain_length] for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_node_graph)
            solutions["remove_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_node"][chain_length]["cycle"] = "No"    
        solutions["remove_node"][chain_length]["print_graph"] = remove_node_graph.copy()

        # Save solutions for mix graph TODO: fix this
        solutions["mix"][chain_length]["node_count"] = str(mix_graph.number_of_nodes())
        solutions["mix"][chain_length]["edge_count"] = str(mix_graph.number_of_edges())
        solutions["mix"][chain_length]["node_degree"] = str(mix_graph.degree[relavent_nodes_final["mix"][chain_length]["node_degree"]])
        solutions["mix"][chain_length]["edge_exists"] = "Yes" if mix_graph.has_edge(relavent_nodes_final["mix"][chain_length]["edge_exists"][0], relavent_nodes_final["mix"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([mix_node_mapping[int(node_b)][chain_length] for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(mix_graph)
            solutions["mix"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["mix"][chain_length]["cycle"] = "No"
        solutions["mix"][chain_length]["print_graph"] = mix_graph.copy()

    def modification_prompt(encoding, modification, chain_num, relevant_nodes_mod):
        #print(f"Generating modification prompt for encoding {encoding}, modification {modification}, chain length {chain_length}...")
        modification_prompt = ""
        for mod_number in range(1, chain_num+1):
            node = relevant_nodes_mod[modification][mod_number]
            #print(f"relevant_nodes_mod[{modification}][{mod_number}]: {node}")
            if modification == "add_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "add_node":
                if encoding == "adjacency_matrix":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                elif encoding in ["adjacency_list", "incident"]:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                else:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node called {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_node":
                if mod_number == 1:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the graph.\n"
                else:
                    if encoding == "adjacency_matrix":
                        modification_prompt += f"{mod_number}: Remove node {remove_node_mapping[node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
            elif modification == "mix":
                mod = node[0]
                mix_node = node[1]
                if mod == "add_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Add an edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_edge":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove the edge between node {mix_node_mapping[mix_node[0]][mod_number-1]} and node {mix_node_mapping[mix_node[1]][mod_number-1]} in the resulting graph of operation {mod_number-1}.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "add_node":
                    if encoding == "adjacency_matrix":
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                    elif encoding in ["adjacency_list", "incident"]:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                    else:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_node":
                    if mod_number == 1:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the graph, and renumber the nodes accordingly.\n"
                        else:   
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the graph.\n"
                    else:
                        if encoding == "adjacency_matrix":
                            modification_prompt += f"{mod_number}: Remove node {mix_node_mapping[mix_node][mod_number-1]} from the resulting graph of operation {mod_number-1}, and renumber the nodes accordingly.\n"
                        else:
                            modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
        return modification_prompt
    
    def question_prompt(task, encoding, modification, chain_num, relevant_nodes_final):
        node = relevant_nodes_final[modification][chain_num][task]
        if task == "node_count":
            return f"Q: How many nodes are in the final resulting graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many edges are in the final resulting graph?\nA: "
        elif task == "node_degree":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    #print(f"In question remove, node: {node}, gets mapped to {remove_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {remove_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                elif modification == "mix":
                    #print(f"In question mix, node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: How many neighbors does node {mix_node_mapping[node][chain_num]} have in the final resulting graph?\nA: "
                else:
                    return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
            else:
                return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
        elif task == "edge_exists":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: Is node {remove_node_mapping[node[0]][chain_num]} connected to node {remove_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                elif modification == "mix":
                    return f"Q: Is node {mix_node_mapping[node[0]][chain_num]} connected to node {mix_node_mapping[node[1]][chain_num]} in the final resulting graph?\nA: "
                else:
                    return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
            else:
                return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
        elif task == "connected_nodes":
            if encoding == "adjacency_matrix":
                if modification == "remove_node":
                    return f"Q: List all neighbors of node {remove_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                elif modification == "mix":
                    #print(f"Node: {node}, gets mapped to {mix_node_mapping[node][chain_num]}")
                    return f"Q: List all neighbors of node {mix_node_mapping[node][chain_num]} in the final resulting graph.\nA: "
                else:
                    return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
            else:
                return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
        elif task == "cycle":
            return f"Q: Does the final resulting graph contain a cycle?\nA: "
        elif task == "print_graph":
            if encoding == "adjacency_matrix":
                return f"Q: What is the final resulting adjacency matrix? Write out the entire final resulting adjacency matrix. \nA: "
            else:
                return f"Q: What is the final resulting graph? Present the graph in the same structure as above, and write out the entire resulting graph.\nA: "
        else:
            print("Task not recognized. Exiting.")
            sys.exit(1)
   
    # generate all prompts
    for task in final_tasks:
        for modification in modifications:
            for encoding in encodings:
                for chain_length in range(1, max_chain_length+1):
                    #if chain_length == 5 and encoding == "adjacency_matrix" and modification == "remove_node" and task == "edge_count":
                    print(f"Generating prompts for task {task}, modification {modification}, encoding {encoding}, and chain length {chain_length}...")
                    # construct example prompts for few + cot

                    # construct init_prompt
                    init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name, graph_type)

                    graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

                    # construct modification prompt
                    modification_prompt_str = modification_prompt(encoding, modification, chain_length, relavent_nodes_mod)

                    # construct question prompt
                    question_prompt_str = question_prompt(task, encoding, modification, chain_length, relavent_nodes_final)

                    # construct solution
                    solution = solutions[modification][chain_length][task]

                    if task == "connected_nodes" and encoding == "adjacency_matrix":
                        if modification == "remove_node":
                            solution = [remove_node_mapping[node][chain_length] for node in solution]
                        elif modification == "mix":
                            solution = [mix_node_mapping[node][chain_length] for node in solution]
                    # construct full prompt
                    full_prompt = init_prompt + graph_string + modification_prompt_str + question_prompt_str # + solution + cot

                    #print(f"Full prompt: {full_prompt}")
                    #print(f"Solution: {solution}")

                    # Write graph to file
                    graph_filename = f"data/{encoding}_chain_big_{graph_type}_no_perform/{task}/{modification}/{chain_length}/input_graphs/{i}.graphml"
                    nx.write_graphml(graph, graph_filename)

                    # Write prompt to file
                    prompt_filename = f"data/{encoding}_chain_big_{graph_type}_no_perform/{task}/{modification}/{chain_length}/prompts/prompt_{i}.txt"
                    with open(prompt_filename, "w") as prompt_file:
                        prompt_file.write(full_prompt)

                    if task == "print_graph":
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}_no_perform/{task}/{modification}/{chain_length}/solutions/solution_{i}.graphml"
                        nx.write_graphml(solution, solution_filename)
                    elif task == "connected_nodes":
                        # Use node_to_name on list
                        solution = [node_to_name[node][encoding] for node in solution]

                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}_no_perform/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(str(solution))
                    else:
                        # Write solution to file
                        solution_filename = f"data/{encoding}_chain_big_{graph_type}_no_perform/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                        with open(solution_filename, "w") as solution_file:
                            solution_file.write(solution)

def get_info(graph, i, max_chain_length):
    encodings = ["adjacency_matrix", "incident", "coauthorship"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
    final_tasks = ["node_count", "edge_count", "node_degree", "edge_exists", "connected_nodes", "cycle", "print_graph"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    nodes = list(graph.nodes)

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]

        remove_node_mapping[n] = {0: n}
        mix_node_mapping[n] = {0: n}

    #add_edge = lambda graph: nx.add_edge(graph, random.choice(list(graph.nodes())), random.choice(list(graph.nodes())))
    add_edge_graph = graph.copy()
    remove_edge_graph = graph.copy()
    add_node_graph = graph.copy()
    remove_node_graph = graph.copy()
    mix_graph = graph.copy()

    relavent_nodes_mod = {
        "add_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_edge": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "add_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "remove_node": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        },
        "mix": {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
    }

    relavent_nodes_final = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        relavent_nodes_final[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            relavent_nodes_final[modification][chain_length] = {}
            for final_task in final_tasks:
                relavent_nodes_final[modification][chain_length][final_task] = 0

    solutions = {
        "add_edge": {},
        "remove_edge": {},
        "add_node": {},
        "remove_node": {},
        "mix": {}
    }

    for modification in modifications:
        solutions[modification] = {}
        for chain_length in range(1, max_chain_length+1):
            solutions[modification][chain_length] = {}
            for task in final_tasks:
                solutions[modification][chain_length][task] = 0

    for chain_length in range(1, max_chain_length+1):
        # -------------------------- #
        # Save relevant modification nodes

        # get "add edge" nodes
        unconnected_nodes = []
        for node_a in add_edge_graph.nodes():
            for node_b in add_edge_graph.nodes():
                if node_a != node_b and not add_edge_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                    unconnected_nodes.append((node_a, node_b))
        node_a, node_b = random.sample(unconnected_nodes, 1)[0]
        #add_edge_nodes.append([node_a, node_b])
        relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]

        # get "add edge" solutions
        add_edge_graph.add_edge(node_a, node_b)

        # get "remove edge" nodes
        edge = random.choice(list(remove_edge_graph.edges()))
        #remove_edge_nodes.append(edge)
        relavent_nodes_mod["remove_edge"][chain_length] = edge
        remove_edge_graph.remove_edge(edge[0], edge[1])

        # get "add node" node
        add_node_graph.add_node(max(add_node_graph.nodes()) + 1)
        #add_node_nodes.append(max(add_node_graph.nodes()))
        relavent_nodes_mod["add_node"][chain_length] = max(add_node_graph.nodes())

        # get "remove node" node
        node_for_removal = random.choice(list(remove_node_graph.nodes()))
        #remove_node_nodes.append(node_for_removal)
        relavent_nodes_mod["remove_node"][chain_length] = node_for_removal
        remove_node_graph.remove_node(node_for_removal)
        #print(f"Remove Node for removal: {node_for_removal}")

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #graph = nx.relabel_nodes(graph, mapping)
        #print(f"Mapping: {mapping}")
        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")

        for node in remove_node_graph.nodes():
            #print(f"Node {node} is being renumbered to {mapping[node]}")
            remove_node_mapping[node][chain_length] = mapping[node]

        #for node, v in remove_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v}")

        # Renumber the nodes to be consecutive
        #mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(remove_node_graph.nodes))}
        #remove_node_graph = nx.relabel_nodes(remove_node_graph, mapping)

        # get "mix" task
        mix_task = random.choice(modifications[:-1])

        # check if remove edge works
        if mix_task == "remove_edge":
            # if there are no more edges, choose one of the other tasks
            if len(list(mix_graph.edges())) == 0:
                mix_task = random.choice(["add_edge", "add_node", "remove_node"]) 

        #print(f"Mix task: {mix_task}")        

        if mix_task == "add_edge":
            # get "add edge" nodes
            unconnected_nodes = []
            for node_a in mix_graph.nodes():
                for node_b in mix_graph.nodes():
                    if node_a != node_b and not mix_graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                        unconnected_nodes.append((node_a, node_b))
            node_a, node_b = random.sample(unconnected_nodes, 1)[0]
            mix_nodes = [node_a, node_b]
            mix_graph.add_edge(node_a, node_b)
        elif mix_task == "remove_edge":
            edge = random.choice(list(mix_graph.edges()))
            mix_nodes = edge
            mix_graph.remove_edge(edge[0], edge[1])
        elif mix_task == "add_node":
            #print(f"Nodes before addition: {mix_graph.nodes()}")
            mix_graph.add_node(max(mix_graph.nodes()) + 1)
            #print(f"Node for addition: {max(mix_graph.nodes())}")
            #print(f"Node for addition + 1: {max(mix_graph.nodes())+1}")
            #print(f"Nodes after addition: {mix_graph.nodes()}")
            mix_nodes = max(mix_graph.nodes())
        elif mix_task == "remove_node":
            node_for_removal = random.choice(list(mix_graph.nodes()))
            #print(f"Mix Node for removal: {node_for_removal}")
            mix_nodes = node_for_removal
            mix_graph.remove_node(node_for_removal)

        #for node, v in mix_node_mapping.items():
        #    print(f"Node {node} is currently mapped to {v} at chain length {chain_length}")
            

        # Renumber the nodes to be consecutive
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(mix_graph.nodes))}
        for node in mix_graph:
            #print(f"Node {node} is being renumbered to {mapping[node]} at chain length {chain_length}")
            mix_node_mapping[node][chain_length] = mapping[node]

        #for node, v in mix_node_mapping.items():
            #print(f"Node {node} has been renumbered to {v} at chain length {chain_length}")

        #print()

        relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

        # -------------------------- #
        # Save relevant final nodes

        # Save releveant nodes for final task on our add edge graph
        relavent_nodes_final["add_edge"][chain_length]["node_degree"] = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["edge_exists"] = random.sample(list(add_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_edge_graph.nodes() if add_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_edge_graph.nodes()))
        relavent_nodes_final["add_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove edge graph
        relavent_nodes_final["remove_edge"][chain_length]["node_degree"] = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["edge_exists"] = random.sample(list(remove_edge_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_edge_graph.nodes() if remove_edge_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #print(graph_to_string_encoder(remove_edge_graph))
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_edge_graph.nodes()))
        relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our add node graph
        relavent_nodes_final["add_node"][chain_length]["node_degree"] = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["edge_exists"] = random.sample(list(add_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in add_node_graph.nodes() if add_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(add_node_graph.nodes()))
        relavent_nodes_final["add_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our remove node graph
        relavent_nodes_final["remove_node"][chain_length]["node_degree"] = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["edge_exists"] = random.sample(list(remove_node_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in remove_node_graph.nodes() if remove_node_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(remove_node_graph.nodes()))
        relavent_nodes_final["remove_node"][chain_length]["connected_nodes"] = node_deg

        # Save releveant nodes for final task on our mix graph
        relavent_nodes_final["mix"][chain_length]["node_degree"] = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["edge_exists"] = random.sample(list(mix_graph.nodes()), 2)
        #nodes_with_neighbors = [node for node in mix_graph.nodes() if mix_graph.degree[node] > 0] # TODO: change this to just all nodes, not just the nodes with at least one neighbor?!
        #node_deg = random.choice(nodes_with_neighbors)
        node_deg = random.choice(list(mix_graph.nodes()))
        relavent_nodes_final["mix"][chain_length]["connected_nodes"] = node_deg

        # -------------------------- #
        # Save solutions

        # Save solutions for add edge graph
        solutions["add_edge"][chain_length]["node_count"] = str(add_edge_graph.number_of_nodes())
        solutions["add_edge"][chain_length]["edge_count"] = str(add_edge_graph.number_of_edges())
        solutions["add_edge"][chain_length]["node_degree"] = str(add_edge_graph.degree[relavent_nodes_final["add_edge"][chain_length]["node_degree"]])
        solutions["add_edge"][chain_length]["edge_exists"] = "Yes" if add_edge_graph.has_edge(relavent_nodes_final["add_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["add_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_edge_graph.neighbors(relavent_nodes_final["add_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_edge_graph)
            solutions["add_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_edge"][chain_length]["cycle"] = "No"
        solutions["add_edge"][chain_length]["print_graph"] = add_edge_graph.copy()

        # Save solutions for remove edge graph
        solutions["remove_edge"][chain_length]["node_count"] = str(remove_edge_graph.number_of_nodes())
        solutions["remove_edge"][chain_length]["edge_count"] = str(remove_edge_graph.number_of_edges())
        solutions["remove_edge"][chain_length]["node_degree"] = str(remove_edge_graph.degree[relavent_nodes_final["remove_edge"][chain_length]["node_degree"]])
        solutions["remove_edge"][chain_length]["edge_exists"] = "Yes" if remove_edge_graph.has_edge(relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_edge"][chain_length]["edge_exists"][1]) else "No"
        solutions["remove_edge"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_edge_graph.neighbors(relavent_nodes_final["remove_edge"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_edge_graph)
            solutions["remove_edge"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_edge"][chain_length]["cycle"] = "No"     
        solutions["remove_edge"][chain_length]["print_graph"] = remove_edge_graph.copy()

        # Save solutions for add node graph
        solutions["add_node"][chain_length]["node_count"] = str(add_node_graph.number_of_nodes())
        solutions["add_node"][chain_length]["edge_count"] = str(add_node_graph.number_of_edges())
        solutions["add_node"][chain_length]["node_degree"] = str(add_node_graph.degree[relavent_nodes_final["add_node"][chain_length]["node_degree"]])
        solutions["add_node"][chain_length]["edge_exists"] = "Yes" if add_node_graph.has_edge(relavent_nodes_final["add_node"][chain_length]["edge_exists"][0], relavent_nodes_final["add_node"][chain_length]["edge_exists"][1]) else "No"
        solutions["add_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in add_node_graph.neighbors(relavent_nodes_final["add_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(add_node_graph)
            solutions["add_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["add_node"][chain_length]["cycle"] = "No"   
        solutions["add_node"][chain_length]["print_graph"] = add_node_graph.copy()

        # Save solutions for remove node graph TODO: fix this
        solutions["remove_node"][chain_length]["node_count"] = str(remove_node_graph.number_of_nodes())
        solutions["remove_node"][chain_length]["edge_count"] = str(remove_node_graph.number_of_edges())
        solutions["remove_node"][chain_length]["node_degree"] = str(remove_node_graph.degree[relavent_nodes_final["remove_node"][chain_length]["node_degree"]])
        solutions["remove_node"][chain_length]["edge_exists"] = "Yes" if remove_node_graph.has_edge(relavent_nodes_final["remove_node"][chain_length]["edge_exists"][0], relavent_nodes_final["remove_node"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([remove_node_mapping[int(node_b)][chain_length] for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        solutions["remove_node"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in remove_node_graph.neighbors(relavent_nodes_final["remove_node"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(remove_node_graph)
            solutions["remove_node"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["remove_node"][chain_length]["cycle"] = "No"    
        solutions["remove_node"][chain_length]["print_graph"] = remove_node_graph.copy()

        # Save solutions for mix graph TODO: fix this
        solutions["mix"][chain_length]["node_count"] = str(mix_graph.number_of_nodes())
        solutions["mix"][chain_length]["edge_count"] = str(mix_graph.number_of_edges())
        solutions["mix"][chain_length]["node_degree"] = str(mix_graph.degree[relavent_nodes_final["mix"][chain_length]["node_degree"]])
        solutions["mix"][chain_length]["edge_exists"] = "Yes" if mix_graph.has_edge(relavent_nodes_final["mix"][chain_length]["edge_exists"][0], relavent_nodes_final["mix"][chain_length]["edge_exists"][1]) else "No"
        #if encoding == "adjacency_matrix":
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([mix_node_mapping[int(node_b)][chain_length] for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        #else:
        #    solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        solutions["mix"][chain_length]["connected_nodes"] = sorted([int(node_b) for node_b in mix_graph.neighbors(relavent_nodes_final["mix"][chain_length]["connected_nodes"])])
        try:
            nx.find_cycle(mix_graph)
            solutions["mix"][chain_length]["cycle"] = "Yes"
        except nx.NetworkXNoCycle:
            solutions["mix"][chain_length]["cycle"] = "No"
        solutions["mix"][chain_length]["print_graph"] = mix_graph.copy()
 
    # Save relavent_nodes_mod dictionary 
    directory = "data/dicts/mod"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/relavent_nodes_mod_{i}.json", "w") as f:
        json.dump(relavent_nodes_mod, f)

    # Save relavent_nodes_final dictionary
    directory = "data/dicts/final"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(f"data/dicts/final/relavent_nodes_final_{i}.json", "w") as f:
        json.dump(relavent_nodes_final, f)

    # Save remove_node_mapping dictionary
    directory = "data/dicts/remove_node_mapping"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"data/dicts/remove_node_mapping/remove_node_mapping_{i}.json", "w") as f:
        json.dump(remove_node_mapping, f)

    # Save mix_node_mapping dictionary
    directory = "data/dicts/mix_node_mapping"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"data/dicts/mix_node_mapping/mix_node_mapping_{i}.json", "w") as f:
        json.dump(mix_node_mapping, f)

def graph_to_prompts_chain_fc(i, max_chain_length):
    encodings = ["adjacency_matrix"]#, "friendship", "social_network", "expert", "politician", "got", "sp"]
    modifications = ["add_edge"]
    final_tasks = ["print_graph"]
    
    # create a list of 25 strings of common names
    names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Mark", "Lisa", "Daniel", "Nancy", "Paul"]
    south_park_names = ["Stan", "Kyle", "Cartman", "Kenny", "Butters", "Wendy", "Randy", "Sharon", "Gerald", "Liane", "Token", "Clyde", "Craig", "Tweek", "Jimmy", "Timmy", "Bebe", "Heidi", "Nichole", "Red", "Principal", "Mackey", "Chef", "Garrison", "Towelie"]
    game_of_thrones_names = ["Jon", "Daenerys", "Tyrion", "Sansa", "Arya", "Bran", "Cersei", "Jaime", "Brienne", "Davos", "Samwell", "Gilly", "Jorah", "Theon", "Yara", "Euron", "Varys", "Melisandre", "Missandei", "Grey Worm", "Hodor", "Beric", "Tormund", "Podrick", "Ned"]
    politician_names = ["Joe", "Kamala", "Donald", "Mike", "Bernie", "Elizabeth", "Nancy", "Mitch", "Chuck", "Lindsey", "Ted", "AOC", "Ilhan", "Rashida", "Ayanna", "Pete", "Andrew", "Amy", "Tulsi", "Tom", "Obama", "Hillary", "Bush", "Reagan", "Carter"]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    node_to_name = {}

    # account for implicit renumbering, if chain length = 1, map n to n, else renumber. We only need this for remove node and mix
    remove_node_mapping = {}
    mix_node_mapping = {}

    # map from node ID to name depending on the encoding
    for n in range(25):
        node_to_name[n] = {}
        node_to_name[n]["adjacency_list"] = n
        node_to_name[n]["incident"] = n
        node_to_name[n]["adjacency_matrix"] = n 
        node_to_name[n]["coauthorship"] = names[n]
        node_to_name[n]["friendship"] = names[n]
        node_to_name[n]["social_network"] = names[n]
        node_to_name[n]["expert"] = letters[n]
        node_to_name[n]["politician"] = politician_names[n]
        node_to_name[n]["got"] = game_of_thrones_names[n]
        node_to_name[n]["sp"] = south_park_names[n]


    def graph_to_init_prompt(nodes, encoding, node_to_name):
        # construct init_prompt
        if encoding == "adjacency_list":
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge G. G describes a graph among {nodes_str}.\nThe edges in G are:\n"
        elif encoding == "incident": # G describes a graph among 0, 1, 2, 3, 4, 5, 6, 7, and 8.
            # create a comma-separated list of nodes, but the last two nodes are separated by ", and"
            nodes_str = ', '.join([str(n) for n in nodes[:-2]]) + ', ' + str(nodes[-2]) + ', and ' + str(nodes[-1])
            return f"G describes a graph among {nodes_str}.\nIn this graph:\n"
        elif encoding == "coauthorship":
            nodes_str = ', '.join([node_to_name[int(n)]["coauthorship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["coauthorship"] + ', and ' + node_to_name[int(nodes[-1])]["coauthorship"]
            return f"G describes a co-authorship graph among {nodes_str}.\nIn this co-authorship graph:\n"
        elif encoding == "friendship":
            nodes_str = ', '.join([node_to_name[int(n)]["friendship"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["friendship"] + ', and ' + node_to_name[int(nodes[-1])]["friendship"]
            return f"G describes a friendship graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "social_network":
            nodes_str = ', '.join([node_to_name[int(n)]["social_network"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["social_network"] + ', and ' + node_to_name[int(nodes[-1])]["social_network"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "expert":
            nodes_str = ', '.join([node_to_name[int(n)]["expert"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["expert"] + ', and ' + node_to_name[int(nodes[-1])]["expert"]
            return f"You are a graph analyst and you have been given a graph G among {nodes_str}.\nG has the following undirected edges:\n"
        elif encoding == "politician":
            nodes_str = ', '.join([node_to_name[int(n)]["politician"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["politician"] + ', and ' + node_to_name[int(nodes[-1])]["politician"]
            return f"G describes a social network graph among {nodes_str}.\nWe have the following edges in G:\n"
        elif encoding == "got":
            nodes_str = ', '.join([node_to_name[int(n)]["got"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["got"] + ', and ' + node_to_name[int(nodes[-1])]["got"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "sp":
            nodes_str = ', '.join([node_to_name[int(n)]["sp"] for n in nodes[:-2]]) + ', ' + node_to_name[int(nodes[-2])]["sp"] + ', and ' + node_to_name[int(nodes[-1])]["sp"]
            return f"G describes a friendship graph among {nodes_str}.\nIn this friendship graph:\n"
        elif encoding == "adjacency_matrix":
            return f"The following matrix represents the adjacency matrix of an undirected graph, where the first row corresponds to node 0, the second row corresponds to node 1, and so on: \n"
        else:
            print("Encoding not recognized. Exiting.")
            sys.exit(1)

    

    def modification_prompt(encoding, modification, chain_length, relevant_nodes_mod):
        print(f"Generating modification prompt for encoding {encoding}, modification {modification}, chain length {chain_length}...")
        modification_prompt = "Perform the following operations on the graph:\n"
        for mod_number in range(1, chain_length+1):
            node = relevant_nodes_mod[modification][mod_number]
            print(f"relevant_nodes_mod[{modification}][{mod_number}]: {node}")
            if modification == "add_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_edge":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]}.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[node[0]][encoding]} and node {node_to_name[node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
            elif modification == "add_node":
                if encoding in ["adjacency_matrix"]:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node to the resulting graph of operation {mod_number-1}.\n"
                else:
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add a node {node_to_name[node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
            elif modification == "remove_node":
                if mod_number == 1:
                    modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the graph.\n"
                else:
                    modification_prompt += f"{mod_number}: Remove node {node_to_name[node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
            elif modification == "mix":
                mod = node[0]
                mix_node = node[1]
                if mod == "add_edge":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        modification_prompt += f"{mod_number}: Add an edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_edge":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]}.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove the edge between node {node_to_name[mix_node[0]][encoding]} and node {node_to_name[mix_node[1]][encoding]} in the resulting graph of operation {mod_number-1}.\n"
                elif mod == "add_node":
                    if encoding in ["adjacency_list", "incident", "adjacency_matrix"]:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                    else:
                        if mod_number == 1:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the graph.\n"
                        else:
                            modification_prompt += f"{mod_number}: Add a node called {node_to_name[mix_node][encoding]} to the resulting graph of operation {mod_number-1}.\n"
                elif mod == "remove_node":
                    if mod_number == 1:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the graph.\n"
                    else:
                        modification_prompt += f"{mod_number}: Remove node {node_to_name[mix_node][encoding]} from the resulting graph of operation {mod_number-1}.\n"
        return modification_prompt
    
    def question_prompt(task, encoding, modification, chain_length, relevant_nodes_final):
        node = relevant_nodes_final[modification][chain_length][task]
        if task == "node_count":
            return f"Q: How many nodes are in the final resulting graph?\nA: "
        elif task == "edge_count":
            return f"Q: How many edges are in the final resulting graph?\nA: "
        elif task == "node_degree":
            return f"Q: How many neighbors does node {node_to_name[node][encoding]} have in the final resulting graph?\nA: "
        elif task == "edge_exists":
            return f"Q: Is node {node_to_name[node[0]][encoding]} connected to node {node_to_name[node[1]][encoding]} in the final resulting graph?\nA: "
        elif task == "connected_nodes":
            return f"Q: List all neighbors of node {node_to_name[node][encoding]} in the final resulting graph.\nA: "
        elif task == "cycle":
            return f"Q: Does the final resulting graph contain a cycle?\nA: "
        elif task == "print_graph":
            if encoding == "adjacency_matrix":
                return f"Q: What is the final resulting adjacency matrix?\nA: "
            else:
                return f"Q: What is the final resulting graph? Present the graph in the same structure as above, and write out the entire resulting graph.\nA: "
        else:
            print("Task not recognized. Exiting.")
            sys.exit(1)

    # relavent_nodes_mod["add_edge"][chain_length] = [node_a, node_b]
    # relavent_nodes_mod["mix"][chain_length] = [mix_task, mix_nodes]

    def examples_to_prompt(graphs, prompts, solutions, task, encoding, modification, chain_length, relavent_nodes_mods, relavent_nodes_finals, node_mappings=None):
        few_shot_prompt = ""
        cot_prompt = ""
        num_examples = len(graphs)
        for i in range(num_examples):
            relavent_nodes_mod = relavent_nodes_mods[i]
            relavent_nodes_final = relavent_nodes_finals[i]
            if node_mappings is not None:
                node_mapping = node_mappings[i]
            few_shot_prompt += f"Example {i+1}:\n"
            few_shot_prompt += prompts[i]
            if task == "print_graph":
                few_shot_prompt += graph_to_string_encoder(solutions[i], encoding)
            else:
                few_shot_prompt += solutions[i]
            few_shot_prompt += "\n\n"
            cot_prompt += f"Example {i+1}:\n"
            cot_prompt += prompts[i]
            #if task == "print_graph":
            #    cot_prompt += graph_to_string_encoder(solutions[i], encoding)
            #else:
            #    cot_prompt += solutions[i]
            #cot_prompt += ", because "
            #print(f"cot_prompt: {cot_prompt}")
            if task == "node_count":
                #print(1)
                if modification == "add_edge":
                    #print(2)
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        #print(3)
                        for mod_num in range(1, chain_length+1):
                            #print(f"relavent_nodes_mod: {relavent_nodes_mod}")
                            #print(f"relavent_nodes_mod[modification]: {relavent_nodes_mod[modification]}")
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        #print(4)
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} rows, so the number of nodes in the graph is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        node_list_string = ', '.join([str(node_to_name[int(n)][encoding]) for n in graphs[i].nodes()])
                        cot_prompt += f"after adding {relavent_nodes_string}the nodes in the graph are {node_list_string}, so the number of nodes in the graph is {solutions[i]}."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} rows, so the number of nodes in the graph is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        node_list_string = ', '.join([str(node_to_name[int(n)][encoding]) for n in graphs[i].nodes()])
                        cot_prompt += f"after removing {relavent_nodes_string}the nodes in the graph are {node_list_string}, so the number of nodes in the graph is {solutions[i]}."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} rows, so the number of nodes in the graph is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        node_list_string = ', '.join([str(node_to_name[int(n)][encoding]) for n in graphs[i].nodes()])
                        cot_prompt += f"after adding {relavent_nodes_string}the nodes in the graph are {node_list_string}, so the number of nodes in the graph is {solutions[i]}."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            print(f"node_mapping: {node_mapping}")
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, " # TODO: add info on mapping/renumbering
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "  # TODO: add info on mapping/renumbering
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} rows, so the number of nodes in the graph is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        node_list_string = ', '.join([str(node_to_name[int(n)][encoding]) for n in graphs[i].nodes()])
                        cot_prompt += f"after removing {relavent_nodes_string}the nodes in the graph are {node_list_string}, so the number of nodes in the graph is {solutions[i]}."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                #print(f"node_a: {node_a}, node_b: {node_b}")
                                #print(f"relavent_nodes_string: {relavent_nodes_string}")
                            elif mix_task == "add_node":
                                node = mix_nodes
                                #print(f"node: {node}")
                                #print(f"node_mapping: {node_mapping}")
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, " # not mod_num-1 because it was just created
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, " # not mod_num-1 because it was just created
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} rows, so the number of nodes in the graph is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and adding the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"adding the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        node_list_string = ', '.join([str(node_to_name[int(n)][encoding]) for n in graphs[i].nodes()])
                        cot_prompt += f"after {relavent_nodes_string}the nodes in the graph are {node_list_string}, so the number of nodes in the graph is {solutions[i]}."
            elif task == "edge_count":
                if modification == "add_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {int(solutions[i])*2} 1's, so the number of edges in the graph is {int(solutions[i])*2}/2 = {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the graph has {solutions[i]} edges, so the number of edges in the graph is {solutions[i]}."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {int(solutions[i])*2} 1's, so the number of edges in the graph is {int(solutions[i])*2}/2 = {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}the graph has {solutions[i]} edges, so the number of edges in the graph is {solutions[i]}."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix has {int(solutions[i])*2} 1's, so the number of edges in the graph is {int(solutions[i])*2}/2 = {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        cot_prompt += f"after adding {relavent_nodes_string}the graph has {solutions[i]} edges, so the number of edges in the graph is {solutions[i]}."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, " # TODO: add info on mapping/renumbering
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "  # TODO: add info on mapping/renumbering
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix has {int(solutions[i])*2} 1's, so the number of edges in the graph is {int(solutions[i])*2}/2 = {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after removing {relavent_nodes_string}the graph has {solutions[i]} edges, so the number of edges in the graph is {solutions[i]}."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, " # TODO: add info on mapping/renumbering
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "  # TODO: add info on mapping/renumbering
                        cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix has {int(solutions[i])*2} 1's, so the number of edges in the graph is {int(solutions[i])*2}/2 = {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"adding the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after {relavent_nodes_string}the graph has {solutions[i]} edges, so the number of edges in the graph is {solutions[i]}."
            elif task == "node_degree":
                node_of_interest = relavent_nodes_final[modification][str(chain_length)][task]
                if modification == "add_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} 1's in node {node_of_interest}'s row, so the degree of node {node_of_interest} is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            # print all the nodes of graphs[i]
                            #print(f"graphs[i].nodes(): {graphs[i].nodes()}")
                            #print(f"node_of_interest: {node_of_interest}")
                            #print(type(node_of_interest))
                        # list all the edges that includes the node of interest
                        node_of_interest_edges = list(graphs[i].edges(str(node_of_interest)))
                        # convert the list into a string of edges represented as tuples: (node_a, node_b)
                        node_of_interest_edges_string = ', '.join([str(edge) for edge in node_of_interest_edges])
                        cot_prompt += f"after adding {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} has the following edges: {node_of_interest_edges_string}, so the degree of node {node_to_name[node_of_interest][encoding]} is {solutions[i]}."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} 1's in node {node_of_interest}'s row, so the degree of node {node_of_interest} is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        # list all the edges that includes the node of interest
                        node_of_interest_edges = list(graphs[i].edges(str(node_of_interest)))
                        # convert the list into a string of edges represented as tuples: (node_a, node_b)
                        node_of_interest_edges_string = ', '.join([str(edge) for edge in node_of_interest_edges])
                        cot_prompt += f"after removing {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} has the following edges: {node_of_interest_edges_string}, so the degree of node {node_to_name[node_of_interest][encoding]} is {solutions[i]}."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} 1's in node {node_of_interest}'s row, so the degree of node {node_of_interest} is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        # list all the edges that includes the node of interest
                        node_of_interest_edges = list(graphs[i].edges(str(node_of_interest)))
                        # convert the list into a string of edges represented as tuples: (node_a, node_b)
                        node_of_interest_edges_string = ', '.join([str(edge) for edge in node_of_interest_edges])
                        cot_prompt += f"after adding {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} has the following edges: {node_of_interest_edges_string}, so the degree of node {node_to_name[node_of_interest][encoding]} is {solutions[i]}."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} 1's in node {node_mapping[str(node_of_interest)][str(mod_num)]}'s row, so the degree of node {node_mapping[str(node_of_interest)][str(mod_num)]} is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        # list all the edges that includes the node of interest
                        node_of_interest_edges = list(graphs[i].edges(str(node_of_interest)))
                        # convert the list into a string of edges represented as tuples: (node_a, node_b)
                        node_of_interest_edges_string = ', '.join([str(edge) for edge in node_of_interest_edges])
                        cot_prompt += f"after removing {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} has the following edges: {node_of_interest_edges_string}, so the degree of node {node_to_name[node_of_interest][encoding]} is {solutions[i]}."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix has {solutions[i]} 1's in node {node_mapping[str(node_of_interest)][str(mod_num)]}'s row, so the degree of node {node_mapping[str(node_of_interest)][str(mod_num)]} is {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"adding the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        # list all the edges that includes the node of interest
                        node_of_interest_edges = list(graphs[i].edges(str(node_of_interest)))
                        # convert the list into a string of edges represented as tuples: (node_a, node_b)
                        node_of_interest_edges_string = ', '.join([str(edge) for edge in node_of_interest_edges])
                        cot_prompt += f"after {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} has the following edges: {node_of_interest_edges_string}, so the degree of node {node_to_name[node_of_interest][encoding]} is {solutions[i]}."
            elif task == "edge_exists":
                nodes_of_interest = relavent_nodes_final[modification][str(chain_length)][task]
                if modification == "add_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has a 1 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is an edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                        else:
                            cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has a 0 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is no edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after adding {relavent_nodes_string}there is an edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                        else:
                            cot_prompt += f"after adding {relavent_nodes_string}there is no edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has a 1 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is an edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                        else:
                            cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix has a 0 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is no edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after removing {relavent_nodes_string}there is an edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                        else:
                            cot_prompt += f"after removing {relavent_nodes_string}there is no edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix has a 1 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is an edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                        else:
                            cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix has a 0 in both entries [{nodes_of_interest[0]}, {nodes_of_interest[1]}] and [{nodes_of_interest[1]}, {nodes_of_interest[0]}], so there is no edge between nodes {nodes_of_interest[0]} and {nodes_of_interest[1]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after adding {relavent_nodes_string}there is an edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                        else:
                            cot_prompt += f"after adding {relavent_nodes_string}there is no edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix has a 0 in both entries [{node_mapping[str(nodes_of_interest[0])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}] and [{node_mapping[str(nodes_of_interest[1])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[0])][str(mod_num)]}], so there is an edge between nodes {node_mapping[str(nodes_of_interest[0])][str(mod_num)]} and {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}."
                        else:
                            cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix has a 1 in both entries [{node_mapping[str(nodes_of_interest[0])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}] and [{node_mapping[str(nodes_of_interest[1])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[0])][str(mod_num)]}], so there is no edge between nodes {node_mapping[str(nodes_of_interest[0])][str(mod_num)]} and {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after removing {relavent_nodes_string}there is an edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                        else:
                            cot_prompt += f"after removing {relavent_nodes_string}there is no edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge" and mod_num != 1:
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge" and mod_num != 1:
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                            elif mix_task == "add_node" and mod_num != 1:
                                node = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                            elif mix_task == "remove_node" and mod_num != 1:
                                node = mix_nodes
                                if mod_num == chain_length:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix has a 0 in both entries [{node_mapping[str(nodes_of_interest[0])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}] and [{node_mapping[str(nodes_of_interest[1])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[0])][str(mod_num)]}], so there is an edge between nodes {node_mapping[str(nodes_of_interest[0])][str(mod_num)]} and {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}."
                        else:
                            cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix has a 1 in both entries [{node_mapping[str(nodes_of_interest[0])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}] and [{node_mapping[str(nodes_of_interest[1])][str(mod_num)]}, {node_mapping[str(nodes_of_interest[0])][str(mod_num)]}], so there is no edge between nodes {node_mapping[str(nodes_of_interest[0])][str(mod_num)]} and {node_mapping[str(nodes_of_interest[1])][str(mod_num)]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"adding the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        if solutions[i] == "Yes":
                            cot_prompt += f"after {relavent_nodes_string}there is an edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
                        else:
                            cot_prompt += f"after {relavent_nodes_string}there is no edge between nodes {node_to_name[nodes_of_interest[0]][encoding]} and {node_to_name[nodes_of_interest[1]][encoding]}."
            elif task == "connected_nodes":
                node_of_interest = relavent_nodes_final[modification][str(chain_length)][task]
                if modification == "add_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        cot_prompt += f"after changing {relavent_nodes_string}node {node_of_interest} has a 1 in the following entries: {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after adding {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} is connected to the following nodes: {solutions[i]}."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        cot_prompt += f"after changing {relavent_nodes_string}node {node_of_interest} has a 1 in the following entries: {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} is connected to the following nodes: {solutions[i]}."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        cot_prompt += f"after adding {relavent_nodes_string}node {node_of_interest} has a 1 in the following entries: {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        cot_prompt += f"after adding {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} is connected to the following nodes: {solutions[i]}."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}node {node_mapping[str(node_of_interest)][str(mod_num)]} has a 1 in the following entries: {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after removing {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} is connected to the following nodes: {solutions[i]}."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        #print(f"node_mapping: {node_mapping}")
                        #print(f"node_of_interest: ", node_of_interest)
                        #print(f"node_mapping[str(node_of_interest)]: ", node_mapping[str(node_of_interest)])
                        #print(f"mod_num: ", mod_num)
                        cot_prompt += f"after {relavent_nodes_string}node {node_mapping[str(node_of_interest)][str(mod_num)]} has a 1 in the following entries: {solutions[i]}."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after {relavent_nodes_string}node {node_to_name[node_of_interest][encoding]} is connected to the following nodes: {solutions[i]}."
            elif task == "print_graph":
                if modification == "add_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 0 to 1, "
                        cot_prompt += f"After changing {relavent_nodes_string}the resulting adjacency matrix is as defined below: \n{graph_to_string_encoder(solutions[i], encoding)}"
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting graph is as defined above."
                elif modification == "remove_edge":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                            else:
                                relavent_nodes_string += f"both entries [{node_a}, {node_b}] and [{node_b}, {node_a}] from 1 to 0, "
                        cot_prompt += f"after changing {relavent_nodes_string}the resulting adjacency matrix is as defined above."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node_a, node_b = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            else:
                                relavent_nodes_string += f"the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting graph is as defined above."
                elif modification == "add_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and a new row and column of zeros for node {node}, "
                            else:
                                relavent_nodes_string += f"a new row and column of zeros for node {node}, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting adjacency matrix is as defined above."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} without connecting it, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} without connecting it, "
                        cot_prompt += f"after adding {relavent_nodes_string}the resulting graph is as defined above."
                elif modification == "remove_node":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                            else:
                                relavent_nodes_string += f"the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting adjacency matrix is as defined above."
                    else:
                        for mod_num in range(1, chain_length+1):
                            node = relavent_nodes_mod[modification][str(mod_num)]
                            if mod_num == chain_length and mod_num != 1:
                                relavent_nodes_string += f"and the node {node_to_name[node][encoding]} and all its edges, "
                            else:
                                relavent_nodes_string += f"the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after removing {relavent_nodes_string}the resulting graph is as defined above."
                elif modification == "mix":
                    relavent_nodes_string = ""
                    if encoding == "adjacency_matrix":
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 0 to 1, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                                else:
                                    relavent_nodes_string += f"changing both entries [{node_mapping[str(node_a)][str(mod_num-1)]}, {node_mapping[str(node_b)][str(mod_num-1)]}] and [{node_mapping[str(node_b)][str(mod_num-1)]}, {node_mapping[str(node_a)][str(mod_num-1)]}] from 1 to 0, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                                else:
                                    relavent_nodes_string += f"adding a new row and column of zeros for node {node_mapping[str(node)][str(mod_num)]}, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                                else:
                                    relavent_nodes_string += f"removing the row and column for node {node_mapping[str(node)][str(mod_num-1)]}, "
                        cot_prompt += f"after {relavent_nodes_string}the resulting adjacency matrix is as defined above."
                    else:
                        for mod_num in range(1, chain_length+1):
                            mix_task, mix_nodes = relavent_nodes_mod[modification][str(mod_num)]
                            if mix_task == "add_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"adding an edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "remove_edge":
                                node_a, node_b = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                                else:
                                    relavent_nodes_string += f"removing the edge between {node_to_name[node_a][encoding]} and {node_to_name[node_b][encoding]}, "
                            elif mix_task == "add_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and adding the node {node_to_name[node][encoding]} without connecting it, "
                                else:
                                    relavent_nodes_string += f"adding the node {node_to_name[node][encoding]} without connecting it, "
                            elif mix_task == "remove_node":
                                node = mix_nodes
                                if mod_num == chain_length and mod_num != 1:
                                    relavent_nodes_string += f"and removing the node {node_to_name[node][encoding]} and all its edges, "
                                else:
                                    relavent_nodes_string += f"removing the node {node_to_name[node][encoding]} and all its edges, "
                        cot_prompt += f"after {relavent_nodes_string}the resulting graph is as defined above."
            cot_prompt += "\n\n"
        #print(f"Few-shot prompt: {few_shot_prompt}")
        #print(f"Cot prompt: {cot_prompt}")
        #sys.exit(0)
        return few_shot_prompt, cot_prompt

    # generate all prompts
    for task in final_tasks:
        for modification in modifications:
            for encoding in encodings:
                for chain_length in range(1, max_chain_length+1):
                    example_graphs = []
                    example_prompts = []
                    example_solutions = []
                    relavent_nodes_mods = []
                    relavent_nodes_finals = []
                    remove_node_mappings = []
                    mix_node_mappings = []
                    js = []
                    for example_num in range(1, 6):
                        print(f"Generating prompts for task {task}, modification {modification}, encoding {encoding}, example number {example_num}, and chain length {chain_length}...")
                        # Generate a random integer from 0 to 149
                        random.seed(i + example_num)
                        j = random.randint(0, 149)
                        while j in js:
                            j = random.randint(0, 149)

                        js.append(j)
                        #print(f"j: {j}")

                        # Sample a random graph
                        example_graph_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/input_graphs/{j}.graphml"
                        example_graph = nx.read_graphml(example_graph_filename)
                        example_graphs.append(example_graph)

                        # Sample a random prompt
                        example_prompt_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/prompts/prompt_{j}.txt"
                        with open(example_prompt_filename, "r") as prompt_file:
                            example_prompt = prompt_file.read()
                        example_prompts.append(example_prompt)

                        # Sample a random solution
                        if task == "print_graph":
                            example_solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{j}.graphml"
                            example_solution = nx.read_graphml(example_solution_filename)
                        else:
                            example_solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{j}.txt"
                            with open(example_solution_filename, "r") as solution_file:
                                example_solution = solution_file.read()
                        example_solutions.append(example_solution)

                        # Read relavent_nodes_mod dictionary 
                        directory = "data/dicts/mod"
                        with open(f"{directory}/relavent_nodes_mod_{j}.json", "r") as f:
                            relavent_nodes_mod = json.load(f)
                        relavent_nodes_mods.append(relavent_nodes_mod)

                        #print(f"example_prompt: {example_prompt}")
                        #print(f"Relavent nodes mod: {relavent_nodes_mod}")

                        #for k in relavent_nodes_mod[modification].keys():
                        #    k = int(k)

                        # Read relavent_nodes_final dictionary
                        directory = "data/dicts/final" 
                        with open(f"data/dicts/final/relavent_nodes_final_{j}.json", "r") as f:
                            relavent_nodes_final = json.load(f)
                        relavent_nodes_finals.append(relavent_nodes_final)

                        #for k in relavent_nodes_final[modification].keys():
                        #    k = int(k)

                        # Read remove_node_mapping dictionary
                        directory = "data/dicts/remove_node_mapping"
                        with open(f"data/dicts/remove_node_mapping/remove_node_mapping_{j}.json", "r") as f:
                            remove_node_mapping = json.load(f)
                        remove_node_mappings.append(remove_node_mapping)

                        # mix_node_mapping[node][chain_length]
                        #for k, d in mix_node_mapping:
                        #    k = int(k)

                        # Read mix_node_mapping dictionary
                        directory = "data/dicts/mix_node_mapping"
                        with open(f"data/dicts/mix_node_mapping/mix_node_mapping_{j}.json", "r") as f:
                            mix_node_mapping = json.load(f)
                        mix_node_mappings.append(mix_node_mapping)

                        # construct few-shot and cot prompts
                        if modification == "mix":
                            few_prompt, cot_prompt = examples_to_prompt(example_graphs, example_prompts, example_solutions, task, encoding, modification, chain_length, relavent_nodes_mods, relavent_nodes_finals, mix_node_mappings)
                        elif modification == "remove_node":
                            few_prompt, cot_prompt = examples_to_prompt(example_graphs, example_prompts, example_solutions, task, encoding, modification, chain_length, relavent_nodes_mods, relavent_nodes_finals, remove_node_mappings)
                        else:
                            few_prompt, cot_prompt = examples_to_prompt(example_graphs, example_prompts, example_solutions, task, encoding, modification, chain_length, relavent_nodes_mods, relavent_nodes_finals)

                        """
                        # construct init_prompt
                        init_prompt = graph_to_init_prompt(nodes, encoding, node_to_name)

                        # construct graph string
                        graph_string = graph_to_string_encoder(graph, encoding, node_to_name)

                        # construct modification prompt
                        modification_prompt_str = modification_prompt(encoding, modification, chain_length, relavent_nodes_mod)

                        # construct question prompt
                        question_prompt_str = question_prompt(task, encoding, modification, chain_length, relavent_nodes_final)

                        # construct solution
                        solution = solutions[modification][chain_length][task]

                        # construct full prompt
                        full_prompt = init_prompt + graph_string + modification_prompt_str + question_prompt_str # + solution + cot
                        """

                        print(f"Few-shot prompt: {few_prompt}")
                        print(f"Cot prompt: {cot_prompt}")

                        #sys.exit(0)

                        # Read graph
                        graph_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/input_graphs/{i}.graphml"
                        graph = nx.read_graphml(graph_filename)

                        # Read prompt   
                        prompt_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/prompts/prompt_{i}.txt"
                        with open(prompt_filename, "r") as prompt_file:
                            prompt = prompt_file.read()

                        # Read solution
                        if task == "print_graph":
                            solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{i}.graphml"
                            solution = nx.read_graphml(solution_filename)
                        else:
                            solution_filename = f"data/{encoding}_chain_no_print/{task}/{modification}/{chain_length}/solutions/solution_{i}.txt"
                            with open(solution_filename, "r") as solution_file:
                                solution = solution_file.read()

                        # Construct few shot prompt
                        final_few_prompt = few_prompt + prompt

                        # Construct cot prompt
                        final_cot_prompt = cot_prompt + prompt

                        # Save few shot graph to file
                        few_graph_filename = f"data/{encoding}_chain_big_few/{task}/{modification}/{chain_length}/{example_num}/input_graphs/{i}.graphml"
                        nx.write_graphml(graph, few_graph_filename)

                        # Save few shot prompt to file
                        few_prompt_filename = f"data/{encoding}_chain_big_few/{task}/{modification}/{chain_length}/{example_num}/prompts/prompt_{i}.txt"
                        with open(few_prompt_filename, "w") as few_prompt_file:
                            few_prompt_file.write(final_few_prompt)

                        # Save few shot solution to file
                        if task == "print_graph":
                            # Write solution to file
                            few_solution_filename = f"data/{encoding}_chain_big_few/{task}/{modification}/{chain_length}/{example_num}/solutions/solution_{i}.graphml"
                            nx.write_graphml(solution, few_solution_filename)
                        else:
                            # Write solution to file
                            few_solution_filename = f"data/{encoding}_chain_big_few/{task}/{modification}/{chain_length}/{example_num}/solutions/solution_{i}.txt"
                            with open(few_solution_filename, "w") as few_solution_file:
                                few_solution_file.write(solution)

                        # Save cot graph to file
                        cot_graph_filename = f"data/{encoding}_chain_big_cot/{task}/{modification}/{chain_length}/{example_num}/input_graphs/{i}.graphml"
                        nx.write_graphml(graph, cot_graph_filename)

                        # Save cot prompt to file
                        cot_prompt_filename = f"data/{encoding}_chain_big_cot/{task}/{modification}/{chain_length}/{example_num}/prompts/prompt_{i}.txt"
                        with open(cot_prompt_filename, "w") as cot_prompt_file:
                            cot_prompt_file.write(final_cot_prompt)

                        # Save cot solution to file
                        if task == "print_graph":
                            # Write solution to file
                            cot_solution_filename = f"data/{encoding}_chain_big_cot/{task}/{modification}/{chain_length}/{example_num}/solutions/solution_{i}.graphml"
                            nx.write_graphml(solution, cot_solution_filename)
                        else:
                            # Write solution to file
                            cot_solution_filename = f"data/{encoding}_chain_big_cot/{task}/{modification}/{chain_length}/{example_num}/solutions/solution_{i}.txt"
                            with open(cot_solution_filename, "w") as cot_solution_file:
                                cot_solution_file.write(solution)

def node_count(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Node count  ---
    # ----------------------------

    node_count = graph.number_of_nodes()
    if in_context:
        # Create prompt string
        node_count_prompt = f"Q: How many nodes are in the resulting graph?\n"
        full_node_count_prompt = init_prompt + graph_str + "\n" + node_count_prompt + end_prompt

        #if few:
        #    full_node_count_prompt += f"{node_count}"
        #else:
        #    full_node_count_prompt += f"{node_count}, because the adjacency matrix has {node_count} rows."

        #return full_node_count_prompt

        full_node_count_prompt_few = full_node_count_prompt + f"{node_count}"
        full_node_count_prompt_cot = full_node_count_prompt + f"{node_count}, because the adjacency matrix has {node_count} rows."

        return full_node_count_prompt_few, full_node_count_prompt_cot
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

def edge_count(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Edge count  ---
    # ----------------------------

    edge_count = graph.number_of_edges()
    if in_context:
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in the resulting graph?\n"
        full_edge_count_prompt = init_prompt + graph_str + "\n" + edge_count_prompt + end_prompt

        #if few:
        #    full_edge_count_prompt += f"{edge_count}"
        #else:
        #    full_edge_count_prompt += f"{edge_count}, because the adjacency matrix has {edge_count*2} non-zero entries, and because the graph is undirected, there are {edge_count*2} // 2 = {edge_count} edges."

        #return full_edge_count_prompt

        full_edge_count_prompt_few = full_edge_count_prompt + f"{edge_count}"
        full_edge_count_prompt_cot = full_edge_count_prompt + f"{edge_count}, because the adjacency matrix has {edge_count*2} non-zero entries, and because the graph is undirected, there are {edge_count*2} // 2 = {edge_count} edges."

        return full_edge_count_prompt_few, full_edge_count_prompt_cot
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

def node_degree(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Node degree  ---
    # ----------------------------

    # Select a random node
    node = random.choice(list(graph.nodes()))
    node_degree = graph.degree[node]

    if in_context:
        # Create prompt string
       
        node_degree_prompt = f"Q: How many neighbors does node {node} have in the resulting graph?\n"
        full_node_degree_prompt = init_prompt + graph_str + "\n" + node_degree_prompt + end_prompt

        #if few:
        #    full_node_degree_prompt += f"{node_degree}"
        #else:
        #    full_node_degree_prompt += f"{node_degree}, because the adjacency matrix has {node_degree} non-zero entries in row {node}."

        #return full_node_degree_prompt

        full_node_degree_prompt_few = full_node_degree_prompt + f"{node_degree}"
        full_node_degree_prompt_cot = full_node_degree_prompt + f"{node_degree}, because the adjacency matrix has {node_degree} non-zero entries in row {node}."

        return full_node_degree_prompt_few, full_node_degree_prompt_cot
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

def edge_exists(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Edge exists  ---
    # ----------------------------

    # Select two random nodes from the graph
    random_nodes = random.sample(list(graph.nodes()), 2)
    node_a, node_b = random_nodes

    if in_context:
        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b} in the resulting graph?\n"
        full_edge_exists_prompt = init_prompt + graph_str + "\n" + edge_exists_prompt + end_prompt

        if graph.has_edge(node_a, node_b):
            #if few:
            #    full_edge_exists_prompt += f"Yes"
            #else:
            #    full_edge_exists_prompt += f"Yes, because the adjacency matrix has a 1 in entries {node_a},{node_b} and {node_b},{node_a}."
            full_edge_exists_prompt_few = full_edge_exists_prompt + f"Yes"
            full_edge_exists_prompt_cot = full_edge_exists_prompt + f"Yes, because the adjacency matrix has a 1 in entries {node_a},{node_b} and {node_b},{node_a}."
        else:
            #if few:
            #    full_edge_exists_prompt += f"No"
            #else:
            #    full_edge_exists_prompt += f"No, because the adjacency matrix has a 0 in entries {node_a},{node_b} and {node_b},{node_a}."

            full_edge_exists_prompt_few = full_edge_exists_prompt + f"No"
            full_edge_exists_prompt_cot = full_edge_exists_prompt + f"No, because the adjacency matrix has a 0 in entries {node_a},{node_b} and {node_b},{node_a}."

        return full_edge_exists_prompt_few, full_edge_exists_prompt_cot
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

def connected_nodes(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Connected nodes  ---
    # ----------------------------

    # Select one node from the graph that has at least one neighbor
    nodes_with_neighbors = [node for node in graph.nodes() if graph.degree[node] > 0]
    node = random.choice(nodes_with_neighbors)

    if in_context:
        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node} in the resulting graph.\n"
        full_connected_nodes_prompt = init_prompt + graph_str + "\n" + connected_nodes_prompt + end_prompt

        connected_nodes = sorted([node_b for node_b in graph.neighbors(node)])
        if len(connected_nodes) == 0:
            #if few:
            #    full_connected_nodes_prompt += f"{connected_nodes}"
            #else:
            #    full_connected_nodes_prompt += f"{connected_nodes}, because the adjacency matrix has a row of zeros corresponding to node {node}."

            full_connected_nodes_prompt_few = full_connected_nodes_prompt + f"{connected_nodes}"
            full_connected_nodes_prompt_cot = full_connected_nodes_prompt + f"{connected_nodes}, because the adjacency matrix has a row of zeros corresponding to node {node}."
            
        else:
            neighbors_str = ''
            for node_b in connected_nodes:
                if node_b == connected_nodes[-1]:
                    neighbors_str += f"and ({node},{node_b})."
                else:
                    neighbors_str += f"({node},{node_b}), "
            #if few:
            #    full_connected_nodes_prompt += f"{connected_nodes}"
            #else:
            #    full_connected_nodes_prompt += f"{connected_nodes}, because the adjacency matrix has a 1 in entries {neighbors_str}"

            full_connected_nodes_prompt_few = full_connected_nodes_prompt + f"{connected_nodes}"
            full_connected_nodes_prompt_cot = full_connected_nodes_prompt + f"{connected_nodes}, because the adjacency matrix has a 1 in entries {neighbors_str}."

        return full_connected_nodes_prompt_few, full_connected_nodes_prompt_cot
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

def cycle(graph, graph_str, init_prompt, end_prompt, i, ablation_dir = "", ablationType = None, in_context = False, graph_type = 'adjacency', encoding_dict = None):
    # ----------------------------
    # --- Cycle  ---
    # ----------------------------

    if in_context:
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
            #if few:
            #    full_cycle_prompt += "Yes"
            #else:
            #    full_cycle_prompt += "Yes, because " + cycle_str
            full_cycle_prompt_few = full_cycle_prompt + "Yes"
            full_cycle_prompt_cot = full_cycle_prompt + "Yes, because " + cycle_str
        except nx.NetworkXNoCycle:
            #if few:
            #    full_cycle_prompt += "No"
            #else:
            #    full_cycle_prompt += "No, because the adjacency matrix does not have a cycle."

            full_cycle_prompt_few = full_cycle_prompt + "No"
            full_cycle_prompt_cot = full_cycle_prompt + "No, because the adjacency matrix does not have a cycle."

        return full_cycle_prompt_few, full_cycle_prompt_cot
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