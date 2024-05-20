
import os
import networkx as nx
import argparse
import random
import sys

def add_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain):
    # ----------------------------
    # --- Add edge  ---
    # ----------------------------

    # Select two random nodes that are not connected
    unconnected_nodes = []
    for node_a in graph.nodes():
        for node_b in graph.nodes():
            if node_a != node_b and not graph.has_edge(node_a, node_b): # TODO: allow for self-loops?
                unconnected_nodes.append((node_a, node_b))
    #print(f"unconnected_nodes: {unconnected_nodes}")
    node_a, node_b = random.sample(unconnected_nodes, 1)[0]

    if part_of_chain:
        add_edge_prompt = f"{i}: Add an edge between node {node_a} and node {node_b}.\n"
        # Create new graph with added edge
        add_edge_graph = graph.copy()
        add_edge_graph.add_edge(node_a, node_b)
        return add_edge_graph, add_edge_prompt
    else:
        # Create prompt string
        add_edge_prompt = f"Q: Add an edge between node {node_a} and node {node_b}. Only write the resulting adjacency matrix.\n"
        full_add_edge_prompt = init_prompt + graph_str + "\n" + add_edge_prompt + end_prompt

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

        return add_edge_graph, add_edge_prompt

def remove_edge(graph, graph_str, init_prompt, end_prompt, i, part_of_chain):
    # ----------------------------
    # --- Remove edge  ---
    # ----------------------------

    # Select a random edge
    edge = random.choice(list(graph.edges()))

    if part_of_chain:
        remove_edge_prompt = f"{i}: Remove the edge between node {edge[0]} and node {edge[1]}.\n"
        # Create new graph with edge removed
        remove_edge_graph = graph.copy()
        remove_edge_graph.remove_edge(*edge)
        return remove_edge_graph, remove_edge_prompt
    else:
        # Create prompt string
        node_a, node_b = edge
        remove_edge_prompt = f"Q: Remove the edge between node {node_a} and node {node_b}. Only write the resulting adjacency matrix.\n"
        full_remove_edge_prompt = init_prompt + graph_str + "\n" + remove_edge_prompt + end_prompt

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

        return remove_edge_graph, remove_edge_prompt

def add_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain):
    # ----------------------------
    # --- Add node  ---
    # ----------------------------
    number_of_nodes = graph.number_of_nodes()

    if part_of_chain:
        add_node_prompt = f"{i}: Add a node to the graph.\n"
        # Create new graph with added node
        add_node_graph = graph.copy()
        add_node_graph.add_node(number_of_nodes)
        return add_node_graph, add_node_prompt
    else:
        # Create prompt string
        add_node_prompt = f"Q: Add a node to the graph. Only write the resulting adjacency matrix.\n"
        full_add_node_prompt = init_prompt + graph_str + "\n" + add_node_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/add_node/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_add_node_prompt)

        #print(f'Original graph: {graph_str}')

        # Add a node to the graph
        add_node_graph = graph.copy()
        #print(f'add_node_graph.nodes(): {add_node_graph.nodes()}')
        add_node_graph.add_node(number_of_nodes)
        #print(f'add_node_graph.nodes() after adding new node: {add_node_graph.nodes()}')

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(add_node_graph).todense().astype(int))

        #print(f'New graph after adding new node: {new_graph_str}')
        
        # Write new graph to file
        solution_filename = f"data/solutions/add_node/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)

        return add_node_graph, add_node_prompt

def remove_node(graph, graph_str, init_prompt, end_prompt, i, part_of_chain):
    # ----------------------------
    # --- Remove node  ---
    # ----------------------------

    # Select a random node
    node = random.choice(list(graph.nodes()))

    if part_of_chain:
        remove_node_prompt = f"{i}: Remove node {node} from the graph.\n"
        # Create new graph with node removed
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)
        return remove_node_graph, remove_node_prompt
    else:
        # Create prompt string
        remove_node_prompt = f"Q: Remove node {node} from the graph. Only write the resulting adjacency matrix.\n"
        full_remove_node_prompt = init_prompt + graph_str + "\n" + remove_node_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/remove_node/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_remove_node_prompt)

        #print(f'Original graph: {graph_str}')

        # Remove node from the graph
        remove_node_graph = graph.copy()
        remove_node_graph.remove_node(node)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(remove_node_graph).todense().astype(int))

        #print(f'New graph: {new_graph_str}')

        # Write new graph to file
        solution_filename = f"data/solutions/remove_node/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)

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
        node_count_prompt = f"Q: How many nodes are in this graph?\n" # TODO: in the resulting graph instead?
        full_chain_prompt += node_count_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_node_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_node_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_count))
    elif final_task == "edge_count":
        edge_count = chain_graph.number_of_edges()
        # Create prompt string
        edge_count_prompt = f"Q: How many edges are in this graph?\n"  # TODO: in the resulting graph instead?
        full_chain_prompt += edge_count_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_edge_count/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_edge_count/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(edge_count))
    elif final_task == "node_degree":
        # Select a random node
        node = random.choice(list(chain_graph.nodes()))
        node_degree = chain_graph.degree[node]

        # Create prompt string
        node_degree_prompt = f"Q: How many neighbors does node {node} have?\n"  # TODO: In the resulting graph,... instead?
        full_chain_prompt += node_degree_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_node_degree/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_node_degree/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(str(node_degree))
    elif final_task == "edge_exists":
        # Select two random nodes from the graph
        random_nodes = random.sample(list(chain_graph.nodes()), 2)
        node_a, node_b = random_nodes

        edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}? Only write 'Yes' or 'No'.\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += edge_exists_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_edge_exists/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_edge_exists/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            if chain_graph.has_edge(node_a, node_b):
                solution_file.write("Yes")
            else:
                solution_file.write("No")
    elif final_task == "connected_nodes":
        # Select one node from the graph that has at least one neighbor
        nodes_with_neighbors = [node for node in chain_graph.nodes() if chain_graph.degree[node] > 0]
        node = random.choice(nodes_with_neighbors)

        # Create prompt string
        connected_nodes_prompt = f"Q: List all neighbors of node {node}.\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += connected_nodes_prompt + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_connected_nodes/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Save solution to file
        solution_filename = f"data/solutions/chain_{chain_length}_connected_nodes/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            connected_nodes = sorted([node_b for node_b in chain_graph.neighbors(node)])
            solution_file.write(str(connected_nodes))
    elif final_task == "cycle":
        # Create prompt string
        cycle_prompt = f"Q: Does this graph contain a cycle? Only write 'Yes' or 'No'.\n" # TODO: In the resulting graph,... instead?
        full_chain_prompt += cycle_prompt + end_prompt

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
            except nx.NetworkXNoCycle:
                solution_file.write("No")
    elif final_task == "print_adjacency_matrix":
        full_chain_prompt += f"What is the resulting adjacency matrix?\n" + end_prompt

        # Save prompt to file
        prompt_filename = f"data/prompts/chain_{chain_length}_print/prompt_{i}.txt"
        with open(prompt_filename, "w") as prompt_file:
            prompt_file.write(full_chain_prompt)

        # Convert graph to string
        new_graph_str = str(nx.adjacency_matrix(chain_graph).todense().astype(int))

        # Write new graph to file
        solution_filename = f"data/solutions/chain_{chain_length}_print/solution_{i}.txt"
        with open(solution_filename, "w") as solution_file:
            solution_file.write(new_graph_str)
    else:
        print("Final task not recognized. Exiting.")
        sys.exit(1)

    return chain_graph

def node_count(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Node count  ---
    # ----------------------------

    node_count = graph.number_of_nodes()
    # Create prompt string
    node_count_prompt = f"Q: How many nodes are in this graph?\n"
    full_node_count_prompt = init_prompt + graph_str + "\n" + node_count_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/prompts/node_count/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_node_count_prompt)

    # Save solution to file
    solution_filename = f"data/solutions/node_count/solution_{i}.txt"
    with open(solution_filename, "w") as solution_file:
        solution_file.write(str(node_count))

def edge_count(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Edge count  ---
    # ----------------------------

    edge_count = graph.number_of_edges()
    # Create prompt string
    edge_count_prompt = f"Q: How many edges are in this graph?\n" # TODO: in this undirected graph instead?
    full_edge_count_prompt = init_prompt + graph_str + "\n" + edge_count_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/prompts/edge_count/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_edge_count_prompt)

    # Save solution to file
    solution_filename = f"data/solutions/edge_count/solution_{i}.txt"
    with open(solution_filename, "w") as solution_file:
        solution_file.write(str(edge_count))

def node_degree(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Node degree  ---
    # ----------------------------

    # Select a random node
    node = random.choice(list(graph.nodes()))
    node_degree = graph.degree[node]

    # Create prompt string
    node_degree_prompt = f"Q: How many neighbors does node {node} have?\n" # TODO: How many neighbors does node {node} have?/What is the degree of node {node}?/How many edges are connected to node {node}?
    full_node_degree_prompt = init_prompt + graph_str + "\n" + node_degree_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/prompts/node_degree/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_node_degree_prompt)

    # Save solution to file
    solution_filename = f"data/solutions/node_degree/solution_{i}.txt"
    with open(solution_filename, "w") as solution_file:
        solution_file.write(str(node_degree))

def edge_exists(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Edge exists  ---
    # ----------------------------

    # Select two random nodes from the graph
    random_nodes = random.sample(list(graph.nodes()), 2)
    node_a, node_b = random_nodes

    edge_exists_prompt = f"Q: Is node {node_a} connected to node {node_b}? Only write 'Yes' or 'No'.\n"
    full_edge_exists_prompt = init_prompt + graph_str + "\n" + edge_exists_prompt + end_prompt

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

def connected_nodes(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Connected nodes  ---
    # ----------------------------

    # Select one node from the graph that has at least one neighbor
    nodes_with_neighbors = [node for node in graph.nodes() if graph.degree[node] > 0]
    node = random.choice(nodes_with_neighbors)

    # Create prompt string
    connected_nodes_prompt = f"Q: List all neighbors of node {node}.\n" #in ascending order, and surround your final answer in brackets, like this: [answer]
    full_connected_nodes_prompt = init_prompt + graph_str + "\n" + connected_nodes_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/prompts/connected_nodes/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_connected_nodes_prompt)

    # Save solution to file
    solution_filename = f"data/solutions/connected_nodes/solution_{i}.txt"
    with open(solution_filename, "w") as solution_file:
        connected_nodes = sorted([node_b for node_b in graph.neighbors(node)])
        solution_file.write(str(connected_nodes))

def cycle(graph, graph_str, init_prompt, end_prompt, i):
    # ----------------------------
    # --- Cycle  ---
    # ----------------------------

    # Create prompt string
    cycle_prompt = f"Q: Does this graph contain a cycle? Only write 'Yes' or 'No'.\n"
    full_cycle_prompt = init_prompt + graph_str + "\n" + cycle_prompt + end_prompt

    # Save prompt to file
    prompt_filename = f"data/prompts/cycle/prompt_{i}.txt"
    with open(prompt_filename, "w") as prompt_file:
        prompt_file.write(full_cycle_prompt)

    # Save solution to file
    solution_filename = f"data/solutions/cycle/solution_{i}.txt"
    with open(solution_filename, "w") as solution_file:
        try:
            nx.find_cycle(graph)
            solution_file.write("Yes")
        except nx.NetworkXNoCycle:
            solution_file.write("No")
        