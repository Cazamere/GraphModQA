# Description: This file contains utility functions for evaluating the model's output.
import networkx as nx
import sys
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from openai import OpenAI
from fireworks.client import Fireworks


def adjacency_matrix_from_string(adj_matrix_str):
        adj_matrix_str = adj_matrix_str.replace(" ", "")
        print("Adjacency matrix string input: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace(" ", "")
        if "],\n" not in adj_matrix_str:
            adj_matrix_str = adj_matrix_str.replace("], ", "],\n")
        adj_matrix_str = adj_matrix_str.replace("[[", "[")
        adj_matrix_str = adj_matrix_str.replace("]]", "]")
        print("Adjacency matrix string after removing double brackets: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace(",", "")
        print("Adjacency matrix string after replacing commas: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace("  ", " ")
        adj_matrix_str = adj_matrix_str.replace(" [", "[")
        adj_matrix_str = adj_matrix_str.replace("```", "")
        adj_matrix_str = adj_matrix_str.replace("*", "")
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        adj_matrix_str = re.sub(r'#.*', '', adj_matrix_str)
        print("Adjacency matrix string after removing double spaces and ```: ", adj_matrix_str)
        adj_matrix_list = adj_matrix_str.split('\n')
        if adj_matrix_list[0] == '':
            adj_matrix_list = adj_matrix_list[1:]
        if adj_matrix_list[-1] == '':
            adj_matrix_list = adj_matrix_list[:-1]
        print("Adjacency matrix list after newline split: ", adj_matrix_list)
        adj_matrix = [[int(num) for num in row.replace(" [", "[").replace("] ", "]").strip("[]").replace(" ", "")] for row in adj_matrix_list]
        print("Adjacency matrix string before being converted into a np array: ", adj_matrix_str)
        adj_matrix = np.array(adj_matrix)
        return adj_matrix

def adjacency_matrix_from_latex_string(adj_matrix_str):
        #print("Adjacency matrix string input: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace("\\begin{pmatrix}", "").replace("\\end{pmatrix}", "")
        adj_matrix_str = adj_matrix_str.replace("\\begin{bmatrix}", "").replace("\\end{bmatrix}", "")
        adj_matrix_str = adj_matrix_str.replace("&", "")
        #print("Adjacency matrix string after replacing commas: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace("  ", " ")
        adj_matrix_str = adj_matrix_str.replace("\\", "[")
        adj_matrix_str = adj_matrix_str.replace("```", "")
        adj_matrix_str = adj_matrix_str.replace("*", "")
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        adj_matrix_str = re.sub(r'#.*', '', adj_matrix_str)
        #print("Adjacency matrix string after removing double spaces and ```: ", adj_matrix_str)
        adj_matrix_list = adj_matrix_str.split('\n')
        if adj_matrix_list[0] == '':
            adj_matrix_list = adj_matrix_list[1:]
        if adj_matrix_list[-1] == '':
            adj_matrix_list = adj_matrix_list[:-1]
        #print("Adjacency matrix list  after newline split: ", adj_matrix_list)
        adj_matrix = [[int(num) for num in row.replace(" ", "")] for row in adj_matrix_list]
        #print("Adjacency matrix string before being converted into a np array: ", adj_matrix_str)
        adj_matrix = np.array(adj_matrix)
        return adj_matrix

def error_analysis(original_graph, model_graph, answer_graph, task): # there's an original graph, answer graph, and input graph, think about these a bit
    # Convert graph into scipy dense sparse matrix
    print()
    print("ENTERED ERROR ANALYSIS!!!")
    print()

    original_graph_matrix = nx.adjacency_matrix(original_graph).todense()
    """
    if latex:
        print("Trying to convert model_graph into adjacency matrix using adjacency_matrix_from_latex_string")
        model_graph_matrix = adjacency_matrix_from_latex_string(model_graph)
    else:
        print("Trying to convert model_graph into adjacency matrix using adjacency_matrix_from_string")
        model_graph_matrix = adjacency_matrix_from_string(model_graph)
    """

    print("Trying to convert model_graph into adjacency matrix using adjacency_matrix_from_string")
    print("Model graph: ", model_graph)
    try:
        model_graph_matrix = adjacency_matrix_from_string(model_graph)
    except:
        print("Model graph string error, possibly due to incorrect dimensions")
        print(model_graph)
        #sys.exit(1)
        return "dimension error"

    # Convert model_graph, which is currently a string of an adjacency matrix using [[row 1], [row 2], ...], into scipy dense sparse matrix
    """
    try:
        if latex:
            print("Trying to convert model_graph into adjacency matrix using adjacency_matrix_from_latex_string")
            model_graph_matrix = adjacency_matrix_from_latex_string(model_graph)
        else:
            print("Trying to convert model_graph into adjacency matrix using adjacency_matrix_from_string")
            model_graph_matrix = adjacency_matrix_from_string(model_graph)
    except:
        print("Model graph string error, possibly due to incorrect dimensions")
        print(model_graph)
        sys.exit(1)
        
        print(f'Attempt to convert final_answer into adjacency matrix: {final_answer}')

        try:
            print(f'Length of final_answer: {len(final_answer)}')
            # check if len is a square number
            if math.sqrt(len(final_answer)) % 1 != 0:
                print("Len is NOT a square number!")
            else:
                print("Len IS a square number!")
            model_graph_matrix = np.array([int(n) for n in final_answer]).reshape(int(math.sqrt(len(final_answer))), int(math.sqrt(len(final_answer))))
        except:
            print("Model graph string error, possibly due to incorrect dimensions")
            print(final_answer)
            sys.exit(0)
            return "model graph string error, possibly due to incorrect dimensions"
            sys.exit(0)
            return "model graph string error, possibly due to incorrect dimensions"
        """

    answer_graph_matrix = nx.adjacency_matrix(answer_graph).todense()

    print("Original graph matrix: ", original_graph_matrix)
    print("Model graph matrix: ", model_graph_matrix)
    print("Answer graph matrix: ", answer_graph_matrix)
    
    # Check missing row: check if the answer graph has an extra row compared to the model graph
    if model_graph_matrix.shape[0] < answer_graph_matrix.shape[0]:
        return "missing row"

    # Check missing column: remove the rightmost column of graph_matrix
    if model_graph_matrix.shape[1] < answer_graph_matrix.shape[1]:
        return "missing column"

    # Check extra row: Add an extra row of zeros to the bottom of graph_matrix
    if model_graph_matrix.shape[0] > answer_graph_matrix.shape[0]:
        return "extra row"

    # Check extra column: Add an extra column of zeros to the right of graph_matrix
    if model_graph_matrix.shape[1] > answer_graph_matrix.shape[1]:
        return "extra column"
    
    # Check if model did not modify the original graph matrix
    if np.array_equal(model_graph_matrix, original_graph_matrix):
        return "no modification"

    # Check added self loop: see if bottom right entry is a 1
    if model_graph_matrix[-1, -1] == 1:
        return "added self loop"
    
    if task == "add_edge":

        #for index in diff_answer_original:
        #    print("index: ", index)
        #    x = index[0]
        #    y = index[1]
        #    print(f"x: {x}, y: {y}")
        #    print(correct_change_array[x, y])
            

        # Check if model added only one edge
        if np.sum(answer_graph_matrix - model_graph_matrix) == 1:
            return "added only one edge"

        # Find the entries that are different between answer_graph_matrix and original_graph_matrix
        diff_answer_original = np.where(answer_graph_matrix - original_graph_matrix == 1)
        diff_answer_model = np.where(answer_graph_matrix - model_graph_matrix == 1)

        for x_answer_model, y_answer_model in zip(diff_answer_model[0], diff_answer_model[1]): # this gives every [x,y] index pair where the model's output graph is different from the correct answer graph
            for x_answer_original, y_answer_original in zip(diff_answer_original[0], diff_answer_original[1]): # this gives every [x,y] index pair where the original graph is different from the correct answer graph
                # Check if x_answer_model, y_answer_model is adjacent to x_answer_original, y_answer_original
                #if (abs(x_answer_model - x_answer_original) == 1 and y_answer_model == y_answer_original) or (abs(y_answer_model - y_answer_original) == 1 and x_answer_model == x_answer_original):
                if (abs(x_answer_model - x_answer_original) == 1) or (abs(y_answer_model - y_answer_original) == 1):
                    # Check if the 1's in diff_answer_model are adjacent to the 1's in diff_answer_original
                    print("Diff answer original: ", diff_answer_original)
                    print("Diff answer model: ", diff_answer_model)
                    correct_change_array = np.asarray(answer_graph_matrix - original_graph_matrix == 1)
                    print(np.asarray(answer_graph_matrix - original_graph_matrix == 1))
                    print(np.asarray(answer_graph_matrix - model_graph_matrix == 1))
                    print((np.asarray(answer_graph_matrix - model_graph_matrix == 1)).nonzero())

                    print("x_answer_model: ", x_answer_model)
                    print("y_answer_model: ", y_answer_model)
                    print("x_answer_original: ", x_answer_original)
                    print("y_answer_original: ", y_answer_original)
                    return "added to adjacent entry"
                
        return "added to non-adjacent entry"

    if task == "remove_edge":
        # Check if model removed only one edge
        if np.sum(answer_graph_matrix - model_graph_matrix) == -1:
            return "removed only one edge"
        
        # Find the entries that are different between answer_graph_matrix and original_graph_matrix
        diff_answer_original = np.where(original_graph_matrix - answer_graph_matrix == 1)
        diff_answer_model = np.where(model_graph_matrix - answer_graph_matrix == 1)

        for x_answer_model, y_answer_model in zip(diff_answer_model[0], diff_answer_model[1]): # this gives every [x,y] index pair where the model's output graph is different from the correct answer graph
            for x_answer_original, y_answer_original in zip(diff_answer_original[0], diff_answer_original[1]): # this gives every [x,y] index pair where the original graph is different from the correct answer graph
                # Check if x_answer_model, y_answer_model is adjacent to x_answer_original, y_answer_original
                #if (abs(x_answer_model - x_answer_original) == 1 and y_answer_model == y_answer_original) or (abs(y_answer_model - y_answer_original) == 1 and x_answer_model == x_answer_original):
                if (abs(x_answer_model - x_answer_original) == 1) or (abs(y_answer_model - y_answer_original) == 1):
                    # Check if the 1's in diff_answer_model are adjacent to the 1's in diff_answer_original
                    print("Diff answer original: ", diff_answer_original)
                    print("Diff answer model: ", diff_answer_model)
                    correct_change_array = np.asarray(answer_graph_matrix - original_graph_matrix == 1)
                    print(np.asarray(answer_graph_matrix - original_graph_matrix == 1))
                    print(np.asarray(answer_graph_matrix - model_graph_matrix == 1))
                    print((np.asarray(answer_graph_matrix - model_graph_matrix == 1)).nonzero())

                    print("x_answer_model: ", x_answer_model)
                    print("y_answer_model: ", y_answer_model)
                    print("x_answer_original: ", x_answer_original)
                    print("y_answer_original: ", y_answer_original)
                    return "removed from adjacent entry"
                
        return "removed from non-adjacent entry"

    if task == "add_node":
        # Check if model added edges to the added node
        if 1 in model_graph_matrix[-1, :] or 1 in model_graph_matrix[:, -1]:
            return "connected the added node"
        
    if task == "remove_node":
        # Check if the model removed the wrong node

        # Iterate through every node in original_graph (which is a networkx graph)
        for node in original_graph.nodes():
            # create a new graph with the node removed
            new_graph = original_graph.copy()
            new_graph.remove_node(node)

            # convert the new graph into an adjacency matrix
            new_graph_matrix = nx.adjacency_matrix(new_graph).todense()

            # Check if the model graph is equal to the new graph matrix
            if np.array_equal(model_graph_matrix, new_graph_matrix):
                return "removed the wrong node"
        
        # Check if the model has the same dimensions as the original graph
        if model_graph_matrix.shape[0] == original_graph_matrix.shape[0] and model_graph_matrix.shape[1] == original_graph_matrix.shape[1]:
            return "adjusted entries without removing a row or column"
        
    if np.not_equal(model_graph_matrix, answer_graph_matrix).any():
        return "modified graph somehow"

    return "unknown"

def extract_final_answer(client, output_string, prompt_modifier, solution=None):
    if prompt_modifier == "mod":
        intermediate_prompt = "The below text is a language model's response to a particular question. What is the final matrix that the language model gives? Write the final matrix as a list of lists separated by newlines, such as [[0, 1, 0],\n [0, 1, 0],\n [1, 1, 0]]." + "\n" + output_string
        #intermediate_prompt = "The below text is a language model's response to a particular question. What is the final answer that the language model gives? Write the final answer as an adjacency matrix surrounded by square brackets in the form of a list of lists, such as [[0, 1], [1, 0]]." + "\n" + output_string
    elif prompt_modifier == "yes/no":
        intermediate_prompt = "The below text is a language model's response to a particular question. What is the final answer that the language model gives? Write either Yes or No" + "\n" + output_string
    elif prompt_modifier == "number":
        intermediate_prompt = "The below text is a language model's response to a particular question. What is the final answer that the language model gives? Write the final answer as a number" + "\n" + output_string
    elif prompt_modifier == "list":
        intermediate_prompt = "The below text is a language model's response to a particular question. What is the final list of neighbors that the language model gives? Write the final answer as a list of numbers surrounded by square brackets. If you think the list should be empty, write []." + "\n" + output_string
    elif prompt_modifier == "list_names":
        intermediate_prompt = "The below text is a language model's response to a particular question. What is the final list of neighbors that the language model gives? Write the final answer as a list of node names surrounded by square brackets. If you think the list should be empty, write []." + "\n" + output_string
    elif prompt_modifier == "list_direct":
        if solution == "None":
            intermediate_prompt = f"The below text is a language model's response to a particular question. Does the model say that there are no isolated nodes? If so, write Yes, otherwise write No." + "\n" + output_string
        else:
            intermediate_prompt = f"The below text is a language model's response to a particular question. Is the final list of neighbors that the language model gives equal to {solution}, irrespective of ordering? If so, write Yes, otherwise write No." + "\n" + output_string
    
    client = OpenAI()
    intermediate_response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": intermediate_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
        #final_answer_output_string = intermediate_response.choices[0].message.content.strip()
    final_answer_output_string = intermediate_response.choices[0].message.content.strip()

    #final_answer_output_string = intermediate_response.choices[0].message.content.strip()

    final_answer_output_string = final_answer_output_string.replace(intermediate_prompt, "")

    print(f"Final answer output string: {final_answer_output_string}")

    return final_answer_output_string

def extract_final_answer_encoding(client, output_string, solution):
    intermediate_prompt = "The below text is composed of two parts. The first is a graph outputted by a language model. The second below that is the correct graph. Are both graphs equal (disregarding the order in which nodes and edges are presented)? If so, and both graphs have exactly the same set of edges, write Yes, otherwise write No" + "\n" + "1. Language model graph:" + "\n" + output_string + "\n" + "2. Correct graph:" + "\n" + solution
    
    client = OpenAI()
    intermediate_response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": intermediate_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )

    final_answer_output_string = intermediate_response.choices[0].message.content.strip()

    final_answer_output_string = final_answer_output_string.replace(intermediate_prompt, "")

    print(f"Final answer output string: {final_answer_output_string}")

    return True if ("Yes" in final_answer_output_string or "yes" in final_answer_output_string) else False

def get_answer_directory_accuracy(answer_directory):
        count = 0
        i = 0
        for filename in os.listdir(answer_directory):
            if i == 250:
                break
            i += 1
            with open(os.path.join(answer_directory, filename), 'r') as f:
                file = f.read()

                if "Correct" in file:
                    count += 1
        total_files = len(os.listdir(answer_directory))
        if total_files > 250:
            total_files = 250
        percentage = (count / total_files) * 100
        return percentage

def get_error_analysis_accuracies(answer_directory):
    missing_row_count = 0
    missing_column_count = 0
    extra_row_count = 0
    extra_column_count = 0
    not_modification_count = 0
    added_self_loop_count = 0
    added_only_one_edge_count = 0
    added_to_adjacent_entry_count = 0   
    added_to_non_adjacent_entry_count = 0
    removed_only_one_edge_count = 0
    removed_from_adjacent_entry_count = 0
    removed_from_non_adjacent_entry_count = 0
    connected_the_added_node_count = 0
    removed_wrong_node_count = 0
    adjusted_entries_without_removing_count = 0
    modified_graph_somehow_count = 0
    dimension_error_count = 0

    total_errors = 0

    for filename in os.listdir(answer_directory):
            with open(os.path.join(answer_directory, filename), 'r') as f:
                file = f.read()

                if "Incorrect" in file:
                    total_errors += 1
    
                    if "missing row" in file:
                        missing_row_count += 1
                    elif "missing column" in file:
                        missing_column_count += 1
                    elif "extra row" in file:
                        extra_row_count += 1
                    elif "extra column" in file:
                        extra_column_count += 1
                    elif "no modification" in file:
                        not_modification_count += 1
                    elif "added self loop" in file:
                        added_self_loop_count += 1
                    elif "added only one edge" in file:
                        added_only_one_edge_count += 1
                    elif "added to adjacent entry" in file:
                        added_to_adjacent_entry_count += 1
                    elif "added to non-adjacent entry" in file:
                        added_to_non_adjacent_entry_count += 1
                    elif "removed only one edge" in file:
                        removed_only_one_edge_count += 1
                    elif "removed from adjacent entry" in file:
                        removed_from_adjacent_entry_count += 1
                    elif "removed from non-adjacent entry" in file:
                        removed_from_non_adjacent_entry_count += 1
                    elif "connected the added node" in file:
                        connected_the_added_node_count += 1
                    elif "removed the wrong node" in file:
                        removed_wrong_node_count += 1
                    elif "adjusted entries without removing a row or column" in file:
                        adjusted_entries_without_removing_count += 1
                    elif "modified graph somehow" in file:
                        modified_graph_somehow_count += 1
                    elif "dimension error" in file:
                        dimension_error_count += 1
                    else:
                        print("Error type not found in file")
                        print(file)
                        sys.exit(1)

    #total_files = len(os.listdir(answer_directory))

    if total_errors == 0:
        error_analysis_accuracies = {
            "missing row": [0, 0],
            "missing column": [0, 0],
            "extra row": [0, 0],
            "extra column": [0, 0],
            "no modification": [0, 0],
            "added self loop": [0, 0],
            "added only one edge": [0, 0],
            "added to adjacent entry": [0, 0],
            "added to non-adjacent entry": [0, 0],
            "removed only one edge": [0, 0],
            "removed from adjacent entry": [0, 0],
            "removed from non-adjacent entry": [0, 0],
            "connected the added node": [0, 0],
            "removed the wrong node": [0, 0],
            "adjusted entries without removing a row or column": [0, 0],
            "modified graph somehow": [0, 0],
            "dimension error": [0, 0]
        }
    else:
        error_analysis_accuracies = {
            "missing row": [missing_row_count, (missing_row_count / total_errors) * 100],
            "missing column": [missing_column_count, (missing_column_count / total_errors) * 100],
            "extra row": [extra_row_count, (extra_row_count / total_errors) * 100],
            "extra column": [extra_column_count, (extra_column_count / total_errors) * 100],
            "no modification": [not_modification_count, (not_modification_count / total_errors) * 100],
            "added self loop": [added_self_loop_count, (added_self_loop_count / total_errors) * 100],
            "added only one edge": [added_only_one_edge_count, (added_only_one_edge_count / total_errors) * 100],
            "added to adjacent entry": [added_to_adjacent_entry_count, (added_to_adjacent_entry_count / total_errors) * 100],
            "added to non-adjacent entry": [added_to_non_adjacent_entry_count, (added_to_non_adjacent_entry_count / total_errors) * 100],
            "removed only one edge": [removed_only_one_edge_count, (removed_only_one_edge_count / total_errors) * 100],
            "removed from adjacent entry": [removed_from_adjacent_entry_count, (removed_from_adjacent_entry_count / total_errors) * 100],
            "removed from non-adjacent entry": [removed_from_non_adjacent_entry_count, (removed_from_non_adjacent_entry_count / total_errors) * 100],
            "connected the added node": [connected_the_added_node_count, (connected_the_added_node_count / total_errors) * 100],
            "removed the wrong node": [removed_wrong_node_count, (removed_wrong_node_count / total_errors) * 100],
            "adjusted entries without removing a row or column": [adjusted_entries_without_removing_count, (adjusted_entries_without_removing_count / total_errors) * 100],
            "modified graph somehow": [modified_graph_somehow_count, (modified_graph_somehow_count / total_errors) * 100],
            "dimension error": [dimension_error_count, (dimension_error_count / total_errors) * 100]
        }

    error_sum = missing_row_count + missing_column_count + extra_row_count + extra_column_count + not_modification_count + added_self_loop_count + added_only_one_edge_count + added_to_adjacent_entry_count + added_to_non_adjacent_entry_count + removed_only_one_edge_count + removed_from_adjacent_entry_count + removed_from_non_adjacent_entry_count + connected_the_added_node_count + removed_wrong_node_count + adjusted_entries_without_removing_count + modified_graph_somehow_count + dimension_error_count

    print(f"Sum of all error types: {error_sum}")
    print(f"Total errors: {total_errors}")

    if error_sum != total_errors:
        print("Error sum does not match total errors")
        sys.exit(1)
    return error_analysis_accuracies

def plot_n_p_info(answer_directory, results_directory):
    correct_n = []
    correct_p = []
    incorrect_n = []
    incorrect_p = []

    i = 0

    for filename in os.listdir(answer_directory):
        if i == 100:
            break
        i += 1
        with open(os.path.join(answer_directory, filename), 'r') as f:
            file = f.read()

            print(f"filename: {filename}")
            print(f"file: {file}")

            if "Correct" in file:
                match = re.search(r'n:\s*(\d+)', file)
                n = int(match.group(1))
                correct_n.append(n)

                match = re.search(r'p:\s*(.*)', file)
                # print the string that matched the pattern
                print(f"Matched string: {match.group(1)}")
                p = float(match.group(1))
                correct_p.append(p)
            else:
                match = re.search(r'n:\s*(\d+)', file)
                n = int(match.group(1))
                incorrect_n.append(n)

                match = re.search(r'p:\s*(.*)', file)
                # print the string that matched the pattern
                print(f"Matched string: {match.group(1)}")
                p = float(match.group(1))
                incorrect_p.append(p)

            print(f"Correct n: {correct_n}")
            print(f"Correct p: {correct_p}")
            print(f"Incorrect n: {incorrect_n}")
            print(f"Incorrect p: {incorrect_p}")

    # Plot the correct points as blue circles
    plt.scatter(correct_n, correct_p, color='blue', marker='o', label='Correct')

    # Plot the incorrect points as red x's
    plt.scatter(incorrect_n, incorrect_p, color='red', marker='x', label='Incorrect')

    # Set the labels for the x-axis and y-axis
    plt.xlabel('n')
    plt.ylabel('p')

    # Add a title to the plot
    plt.title('n vs. p')

    # Set x-axis and y-axis limits
    plt.xlim(5, 20)
    plt.ylim(0, 1)

    # Add a legend to the plot
    plt.legend()

    # Save this plot in the results directory
    plt.savefig(os.path.join(results_directory, 'n_p_plot.png'))

# Load the checkpoint (i.e., the last processed example index) from a file
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0  # If no checkpoint exists, start from the first example

# Save the checkpoint (i.e., the current index) to a file
def save_checkpoint(checkpoint_file, current_index):
    with open(checkpoint_file, 'w') as f:
        f.write(str(current_index))