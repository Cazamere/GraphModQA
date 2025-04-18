import os
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def adjacency_matrix_from_string(adj_matrix_str):
        adj_matrix_str = adj_matrix_str.replace(" ", "")
        adj_matrix_str = adj_matrix_str.replace(".", "")
        ##print("Adjacency matrix string input: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace(" ", "")
        if "],\n" not in adj_matrix_str:
            adj_matrix_str = adj_matrix_str.replace("], ", "],\n")
        adj_matrix_str = adj_matrix_str.replace("[[", "[")
        adj_matrix_str = adj_matrix_str.replace("]]", "]")
        ##print("Adjacency matrix string after removing double brackets: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace(",", "")
        ##print("Adjacency matrix string after replacing commas: ", adj_matrix_str)
        adj_matrix_str = adj_matrix_str.replace("  ", " ")
        adj_matrix_str = adj_matrix_str.replace(" [", "[")
        adj_matrix_str = adj_matrix_str.replace("```", "")
        adj_matrix_str = adj_matrix_str.replace("*", "")
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        #adj_matrix_str = re.sub(r'[^01]', '', adj_matrix_str)
        adj_matrix_str = re.sub(r'#.*', '', adj_matrix_str)
        ##print("Adjacency matrix string after removing double spaces and ```: ", adj_matrix_str)
        adj_matrix_list = adj_matrix_str.split('\n')
        if adj_matrix_list[0] == '':
            adj_matrix_list = adj_matrix_list[1:]
        if adj_matrix_list[-1] == '':
            adj_matrix_list = adj_matrix_list[:-1]
        ##print("Adjacency matrix list after newline split: ", adj_matrix_list)
        adj_matrix = [[int(num) for num in row.replace(" [", "[").replace("] ", "]").strip("[]").replace(" ", "")] for row in adj_matrix_list]
        ##print("Adjacency matrix string before being converted into a np array: ", adj_matrix_str)
        adj_matrix = np.array(adj_matrix)
        return adj_matrix

def binary_string_to_adjacency_matrix(binary_string):
    try:
        binary_string = binary_string.replace(" ", "")
        binary_string = binary_string.replace(",", "")
        binary_string = binary_string.replace("\n", "")
        binary_string = binary_string.replace("\t", "")
        binary_string = binary_string.replace("\r", "")
        binary_string = binary_string.replace("[[", "[")
        binary_string = binary_string.replace("]]", "]")
        binary_string = binary_string.replace("[", "")
        binary_string = binary_string.replace("]", "")
        binary_string = binary_string.replace(" ", "")
        binary_string = binary_string.replace("0", "0 ")
        binary_string = binary_string.replace("1", "1 ")
        binary_string = binary_string.strip()
        binary_list = binary_string.split()
        matrix_size = int(len(binary_list) ** 0.5)
        adjacency_matrix = np.array(binary_list).reshape(matrix_size, matrix_size).astype(int)
        return adjacency_matrix
    except Exception as e:
        #print("Error converting binary string to adjacency matrix: ", e)
        return None


def get_error_analysis_accuracies(model, prompt_type, modification):
    big_dict = {}
    for chain_length in range(1, 6):
        answer_directory = f"answers/encoding_chain_no_print/{model}/{prompt_type}/{modification}/{chain_length}/adjacency_matrix_big"
        
        total_errors = 0
        add_edge_error_dict = {
            "Changed dimensionality": 0,
            "No modification made": 0,
            "Added extra edges": 0,
            "Altered non-adjacent index": 0,
            "Directed modification": 0,
            "Dimension error": 0,
            "Alterered adjacent index": 0,
            "Altered correct index and adjacent index": 0,
        }

        remove_edge_error_dict = {
            "Changed dimensionality": 0,
            "No modification made": 0,
            "Removed extra edges": 0,
            "Altered non-adjacent index": 0,
            "Directed modification": 0,
            "Dimension error": 0,
            "Alterered adjacent index": 0,
            "Altered correct index and adjacent index": 0,
        }

        add_node_error_dict = {
            "Did not expand dimensionality": 0,
            "No modification made": 0,
            "Dimension error": 0,
            "Added two nodes": 0,
            "Miscopy error": 0,
            "Connected added node": 0,
            "Added too few nodes": 0,
            "Added too many nodes": 0,
        }

        remove_node_error_dict = {
            "Did not expand dimensionality": 0,
            "No modification made": 0,
            "Dimension error": 0,
            "Extra dimension added": 0,
            "Removed two nodes": 0,
            "Removed more than two nodes": 0,
            "Removed wrong row and column": 0,
            "Incorrect implementation": 0,
            "Removed too few nodes": 0,
            "Removed too many nodes": 0,
            "Wrong node removed": 0,
        }

        if modification == "add_edge":
            error_dict = add_edge_error_dict
        elif modification == "remove_edge":
            error_dict = remove_edge_error_dict
        elif modification == "add_node":
            error_dict = add_node_error_dict
        elif modification == "remove_node":
            error_dict = remove_node_error_dict
        else:
            #print("Invalid modification")
            return

        # assign each error type above to a unique color in a dictionary for a plot
        color_dict = {
            "Changed dimensionality": "blue",
            "No modification made": "orange",
            "Added extra edges": "green",
            "Altered non-adjacent index": "red",
            "Directed modification": "purple",
            "Dimension error": "brown",
            "Alterered adjacent index": "pink",
            "Altered correct index and adjacent index": "gray",
            "Did not expand dimensionality": "cyan",
            "Added two nodes": "darkgoldenrod",
            "Miscopy error": "teal",
            "Connected added node": "magenta",
            "Added too few nodes": "lime",
            "Added too many nodes": "indigo",
            "Extra dimension added": "olive",
            "Removed two nodes": "skyblue",
            "Removed more than two nodes": "salmon",
            "Removed wrong row and column": "gold",
            "Removed too few nodes": "violet",
            "Removed too many nodes": "turquoise",
            "Wrong node removed": "coral",
            "Removed extra edges": "black",
            "Incorrect implementation": "darkgoldenrod",
        }

        l = 0
        for filename in os.listdir(answer_directory):
            l += 1
            if l > 250:
                break
            with open(os.path.join(answer_directory, filename), 'r') as f:
                file = f.read()

                if "Incorrect" in file:
                    total_errors += 1
                    ##print("--------------------"*3)
                    ##print(filename)
                    ##print(file)
                    ###print("--------------------")
                    # Extract adjacency matrix using regular expressions
                    adjacency_matrix_str = re.findall(r'\[\[.*?\]\]', file, re.DOTALL)
                    
                    if adjacency_matrix_str:
                        original_adjacency_matrix = adjacency_matrix_from_string(adjacency_matrix_str[0])
                        solution_adjacency_matrix = adjacency_matrix_from_string(adjacency_matrix_str[1])
                        ##print(original_adjacency_matrix)
                        #print(solution_adjacency_matrix)
                        #return
                    else:
                        print("No adjacency matrix found")
                        #return
                    #print()
                    # Extract binary string using regular expressions
                    binary_string = re.findall(r'Model output: .*?\n', file, re.DOTALL)
                    if binary_string:
                        binary_string = binary_string[0].replace("Model output: ", "")
                        model_adjacency_matrix = binary_string_to_adjacency_matrix(binary_string)
                        #print(model_adjacency_matrix)
                        if model_adjacency_matrix is None:
                            error_dict["Dimension error"] += 1
                            continue
                        #return
                    else:
                        print("No binary string found")
                        #return
                    #print()

                    # Check if the original and model matrices are the same
                    if np.array_equal(original_adjacency_matrix, model_adjacency_matrix):
                        #print("No modification made")
                        error_dict["No modification made"] += 1
                        continue

                    if modification == "add_edge" or modification == "remove_edge":
                        # Check if the original and model matrices are the same size
                        if original_adjacency_matrix.shape != model_adjacency_matrix.shape:
                            #print("Matrix sizes do not match")
                            #print(original_adjacency_matrix.shape)
                            #print(model_adjacency_matrix.shape)
                            error_dict["Changed dimensionality"] += 1
                            continue
                        #else:
                        #    print("Matrix sizes match")

                        # #print the matrix of differences between the two adjacency matrices
                        #print("Difference matrix:")
                        model_solution_difference = model_adjacency_matrix - solution_adjacency_matrix
                        #print(f"Model - Solution:\n{model_solution_difference}")
                        model_original_difference = model_adjacency_matrix - original_adjacency_matrix
                        #print(f"Model - Original:\n{model_original_difference}")
                        solution_original_difference = solution_adjacency_matrix - original_adjacency_matrix
                        #print(f"Solution - Original:\n{solution_original_difference}")

                        # Find the indices where solution_original_difference is not zero
                        #print("Indices where solution_original_difference is not zero:")
                        diff_indices = np.where(solution_original_difference != 0)
                        #print(diff_indices)

                        # For each index in the diff_indices, create a list of adjacent indices (e.g. if the index is (1, 2), the 8 adjacent indices are [(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3)])
                        #print("Adjacent indices:")
                        found = False
                        adj = []
                        for index in zip(*diff_indices):
                            #print(index)
                            adjacent_indices = [(i, j) for i in range(index[0] - 1, index[0] + 2) for j in range(index[1] - 1, index[1] + 2) if 0 <= i < solution_original_difference.shape[0] and 0 <= j < solution_original_difference.shape[1] and (i, j) != index]
                            #print(adjacent_indices)
                            adj += adjacent_indices
                            for ad_index in adjacent_indices:
                                if model_original_difference[ad_index] in [1, -1]:
                                    #print("Alteration is adjacent")
                                    if model_original_difference[index] in [1, -1]:
                                        #print("Both the correct index and an adjacent index were altered")
                                        error_dict["Altered correct index and adjacent index"] += 1
                                        ##print(firsthaha)
                                        found = True
                                        break
                                    error_dict["Alterered adjacent index"] += 1
                                    ##print(secondhaha)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            continue

                        model_diff_indices = np.where(model_original_difference != 0)
                        for ind in zip(*model_diff_indices):
                            if ind not in adj:
                                #print("Altered non-adjacent index")
                                error_dict["Altered non-adjacent index"] += 1
                                found = True
                                break
                        if found:
                            continue

                        if np.count_nonzero(model_original_difference) % 2 == 1:
                            #print("Directed modification")
                            error_dict["Directed modification"] += 1
                            continue

                        print(original_adjacency_matrix)
                        print(solution_adjacency_matrix)
                        print(model_adjacency_matrix)
                        print(model_original_difference)
                        print(solution_original_difference)
                        print(model_solution_difference)
                        print(fsidubf)

                        # If the number of 1's in the difference matrix is greater than 2, #print("added extra edges")
                        if np.count_nonzero(model_original_difference) > 2*chain_length:
                            #print("Added extra edges")
                            if modification == "add_edge":
                                error_dict["Added extra edges"] += 1
                            elif modification == "remove_edge":
                                error_dict["Removed extra edges"] += 1
                        elif np.count_nonzero(model_original_difference) % 2 == 1:
                            #print("Directed modification")
                            error_dict["Directed modification"] += 1
                        else:
                            #print("Wrong index")
                            error_dict["Altered non-adjacent index"] += 1


                    if modification == "add_node" or modification == "remove_node":
                        # Check if the matrices are the same size
                        if original_adjacency_matrix.shape == model_adjacency_matrix.shape:
                            #print("Matrix sizes match")
                            #print(original_adjacency_matrix.shape)
                            #print(model_adjacency_matrix.shape)
                            error_dict["Did not expand dimensionality"] += 1 # did not expa
                            continue

                        # miscopy error
                        
                    if modification == "add_node":
                        # Added two nodes
                        #if model_adjacency_matrix.shape[0] == original_adjacency_matrix.shape[0] + 2 and model_adjacency_matrix.shape[1] == original_adjacency_matrix.shape[1] + 2:
                        #    #print("Added two nodes")
                        #    error_dict["Added two nodes"] += 1
                        #    continue

                        # Added too few nodes: Check if solutions shape is larger than the model shape
                        if solution_adjacency_matrix.shape[0] > model_adjacency_matrix.shape[0] and solution_adjacency_matrix.shape[1] > model_adjacency_matrix.shape[1]:
                            #print("Added too few nodes")
                            error_dict["Added too few nodes"] += 1
                            continue

                        # Added too many nodesL Check if solutions shape is smaller than the model shape
                        if solution_adjacency_matrix.shape[0] < model_adjacency_matrix.shape[0] and solution_adjacency_matrix.shape[1] < model_adjacency_matrix.shape[1]:
                            #print("Added too many nodes")
                            error_dict["Added too many nodes"] += 1
                            continue

                        # check if the final row or columns of model_adjacency_matrix have any non-zero values
                        if np.count_nonzero(model_adjacency_matrix[-1]) > 0 or np.count_nonzero(model_adjacency_matrix[:, -1]) > 0:
                            #print("Connected added node")
                            error_dict["Connected added node"] += 1
                            continue



                        # miscopy error: if model - solution has any non-zero values, then it is a miscopy error
                        miscopy_error = model_adjacency_matrix - solution_adjacency_matrix
                        if np.count_nonzero(miscopy_error) > 0:
                            #print("Miscopy error")
                            #print(miscopy_error)
                            error_dict["Miscopy error"] += 1
                            continue               

                    if modification == "remove_node":
                        # Removed two nodes error
                        #if model_adjacency_matrix.shape[0] == original_adjacency_matrix.shape[0] - 2 and model_adjacency_matrix.shape[1] == original_adjacency_matrix.shape[1] - 2:
                        #    #print("Removed two nodes")
                        #    error_dict["Removed two nodes"] += 1
                        #    continue

                        # Removed more than two nodes error
                        #if model_adjacency_matrix.shape[0] < original_adjacency_matrix.shape[0] - 2 or model_adjacency_matrix.shape[1] < original_adjacency_matrix.shape[1] - 2:
                        #    #print("Removed more than two nodes")
                        #    error_dict["Removed more than two nodes"] += 1
                        #    continue

                        # Removed too few nodes: Check if solutions shape is smaller than the model shape
                        if solution_adjacency_matrix.shape[0] < model_adjacency_matrix.shape[0] and solution_adjacency_matrix.shape[1] < model_adjacency_matrix.shape[1]:
                            #print("Removed too few nodes")
                            error_dict["Removed too few nodes"] += 1
                            continue

                        # Removed too many nodes: Check if solutions shape is larger than the model shape
                        if solution_adjacency_matrix.shape[0] > model_adjacency_matrix.shape[0] and solution_adjacency_matrix.shape[1] > model_adjacency_matrix.shape[1]:
                            #print("Removed too many nodes")
                            error_dict["Removed too many nodes"] += 1
                            continue
                        """
                        found = False
                        for i in range(len(original_adjacency_matrix)):
                            for j in range(len(original_adjacency_matrix)):
                                modified_matrix = np.delete(original_adjacency_matrix, i, axis=0)
                                modified_matrix = np.delete(modified_matrix, j, axis=1)
                                if np.array_equal(modified_matrix, model_adjacency_matrix):
                                    #print("Removed wrong row and column")
                                    error_dict["Removed wrong row and column"] += 1
                                    found = True
                                    break
                            if found:
                                break

                        # Check if the model removed a row and column that it wasn't supposed to
                        removed_rows = []
                        # Iterate through each row in original_adjacency_matrix
                        for i in range(len(original_adjacency_matrix)):
                            # If the row is not in the model_adjacency_matrix, then the row was removed
                            row = original_adjacency_matrix[i]
                            if row not in solution_adjacency_matrix:
                                removed_rows.append(row)
                                #break # this will always break

                        wrong_row_removed = False
                        for row in removed_rows:
                            # If the row is not in the model_adjacency_matrix, then the row was removed
                            if row in model_adjacency_matrix:
                                #print("Wrong node removed")
                                error_dict["Wrong node removed"] += 1
                                wrong_row_removed = True
                                break
                        
                        if wrong_row_removed:
                            continue
                            """
                        
                        miscopy_error = model_adjacency_matrix - solution_adjacency_matrix
                        if np.count_nonzero(miscopy_error) > 0:
                            #print("Miscopy error")
                            #print(miscopy_error)
                            error_dict["Incorrect implementation"] += 1
                            continue


                    #return
                    #print("--------------------"*3)

                #if total_errors > 7:
                #    break
        #print("Total errors: ", total_errors)
        #print("Error dictionary: ", error_dict)
        #print(sum(error_dict.values()))
        if total_errors != sum(error_dict.values()):
            print("Error in error_dict")
            print(fisbf)
            return
        # convert each value in the error_dict to a percentage of the total errors
        #for key in error_dict:
        #    if total_errors == 0:
        #        error_dict[key] = 0
        #    else:
        #        error_dict[key] = round(error_dict[key] / total_errors * 100, 2)
        #print("Error dictionary as percentages: ", error_dict)
        big_dict[chain_length] = error_dict
    #print("Big dictionary: ", big_dict)

    new_dict = {}
    for error in big_dict[1]:
        new_dict[error] = []
        for chain_length in big_dict:
            new_dict[error].append(big_dict[chain_length][error])

    plt.figure(figsize=(10.5, 8.5))  # Set the figure size to 10 inches by 6 inches

    # Plot each value in new_dict as a line
    for error in new_dict:
        # if new_dict[error] is all zeros, then skip
        if sum(new_dict[error]) == 0:
            continue
        line_color = color_dict[error]
        plt.plot([1, 2, 3, 4, 5], new_dict[error], label=error, color=line_color, linewidth=3)

    # Add labels and legend
    plt.xlabel('Number of modifications', fontsize=30)
    plt.ylabel('Number of Errors Made', fontsize=30)
    plt.legend(fontsize=20)
    plt.xlim(1, 5)
    #plt.ylim(0, 100)
    plt.xticks(range(1, 6), fontsize=19)
    plt.yticks(fontsize=19)
    plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
    plt.tick_params(right=False, top=False)

    if model == "claude-3-5-sonnet-20240620_70B":
        model_name = "Claude 3.5 Sonnet"
    elif model == "o1-mini_70B":
        model_name = "o1-mini"
    elif model == "llama3.1_405B":
        model_name = "Llama 3.1 405B"
    elif model == "gpt-4o-mini_70B":
        model_name = "GPT-4o-mini"
    
    if modification == "add_edge":
        modification_name = "Add Edge"
    elif modification == "remove_edge":
        modification_name = "Remove Edge"
    elif modification == "add_node":
        modification_name = "Add Node"
    elif modification == "remove_node":
        modification_name = "Remove Node"
    plt.title(f'Errors, {model_name}, {modification_name}', fontsize=30)


    # Show the plot
    plt.show()

    # Save the plot
    # Create directories if they don't exist
    os.makedirs(f'plots/errors/{model}/{modification}', exist_ok=True)

    plt.savefig(f'plots/errors/{model}/{modification}/plot.png')

    # clear the plot
    plt.clf()

    return



if __name__ == "__main__":
    #model = "o1-mini_70B" # "claude-3-5-sonnet-20240620_70B", o1-mini_70B, llama3.1_405B, gpt-4o-mini_70B
    prompt_type = "print_graph"
    #modification = "add_node"
    #chain_length = 4
    #encoding = "adjacency_matrix"
    #answer_dir = f"answers/encoding_chain_no_print/{model}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
    for model in ["claude-3-5-sonnet-20240620_70B", "o1-mini_70B", "llama3.1_405B", "gpt-4o-mini_70B"]:
        for modification in ["add_edge", "remove_edge", "add_node", "remove_node"]:
            print(f"Model: {model}, Modification: {modification}")
            get_error_analysis_accuracies(model, prompt_type, modification)