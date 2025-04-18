import os

model_name = "palm2_70B" # choices are: gemma_9b, gemma_27b, llama3_70B, llama3.1_70B, palm2_70B, gpt-4-0125-preview_70B, gpt-4-0613_70B
task = "add_node" # choices are: add_edge, add_node, remove_edge, remove_node, connected_nodes
score = 0
num_files = 500
for i in range(num_files):
    if i in [245, 305, 350, 497, 498, 499]:
        continue
    filename = f'answers/base/{model_name}/{task}/{i}.txt'
    with open(filename, 'r') as file:
        contents = file.read()
        #print(f'Contents of file {filename}:')
        #print(contents)
        if task in ["add_edge", "add_node", "remove_edge", "remove_node"]:
            # Find the start and end indices of the adjacency matrix within the output string
            solution_start_index = contents.find("Solution:")
            solution_end_index = contents.find("]]", solution_start_index)
            solution = contents[solution_start_index:solution_end_index+1].replace("Solution: ", "")

            new_solution = solution.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "")
            #print()
            #print(f"Solution: {solution}")
            print(f"new_solution: {new_solution}")

            # Extract the adjacency matrix substring
            start_index = contents.find("output:")
            adjacency_matrix_output = contents[start_index:].replace("output: ", "").replace("\n", "").replace(" ", "").replace("Correct", "")

            print(f"Model Output: {adjacency_matrix_output}")

            if new_solution == adjacency_matrix_output:
                #print("Correct!")
                score += 1
            else:
                #print("Incorrect!")
                pass
        elif task == "connected_nodes":
            solution_start_index = contents.find("Solution:")
            solution_end_index = contents.find("]", solution_start_index)
            solution = contents[solution_start_index:solution_end_index+1].replace("Solution: ", "")

            new_solution = solution.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "")
            #print()
            # convert new_solution to a list of integers
            new_solution = new_solution.split(",")
            new_solution = sorted([int(i) for i in new_solution])
            print(f"Solution: {solution}")
            print(f"new_solution: {new_solution}")

            # Extract the adjacency matrix substring
            start_index = contents.find("output:")
            output = contents[start_index:].replace("output: ", "")
            print(f'raw output: {output}')
            print(f'output split on new line:{output.split("\n")}')
            output = output.split("\n")[0]
            print(f'output after taking first element post new line split: {output}')
            output = output.split(".")[0]
            print(f'output after taking first element post . split: {output}')
            output = output.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("Node", "").replace("and", ",")
            print(output)
            if len(output) > 20:
                #print("Output too long, old output: ", output)
                output = output[0]
                #print("New output: ", output)
            # convert output to a list of integers
            output = output.split(",")
            print(f'output after split: {output}')
            # if '' is in the list, remove it
            if '' in output:
                output.remove('')
            print(output)
            output = sorted([int(i.replace(".", "")) for i in output])

            print(f"Model Output: {output}")
            print()

            if new_solution == output:
                #print("Correct!")
                score += 1
            else:
                #print("Incorrect!")
                pass

            #break
        else:
            # find the index in contents.split() where "Solution:" is
            contents = contents.split()
            for i in range(len(contents)):
                if contents[i] == "Solution:":
                    solution_index = i

                    # solution = int(contents[solution_index+1]) # ints
                    solution = contents[solution_index+1] # strings

                    print(f"Solution extracted: {solution}")
                if contents[i] == "output:":
                    output_index = i

                    # output = int(contents[output_index+1]) # ints
                    output = contents[output_index+1] # strings

                    print(f"Model output extracted: {output}")

            if solution == output:
                print("Correct!")
                score += 1
            else:
                print("Incorrect!")



print(f'Score: {(score/num_files)*100}')
        