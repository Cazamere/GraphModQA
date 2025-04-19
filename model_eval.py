import os
import argparse
import torch
import sys
import numpy as np
from openai import OpenAI
import anthropic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, OPTForCausalLM, AutoModelForCausalLM
from huggingface_hub import login
from graph_generator_utils import graph_to_string_encoder
#from model_eval_utils import *
from model_eval_utils import *
import re
import networkx as nx
from fireworks.client import Fireworks
import json
import requests

import pprint
import google.generativeai as genai

def eval(input_dir, prompt_dir, model_name, tokenizer, device, model, prompt_type, solution_dir, augment_tasks, size, ablation=False, ablation_type = "p", p=0.5, n=5, chain_length=1, task="add_edge", cot_example_number=1, encoding_type="adjacency_matrix", answer_dir=None, results_dir=None, checkpoint_dir=None):
    # prompt_type has the form "chain_{chain_length}_node_count", change this to "chain_node_count"
    old_prompt_type = prompt_type
    prompt_type = prompt_type.replace("_1_", "_").replace("_2_", "_").replace("_3_", "_").replace("_4_", "_").replace("_5_", "_")
    print(f"Core Prompt type (removed the number in the middle): {prompt_type}")
    client = OpenAI()

    if ablation:
        if ablation_type == "p":
            print(f"p: {p}, n: {n}")
        elif ablation_type == "n" or ablation_type == "d":
            print(f"n: {n}")
        else:
            pass # TODO: clean up

    if ablation and ablation_type == "p":
        # convert p to string, and remove the decimal point
        p = str(p).replace(".", "")

    # Iterate over files
    score = 0

    # if the string "chain" is in prompt_type, then we are dealing with a chain task
    if "chain" in prompt_type:
        total_files = 100
    else:
            # Calculate the number of files in input_dir
        total_files = len([filename for filename in os.listdir(input_dir) if filename.endswith(".txt")])

    if model_name == 'palm2':
        total_files = 50
    elif ablation_type == "no_force":
        total_files = 5
    else:
        pass

    number_of_files_in_dataset = len([filename for filename in os.listdir(input_dir) if filename.endswith(".graphml")])

    print(f"Number of files in dataset: {number_of_files_in_dataset}")

    checkpoint_filename = "checkpoint.txt"

    last_processed_index = load_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename)) # REMEMBER: to manually reset this, delete the checkpoint file

    print(f"Last processed index: {last_processed_index}")

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

    number_of_files_in_dataset = 100

    for i in range(last_processed_index, number_of_files_in_dataset):
        print('-----------------------------------')
        print(f"Evaluating on graph {i+1}")

        input_filename = f"{i}.graphml"
        graph_info_filename = f"{i}.txt"
        prompt_filename = f"prompt_{i}.txt"
        solution_filename = f"solution_{i}.graphml" if prompt_type in augment_tasks else f"solution_{i}.txt"
        
        if ablation_type == "preserve_path":
            solution_filename_1 = f"solution_{i}_0.graphml"
            solution_filename_2 = f"solution_{i}_1.graphml"

        # Read input graph
        with open(os.path.join(input_dir, input_filename), "r") as input_file:
            #adjacency_matrix = input_file.read()
            adjacency_matrix_raw = nx.read_graphml(input_file)
            adjacency_matrix = graph_to_string_encoder(adjacency_matrix_raw)
        """"
        with open(os.path.join(input_dir, graph_info_filename), "r") as graph_file:
            graph_info = graph_file.read()
            numbers_list = graph_info.split(", ")
            graph_n = int(numbers_list[0].split(": ")[1])
            graph_p = float(numbers_list[1].split(": ")[1])
        """
        if not ablation:
            n = adjacency_matrix.count("[") - 1

        # Read prompt
        with open(os.path.join(prompt_dir, prompt_filename), "r") as prompt_file:
            prompt = prompt_file.read()

        if model_name == 'gpt-4o-mini' or model_name == 'gpt-4o' or model_name == 'gpt-4-0125-preview' or model_name == 'gpt-3.5-turbo-0125' or model_name == 'gpt-3.5-turbo-1106' or model_name == 'gpt-4-0613' or model_name == 'o1-mini' or model_name == 'o3-mini':
            # Generate output from the OpenAI model

            #client = OpenAI()

            if model_name == 'gpt-4o-mini':
                max_tokens = 4096*4
            else:
                max_tokens = 4096

            if model_name == 'o1-mini':
                response = client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ],
                #temperature=0.0,
                #max_completion_tokens=4096*4,
            )
            elif model_name == 'o3-mini':
                response = client.chat.completions.create(
                model="o3-mini-2025-01-31",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ],
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096*4,
                )
            output_string = response.choices[0].message.content.strip()
            print(f"Model output: {response.choices[0].message.content}")
            #print(f'Output length: {len(output_string)}')
            #continue
        elif model_name == 'palm2':
            completion = genai.generate_text(
                model='models/text-bison-001',
                prompt=prompt,
                temperature=0,
                # The maximum length of the response
                max_output_tokens=4096,
            )

            output_string = completion.result
            #print(f"Model output: {output_string}")
            #continue
        elif model_name == 'gemini-1.5-flash' or model_name == 'gemini-1.5-pro':

            model = genai.GenerativeModel("gemini-1.5-pro-002")
            response = model.generate_content(
                prompt,
                generation_config = genai.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=0.0,
                )
            )
            output_string = response.text

            
        elif model_name == 'claude-3-haiku-20240307' or model_name == 'claude-3-5-sonnet-20240620':
            claude_client = anthropic.Anthropic()
            mt = 4096 if model_name == 'claude-3-haiku-20240307' else 8192

            message = claude_client.messages.create(
                model=model_name,
                max_tokens=mt,
                temperature=0,
                system="You are a helpful assistant.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            output_string = message.content[0].text
            print(f"Model output: {output_string}")

            
        elif model_name in ['llama3.1', 'deepseek']:

            if model_name == 'llama3.1':
                router_model_name = "meta-llama/llama-3.1-405b-instruct"
            elif model_name == 'deepseek':
                router_model_name = "deepseek/deepseek-r1"

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"",
                },
                data=json.dumps({
                    "model": router_model_name, # Optional
                    "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                    ]
                    
                })
            )

            # Parse the JSON response
            data = response.json()

            #print(data)
            #sys.exit(0)

            # Access and print the completion text
            output_string = data['choices'][0]['message']['content'].strip()

        
        with open(os.path.join(solution_dir, solution_filename), "r") as solution_file:
            #solution = solution_file.read()
            if prompt_type in augment_tasks:
                solution_raw = nx.read_graphml(solution_file)
                solution = graph_to_string_encoder(solution_raw, graph_type=encoding_type)
            else:
                solution = solution_file.read()
        
        print()
        print('Input Graph:')
        print(f'{adjacency_matrix}')
        print()
        print('Task:')
        print(f'{prompt}')
        print()
        print('Solution:')
        print(f'{solution}')

        output_string = output_string.replace(prompt, "")
        # Remove the input prompt from the output
        print("Model output without input prompt:")
        print(output_string)

        correct = False
        new_solution = ""
        error = ""

        if prompt_type in augment_tasks:

            if encoding_type == "adjacency" or encoding_type == "adjacency_matrix":
                new_solution = solution.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace(".", "")
                claude_client = anthropic.Anthropic()
                final_answer_output_string = extract_final_answer(client=claude_client, output_string=output_string, prompt_modifier="mod")

                # Find the first adjacency matrix in the output string
                adjacency_matrix_output = re.search(r'\[\[.*?\]\]', final_answer_output_string, re.DOTALL)

                if adjacency_matrix_output:
                    adjacency_matrix_output = adjacency_matrix_output.group()
                    adjacency_matrix_numbers = adjacency_matrix_output.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace(",", "").replace(".", "")
                    final_answer = re.sub(r'[^01]', '', adjacency_matrix_numbers)

                    print(f'Original adjacency matrix found inside model output:')
                    print(final_answer)
                    print(new_solution)

                    # Compare the adjacency matrix output with the solution
                    if ablation_type == "preserve_path":
                        new_solution_1 = solution_1.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace(".", "")
                        new_solution_2 = solution_2.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace(".", "")

                        if final_answer == new_solution_1 or final_answer == new_solution_2:
                            print('Correct!')
                            score += 1
                            correct = True
                        else:
                            print('Incorrect!')
                            error_1 = error_analysis(original_graph=adjacency_matrix_raw, model_graph=adjacency_matrix_output, answer_graph=solution_raw_1, task=prompt_type)
                            error_2 = error_analysis(original_graph=adjacency_matrix_raw, model_graph=adjacency_matrix_output, answer_graph=solution_raw_2, task=prompt_type)

                            print(f"Error analysis resulted in the following error 1: {error_1}")
                            print(f"Error analysis resulted in the following error 2: {error_2}")

                            error = error_1 # TODO: clean up

                            print(f"Error analysis resulted in the following error: {error}")

                            if error == "unknown":
                                sys.exit(0)
                    else:
                        if final_answer == new_solution:
                            print('Correct!')
                            score += 1
                            correct = True
                        else:
                            print('Incorrect!')
                            error = error_analysis(original_graph=adjacency_matrix_raw, model_graph=adjacency_matrix_output, answer_graph=solution_raw, task=prompt_type)

                            print(f"Error analysis resulted in the following error: {error}")

                            if error == "unknown":
                                sys.exit(0)
                else:
                    print("No adjacency matrix found inside model output")
                    final_answer = "N/A"
                    error = "dimension error"

                
            else:
                claude_client = anthropic.Anthropic()
                correct = extract_final_answer_encoding(client=claude_client, output_string=output_string, solution=solution)

                if correct:
                    print('Correct!')
                    final_answer = "Correct"
                    score += 1
                    correct = True
                else:
                    print('Incorrect!')
                    final_answer = "Incorrect"
                    error = "N/A"
                #sys.exit(0)

        elif prompt_type in ["edge_exists", "cycle", "chain_edge_exists", "chain_cycle"]:
            found = False
            if model_name == "qwen2.5" and prompt_type == "edge_exists":
                # Check if 1 or "Yes" is in the output_string[:5]
                if "1" in output_string[:5] or "Yes" in output_string[:5]:
                    final_answer = "Yes"
                    found = True
                elif "0" in output_string[:5] or "No" in output_string[:5]:
                    final_answer = "No"
                    found = True

                if found:
                    if final_answer == solution:
                        print('Correct!')
                        score += 1
                        correct = True
                    else:
                        print('Incorrect!')
            if not found:
                claude_client = anthropic.Anthropic()
                final_answer_output_string = extract_final_answer(client=claude_client, output_string=output_string, prompt_modifier="yes/no")

                if "Yes" in final_answer_output_string:
                    final_answer = "Yes"
                elif "No" in final_answer_output_string:
                    final_answer = "No"
                else:
                    print("Error: No Yes or No found in model output")
                    sys.exit(1)

                # Compare the adjacency matrix output with the solution
                if final_answer == solution:
                    print('Correct!')
                    score += 1
                    correct = True
                else:
                    print('Incorrect!')

            # Find n and p for the adjacency_matrix_raw graph
            #n = adjacency_matrix_raw.number_of_nodes()


        elif prompt_type in ["isolated"]:
            final_answer = extract_final_answer(client=client, output_string=output_string, prompt_modifier="list_direct", solution=solution)
            if "Yes" in final_answer or "yes" in final_answer:
                print('Correct!')
                score += 1
                correct = True
            else:
                print('Incorrect!')
            #sys.exit(0)
        elif prompt_type in ["triangle"]:
            final_answer = extract_final_answer(client=client, output_string=output_string, prompt_modifier="number")
            # Find the first number in the output string
            number = re.search(r'\d+', final_answer)
            if number:
                final_answer = number.group()

                if int(final_answer) == int(solution):
                    print('Correct!')
                    score += 1
                    correct = True
                else:
                    print('Incorrect!')
                
        elif prompt_type in ["connected_nodes", "chain_connected_nodes"]:
            claude_client = anthropic.Anthropic()
            if encoding_type in ["adjacency_matrix", "adjacency_list", "incident"]:
                final_answer_output_string = extract_final_answer(client=client, output_string=output_string, prompt_modifier="list")
            else:
                final_answer_output_string = extract_final_answer(client=claude_client, output_string=output_string, prompt_modifier="list_names")

            found = False

            # Find the list in the output string
            final_answer = re.search(r'\[.*?\]', final_answer_output_string, re.DOTALL)

            if final_answer == None:
                print("Error: Can't find a list of numbers in output")
                #sys.exit(1)
                final_answer = "N/A"
            else:
                found = True

            if found:
                # Convert the list to a python list
                final_answer = final_answer.group().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("node", "")
                final_answer = final_answer.split(",")
                print(f"Final answer: {final_answer}")
                
                if encoding_type in ["adjacency", "adjacency_matrix", "adjacency_list", "incident"]:
                    
                    if final_answer == ['']:
                        final_answer = []
                    else:
                        try:
                            final_answer = [int(x) for x in final_answer]
                        except:
                            final_answer = []

                    print(f"Final answer: {final_answer}")

                    solution = solution.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
                    solution = solution.split(",")
                    print(f"Solution: {solution}")
                    if solution == ['']:
                        solution = []
                    else:
                        try:
                            solution = [int(x) for x in solution]
                        except:
                            solution = []

                    print(f"Solution: {solution}")  

                    # Compare the final answer with the solution
                    print(f"Sorted final_answer: {sorted(final_answer)}")
                    print(f"Sorted python_solution: {sorted(solution)}")
                    if sorted(final_answer) == sorted(solution):
                        print('Correct!')
                        score += 1
                        correct = True
                    else:
                        print('Incorrect!')
                else:
                    if final_answer == ['']:
                        final_answer = []
                    else:
                        try:
                            final_answer = [x for x in final_answer]
                        except:
                            final_answer = []

                    print(f"Final answer: {final_answer}")

                    solution = solution.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
                    solution = solution.split(",")
                    print(f"Solution: {solution}")
                    if solution == ['']:
                        solution = []
                    else:
                        try:
                            solution = [x for x in solution]
                        except:
                            solution = []

                    print(f"Solution: {solution}")  

                    # Compare the final answer with the solution
                    print(f"Sorted final_answer: {sorted(final_answer)}")
                    print(f"Sorted python_solution: {sorted(solution)}")
                    if sorted(final_answer) == sorted(solution):
                        print('Correct!')
                        score += 1
                        correct = True
                    else:
                        print('Incorrect!')
                    #sys.exit(1)

            
        else:
            print('Invalid prompt type')
            return
        
        # Compare output with solution
        #if output_string == solution:
        #    score += 1
        # Save final answer in a file

        correct_print = "Correct" if correct else "Incorrect"

        with open(f"{answer_dir}/{i}.txt", "w") as answer_file:
            #answer_file.write(f"Task: {prompt}\n Solution: {solution}\n Model output: {final_answer}\n{correct_print}\nError: {error}\nn: {graph_n}\np: {graph_p}\n")
            answer_file.write(f"Task: {prompt}\n Solution: {solution}\n Model output: {final_answer}\n{correct_print}\nError: {error}")

        # Save the current index as the checkpoint
        save_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename), i + 1)  # Save the next index to resume from

    # Calculate accuracy
    accuracy = get_answer_directory_accuracy(answer_directory=answer_dir)

    print(f"Accuracy: {accuracy}%")

    # Save the accuracy to a file
    with open(f"{results_dir}/results.txt", "w") as results_file:
        results_file.write(f"Accuracy: {accuracy}%")

    if prompt_type in augment_tasks and encoding_type == "adjacency_matrix":
        # Error analysis
        error_analysis_dict = get_error_analysis_accuracies(answer_directory=answer_dir)

        print(f"Error analysis: {error_analysis_dict}")

        # Append every key-value pair in the error analysis dictionary into a string
        error_analysis_string = ""
        for key, value in error_analysis_dict.items():
            error_analysis_string += f"{key}: {value[0]}, {value[1]}%\n"

        # Save the error analysis to a file
        with open(f"{results_dir}/error_analysis.txt", "w") as error_analysis_file:
            error_analysis_file.write(error_analysis_string)

        #plot_n_p_info(answer_directory=answer_dir, results_directory=results_dir)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Print memory of current GPU
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(device)
        print(f"GPU Memory: {gpu_properties.total_memory / 1024**3} GB")
    else:
        print("No GPU available")
        #return

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama2", "llama3", "opt", "openelm", "mistral", "mixtral", "phi", "gpt-4o-mini", "gpt-4o", "gpt-4-0125-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-0613", "o1-mini", "qwen2", "qwen2-math", "palm2", "gemma", "llama3.1", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "qwen2.5", "qwen2.5-math", "o3-mini"], required=True, help="Specify the transformer model")
    parser.add_argument("--size", type=str, required=True, help="Specify the size of the model (OPT: 125m, 350m, 1.3b, 6.7b, OpenELM: 270M, 450M, 1_1B, 3B, Llama2: 7b, 13b, 70b, Llama3: 8B, 70B, Mistral: 7B, Phi: 4k, 128k)")
    parser.add_argument("--prompt_types", type=str, default="add_edge", nargs='+', help="type of prompt")
    parser.add_argument("--max_length", type=int, default=200, help="maximum length of the output")
    parser.add_argument("--ablation", type=bool, required=True, help="whether to evaluate on ablation graphs")
    parser.add_argument("--ablationType", choices=["p", "n", "d", "few", "cot", "chain", "few_chain", "cot_chain", "barabasi_albert", "path", "star", "stochastic_block", "encoding", "no_force", "preserve_path", "preserve_star", "node_connect", "encoding_chain", "encoding_chain_no_print", "encoding_chain_few", "encoding_chain_cot", "encoding_chain_graph_type"], help="what type of ablation study")
    parser.add_argument("--exampleNumber", type=int, help="number of few/cot examples")
    parser.add_argument("--chainLength", type=int, help="number of few/cot examples")
    parser.add_argument("--graphType", choices=["star", "path", "empty", "complete"], help="number of few/cot examples")
    parser.add_argument("--modification", choices=["add_edge", "remove_edge", "add_node", "remove_node", "mix"], help="number of few/cot examples")
    parser.add_argument("--encoding", choices=["adjacency_matrix", "incident", "coauthorship", "friendship", "social_network", "expert", "politician", "got", "sp"], help="number of few/cot examples")
    parser.add_argument("--p", type=float, help="number of few/cot examples")
    parser.add_argument("--n", type=int, help="number of few/cot examples")
    args = parser.parse_args()

    # print all arguments
    args.ablation = True
    #print(args)
    #return
    access_token = os.getenv("HF_TOKEN")

    # Set up the transformer model
    if args.model == "opt":
        model_name = f"facebook/opt-{args.size}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = OPTForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    elif args.model == "llama2":
        model_name = f"meta-llama/Llama-2-{args.size}-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True,  use_auth_token=access_token)
    elif args.model == "llama3":
        model_name = f"meta-llama/Meta-Llama-3-{args.size}-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
    elif args.model == "llama3.1":
        #if args.size == "405B":
        model_name = "llama3.1"
        model = None
        tokenizer = None
        #else:
        #    model_name = f"meta-llama/Meta-Llama-3.1-{args.size}-Instruct"
        #    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        #    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
    elif args.model == "openelm":
        model_name = f"apple/OpenELM-{args.size}-Instruct"
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    elif args.model == "mistral":
        model_name = f"mistralai/Mistral-{args.size}-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, use_auth_token=access_token)
    elif args.model == "mixtral":
        model_name = f"mistralai/Mixtral-8x{args.size}-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, use_auth_token=access_token)    
    elif args.model == "phi":
        model_name = f"microsoft/Phi-3-mini-{args.size}-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True)
    elif args.model == "qwen2":
        model_name = f"Qwen/Qwen2-{args.size}-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
    elif args.model == "qwen2.5":
        #if args.prompt_types[0] == "print_graph":
        #    model_name = f"Qwen/Qwen2.5-{args.size}-Instruct"
        #    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        #    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
        #else:
        model_name = "qwen2.5"
        model = None
        tokenizer = None
    elif args.model == "qwen2-math":
        model_name = f"Qwen/Qwen2-Math-{args.size}-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
    elif args.model == "qwen2.5-math":
        model_name = f"Qwen/Qwen2.5-Math-{args.size}-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=access_token)
    elif args.model == "gemma":
        model_name = f"google/gemma-2-{args.size}-it"
        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True,  use_auth_token=access_token)

        tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{args.size}-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-27b-it",
            device_map="auto",
            torch_dtype="auto"
            #torch_dtype=torch.bfloat16
        )
    elif args.model == "gpt-3.5-turbo-0125":
        model_name = "gpt-3.5-turbo-0125"
        model = None
        tokenizer = None
    elif args.model == "gpt-3.5-turbo-1106":
        model_name = "gpt-3.5-turbo-1106"
        model = None
        tokenizer = None
    elif args.model == "gpt-4o-mini":
        model_name = "gpt-4o-mini"
        model = None
        tokenizer = None
    elif args.model == "gpt-4o":
        model_name = "gpt-4o"
        model = None
        tokenizer = None
    elif args.model == "gpt-4-0125-preview":
        model_name = "gpt-4-0125-preview"
        model = None
        tokenizer = None
    elif args.model == "gpt-4-0613":
        model_name = "gpt-4-0613"
        model = None
        tokenizer = None
    elif args.model == "palm2":
        models = [m for m in genai.list_models() if 'generateText' in m.supported_generation_methods]
        model_name = models[0].name
        model = None
        tokenizer = None
    elif args.model == "gemini-1.5-flash":
        model_name = "gemini-1.5-flash"
        model = None
        tokenizer = None
    elif args.model == "gemini-1.5-pro":
        model_name = "gemini-1.5-pro"
        model = None
        tokenizer = None
    elif args.model == "claude-3-haiku-20240307":
        model_name = "claude-3-haiku-20240307"
        model = None
        tokenizer = None
    elif args.model == "claude-3-5-sonnet-20240620":
        model_name = "claude-3-5-sonnet-20240620"
        model = None
        tokenizer = None
    elif args.model == "o1-mini":
        model_name = "o1-mini"
        model = None
        tokenizer = None
    elif args.model == "o3-mini":
        model_name = "o3-mini"
        model = None
        tokenizer = None
    #elif args.model == "qwen2.5":
    #    model_name = "qwen2.5"
    #    model = None
    #    tokenizer = None
    else:
        print("Invalid model name")
        return
    
    open_ai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo-0125"]

    augment_tasks = ["add_edge", "remove_edge", "add_node", "remove_node", "chain_print", "print_adjacency_matrix", "print_graph"]
    static_tasks = ["edge_exists", "cycle", "node_count", "edge_count", "node_degree", "connected_nodes", "chain_edge_exists", "chain_cycle", "chain_node_count", "chain_edge_count", "chain_node_degree", "chain_connected_nodes"]

    # Clean up: if HuggingFace model, move to device, so make a boolean for this
    if args.model not in ["llama2", "llama3", "llama3.1", "phi", "mistral", "mixtral", "opt", "qwen2", "palm2", "gpt-4o-mini", "gpt-4o", "gemma", "gpt-4-0125-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-0613", "gemini-1.5-flash", "claude-3-haiku-20240307", "o1-mini", "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "qwen2.5", "qwen2.5-math", "o3-mini"]:
        model = model.to(device)

    print(f"Model: {model_name}")
    print(f"Prompt types: {args.prompt_types}")
    print()
    
    for prompt_type in args.prompt_types:
        print('------------------------------------------------------------------------')
        print(f"New Task!! Prompt type: {prompt_type}")
        print('------------------------------------------------------------------------')
        print()
        # Set up data directories
        print(f'args.ablation: {args.ablation}')
        if args.ablation: # Evaluate on ablation graphs
            if args.ablationType == "p":
                print("Evaluating on ablation graphs with p ablation")

                encoding = args.encoding
                modification = args.modification
                chain_length = args.chainLength
                p = args.p
                n = args.n

                input_dir = f"data/{encoding}_chain_p/{prompt_type}/{modification}/{chain_length}/{p}/{n}/input_graphs/"
                prompt_dir = f"data/{encoding}_chain_p/{prompt_type}/{modification}/{chain_length}/{p}/{n}/prompts"
                solution_dir = f"data/{encoding}_chain_p/{prompt_type}/{modification}/{chain_length}/{p}/{n}/solutions"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{p}/{n}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{p}/{n}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{p}/{n}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="p",
                    p=p,
                    n=n,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )

                #return
            elif args.ablationType == "n":
                print("Evaluating on ablation graphs with n ablation")
                for n in range(5, 21):
                    ablation_dir = f"ablation_n/{str(n)}"
                    input_dir = f"data/{ablation_dir}/input_graphs"
                    prompt_dir = f"data/{ablation_dir}/prompts/{prompt_type}"
                    solution_dir = f"data/{ablation_dir}/solutions/{prompt_type}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True)

                    checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="n",
                        n=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir,
                        checkpoint_dir=checkpoint_dir
                        )

                    #print(p, n)
            elif args.ablationType == "d":
                print("Evaluating on ablation graphs with d ablation")
                for n in range(5, 11):
                    ablation_dir = f"ablation_d/{str(n)}"
                    input_dir = f"data/{ablation_dir}/input_graphs"
                    prompt_dir = f"data/{ablation_dir}/prompts/{prompt_type}"
                    solution_dir = f"data/{ablation_dir}/solutions/{prompt_type}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True)

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="d",
                        n=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir
                        )
            elif args.ablationType == "few":
                max_number_of_examples = 5
                for n in range(1, max_number_of_examples+1):
                    print('Evaluating on Few Shot ablation graphs')
                    input_dir = "data/ablation_few/input_graphs"
                    prompt_dir = f"data/ablation_few/prompts/{prompt_type}/{n}"
                    solution_dir = f"data/ablation_few/solutions/{prompt_type}/{n}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True)

                    checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    #print(prompt_dir)
                    #print(solution_dir)

                    #return

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="few",
                        cot_example_number=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir,
                        checkpoint_dir=checkpoint_dir
                    )

                    #return
            elif args.ablationType == "cot":
                max_number_of_examples = 5
                for n in range(1, max_number_of_examples+1):
                    print('Evaluating on CoT ablation graphs')
                    input_dir = "data/ablation_cot/input_graphs"
                    prompt_dir = f"data/ablation_cot/prompts/{prompt_type}/{n}"
                    solution_dir = f"data/ablation_cot/solutions/{prompt_type}/{n}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{model_name}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True)

                    checkpoint_dir = f"checkpoints/{args.ablationType}/{model_name}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    print(prompt_dir)
                    print(solution_dir)

                    #return

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="cot",
                        cot_example_number=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir,
                        checkpoint_dir=checkpoint_dir
                    )

                    #return
            elif args.ablationType == "chain":
                for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
                    for chain_length in range(1, 6):
                        print(f"Evaluating on chain graphs with chain length {chain_length}, construction task {task}, and final task {prompt_type}")
                        input_dir = f"data/chains_same/{prompt_type}/{task}/{chain_length}/input_graphs"
                        prompt_dir = f"data/chains_same/{prompt_type}/{task}/{chain_length}/prompts"
                        solution_dir = f"data/chains_same/{prompt_type}/{task}/{chain_length}/solutions"

                        answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}"
                        os.makedirs(answer_dir, exist_ok=True)

                        results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}"
                        os.makedirs(results_dir, exist_ok=True)

                        checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}"
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        eval(
                            input_dir=input_dir,
                            prompt_dir=prompt_dir,
                            model_name=args.model,
                            tokenizer=tokenizer,
                            device=device,
                            model=model,
                            prompt_type=prompt_type,
                            solution_dir=solution_dir,
                            augment_tasks=augment_tasks,
                            size=args.size,
                            ablation=args.ablation,
                            ablation_type="chain",
                            chain_length=chain_length,
                            task=task,
                            answer_dir=answer_dir,
                            results_dir=results_dir,
                            checkpoint_dir=checkpoint_dir
                        )
            elif args.ablationType == "few_chain":
                for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
                    for chain_length in range(1, 6):

                        input_dir = f"data/chains_same_few/{prompt_type}/{task}/{chain_length}/input_graphs"

                        for num_examples in range(1, 6):
                            print(f"Evaluating on chain graphs with chain length {chain_length}, construction task {task}, final task {prompt_type}, and number of examples {num_examples}")
                            prompt_dir = f"data/chains_same_few/{prompt_type}/{task}/{chain_length}/{num_examples}/prompts"
                            solution_dir = f"data/chains_same_few/{prompt_type}/{task}/{chain_length}/{num_examples}/solutions"

                            answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(answer_dir, exist_ok=True)

                            results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(results_dir, exist_ok=True)

                            checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            eval(
                                input_dir=input_dir,
                                prompt_dir=prompt_dir,
                                model_name=args.model,
                                tokenizer=tokenizer,
                                device=device,
                                model=model,
                                prompt_type=prompt_type,
                                solution_dir=solution_dir,
                                augment_tasks=augment_tasks,
                                size=args.size,
                                ablation=args.ablation,
                                ablation_type="few_chain",
                                chain_length=chain_length,
                                task=task,
                                cot_example_number=num_examples,
                                answer_dir=answer_dir,
                                results_dir=results_dir,
                                checkpoint_dir=checkpoint_dir
                            )
            elif args.ablationType == "cot_chain":
                for task in ["add_edge", "remove_edge", "add_node", "remove_node"]:
                    for chain_length in range(1, 6):

                        input_dir = f"data/chains_same_cot/{prompt_type}/{task}/{chain_length}/input_graphs"

                        for num_examples in range(1, 6):
                            print(f"Evaluating on chain graphs with chain length {chain_length}, construction task {task}, final task {prompt_type}, and number of examples {num_examples}")
                            prompt_dir = f"data/chains_same_cot/{prompt_type}/{task}/{chain_length}/{num_examples}/prompts"
                            solution_dir = f"data/chains_same_cot/{prompt_type}/{task}/{chain_length}/{num_examples}/solutions"

                            answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(answer_dir, exist_ok=True)

                            results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(results_dir, exist_ok=True)

                            checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{task}/{chain_length}/{num_examples}"
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            eval(
                                input_dir=input_dir,
                                prompt_dir=prompt_dir,
                                model_name=args.model,
                                tokenizer=tokenizer,
                                device=device,
                                model=model,
                                prompt_type=prompt_type,
                                solution_dir=solution_dir,
                                augment_tasks=augment_tasks,
                                size=args.size,
                                ablation=args.ablation,
                                ablation_type="cot_chain",
                                chain_length=chain_length,
                                task=task,
                                cot_example_number=num_examples,
                                answer_dir=answer_dir,
                                results_dir=results_dir,
                                checkpoint_dir=checkpoint_dir
                            )
            elif args.ablationType == "barabasi_albert":
                print('Evaluating on Barabasi-Albert graphs')
                input_dir = "data/ablation_graph_type_barabasi_albert/input_graphs"
                prompt_dir = f"data/ablation_graph_type_barabasi_albert/prompts/{prompt_type}"
                solution_dir = f"data/ablation_graph_type_barabasi_albert/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(results_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="barabasi_albert",
                    answer_dir=answer_dir,
                    results_dir=results_dir
                )
            elif args.ablationType == "path":
                print('Evaluating on Path graphs')
                input_dir = "data/ablation_graph_type_path/input_graphs"
                prompt_dir = f"data/ablation_graph_type_path/prompts/{prompt_type}"
                solution_dir = f"data/ablation_graph_type_path/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="path",
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "star":
                print('Evaluating on Star graphs')
                input_dir = "data/ablation_graph_type_star/input_graphs"
                prompt_dir = f"data/ablation_graph_type_star/prompts/{prompt_type}"
                solution_dir = f"data/ablation_graph_type_star/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="star",
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "stochastic_block":
                print('Evaluating on Stochastic Block graphs')
                input_dir = "data/ablation_graph_type_stochastic_block/input_graphs"
                prompt_dir = f"data/ablation_graph_type_stochastic_block/prompts/{prompt_type}"
                solution_dir = f"data/ablation_graph_type_stochastic_block/solutions/{prompt_type}"

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="stochastic_block"
                )
            elif args.ablationType == "encoding":
                encoding = args.encoding
                print(f'Evaluating on {encoding} graphs')
                input_dir = f"data/{encoding}/input_graphs/"
                prompt_dir = f"data/{encoding}/prompts/{prompt_type}"
                solution_dir = f"data/{encoding}/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{encoding}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{encoding}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{encoding}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="encoding",
                    encoding_type=encoding,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "encoding_chain":
                #encodings = ["adjacency_matrix", "incident", "coauthorship"]
                #modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
                encoding = args.encoding
                modification = args.modification
                chain_length = args.chainLength
                #for encoding in encodings:
                #    for modification in modifications:
                #        for chain_length in range(1, 6):
                #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
                print(f'Evaluating on {encoding} chain graphs')
                input_dir = f"data/{encoding}_chain_big/{prompt_type}/{modification}/{chain_length}/input_graphs/"
                prompt_dir = f"data/{encoding}_chain_big/{prompt_type}/{modification}/{chain_length}/prompts"
                solution_dir = f"data/{encoding}_chain_big/{prompt_type}/{modification}/{chain_length}/solutions"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="encoding",
                    encoding_type=encoding,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "encoding_chain_no_print":
                encoding = args.encoding
                modification = args.modification
                chain_length = args.chainLength
                #for encoding in encodings:
                #    for modification in modifications:
                #        for chain_length in range(1, 6):
                #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
                print(f'Evaluating on {encoding} chain graphs')
                input_dir = f"data/{encoding}_chain_no_print/{prompt_type}/{modification}/{chain_length}/input_graphs/"
                prompt_dir = f"data/{encoding}_chain_no_print/{prompt_type}/{modification}/{chain_length}/prompts"
                solution_dir = f"data/{encoding}_chain_no_print/{prompt_type}/{modification}/{chain_length}/solutions"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="encoding",
                    encoding_type=encoding,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "encoding_chain_few":
                encodings = ["adjacency_matrix"]
                modifications = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
                for encoding in encodings:
                    for modification in modifications:
                        for chain_length in range(1, 6):
                            example_number = args.exampleNumber
                            #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
                            print(f'Evaluating on {encoding} chain graphs with few shot examples')
                            input_dir = f"data/{encoding}_chain_big_few/{prompt_type}/{modification}/{chain_length}/{example_number}/input_graphs/"
                            prompt_dir = f"data/{encoding}_chain_big_few/{prompt_type}/{modification}/{chain_length}/{example_number}/prompts"
                            solution_dir = f"data/{encoding}_chain_big_few/{prompt_type}/{modification}/{chain_length}/{example_number}/solutions"

                            answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                            os.makedirs(answer_dir, exist_ok=True)

                            results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                            os.makedirs(results_dir, exist_ok=True)

                            checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            eval(
                                input_dir=input_dir,
                                prompt_dir=prompt_dir,
                                model_name=args.model,
                                tokenizer=tokenizer,
                                device=device,
                                model=model,
                                prompt_type=prompt_type,
                                solution_dir=solution_dir,
                                augment_tasks=augment_tasks,
                                size=args.size,
                                ablation=args.ablation,
                                ablation_type="encoding",
                                encoding_type=encoding,
                                answer_dir=answer_dir,
                                results_dir=results_dir,
                                checkpoint_dir=checkpoint_dir
                            )
            elif args.ablationType == "encoding_chain_cot":
                encoding = args.encoding
                modification = args.modification
                chain_length = args.chainLength
                example_number = args.exampleNumber
                #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
                print(f'Evaluating on {encoding} chain graphs with cot examples')
                input_dir = f"data/{encoding}_chain_big_cot/{prompt_type}/{modification}/{chain_length}/{example_number}/input_graphs/"
                prompt_dir = f"data/{encoding}_chain_big_cot/{prompt_type}/{modification}/{chain_length}/{example_number}/prompts"
                solution_dir = f"data/{encoding}_chain_big_cot/{prompt_type}/{modification}/{chain_length}/{example_number}/solutions"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{example_number}/{encoding}_big"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="encoding",
                    encoding_type=encoding,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "encoding_chain_graph_type":
                graph_type = args.graphType
                encoding = args.encoding
                modification = args.modification
                chain_length = args.chainLength


                #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
                print(f'Evaluating on {encoding} chain graphson graph type {graph_type}')
                input_dir = f"data/{encoding}_chain_big_{graph_type}/{prompt_type}/{modification}/{chain_length}/input_graphs/"
                prompt_dir = f"data/{encoding}_chain_big_{graph_type}/{prompt_type}/{modification}/{chain_length}/prompts"
                solution_dir = f"data/{encoding}_chain_big_{graph_type}/{prompt_type}/{modification}/{chain_length}/solutions"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{graph_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{graph_type}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{modification}/{chain_length}/{encoding}_big/{graph_type}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="encoding",
                    encoding_type=encoding,
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "no_force":
                print('Evaluating on normal graphs without forcing output')
                input_dir = "data/ablation_no_force/input_graphs"
                prompt_dir = f"data/ablation_no_force/prompts/{prompt_type}"
                solution_dir = f"data/ablation_no_force/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="no_force",
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            elif args.ablationType == "preserve_path":
                for n in range(5, 15):
                    print('Evaluating on path graphs, testing their ability to preserve the underlying graph structure')
                    input_dir = f"data/ablation_preserve/path/{n}/input_graphs"
                    prompt_dir = f"data/ablation_preserve/path/{n}/prompts/{prompt_type}"
                    solution_dir = f"data/ablation_preserve/path/{n}/solutions/{prompt_type}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True) 

                    checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="preserve_path",
                        n=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir,
                        checkpoint_dir=checkpoint_dir
                    )
            elif args.ablationType == "preserve_star":
                for n in range(5, 15):
                    print('Evaluating on star graphs, testing their ability to preserve the underlying graph structure')
                    input_dir = f"data/ablation_preserve/star/{n}/input_graphs"
                    prompt_dir = f"data/ablation_preserve/star/{n}/prompts/{prompt_type}"
                    solution_dir = f"data/ablation_preserve/star/{n}/solutions/{prompt_type}"

                    answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(answer_dir, exist_ok=True)

                    results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(results_dir, exist_ok=True)

                    checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}/{n}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    eval(
                        input_dir=input_dir,
                        prompt_dir=prompt_dir,
                        model_name=args.model,
                        tokenizer=tokenizer,
                        device=device,
                        model=model,
                        prompt_type=prompt_type,
                        solution_dir=solution_dir,
                        augment_tasks=augment_tasks,
                        size=args.size,
                        ablation=args.ablation,
                        ablation_type="preserve_star",
                        n=n,
                        answer_dir=answer_dir,
                        results_dir=results_dir,
                        checkpoint_dir=checkpoint_dir
                    )
            elif args.ablationType == "node_connect":
                print('Evaluating on normal graphs while ensuring the model doesn\'t connect the new node')
                input_dir = "data/ablation_add_node_without_connecting/input_graphs"
                prompt_dir = f"data/ablation_add_node_without_connecting/prompts/{prompt_type}"
                solution_dir = f"data/ablation_add_node_without_connecting/solutions/{prompt_type}"

                answer_dir = f"answers/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(answer_dir, exist_ok=True)

                results_dir = f"results/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(results_dir, exist_ok=True)

                checkpoint_dir = f"checkpoints/{args.ablationType}/{args.model}_{args.size}/{prompt_type}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                eval(
                    input_dir=input_dir,
                    prompt_dir=prompt_dir,
                    model_name=args.model,
                    tokenizer=tokenizer,
                    device=device,
                    model=model,
                    prompt_type=prompt_type,
                    solution_dir=solution_dir,
                    augment_tasks=augment_tasks,
                    size=args.size,
                    ablation=args.ablation,
                    ablation_type="no_force",
                    answer_dir=answer_dir,
                    results_dir=results_dir,
                    checkpoint_dir=checkpoint_dir
                )
            else:
                print("Invalid ablation type")
                return
        else: # Evaluate on normal graphs
            print('Evaluating on normal graphs')
            input_dir = "data/input_graphs"
            prompt_dir = f"data/prompts/{prompt_type}"
            solution_dir = f"data/solutions/{prompt_type}"

            answer_dir = f"answers/base/{args.model}_{args.size}/{prompt_type}"
            os.makedirs(answer_dir, exist_ok=True)

            results_dir = f"results/base/{args.model}_{args.size}/{prompt_type}"
            os.makedirs(results_dir, exist_ok=True)

            eval(
                input_dir=input_dir,
                prompt_dir=prompt_dir,
                model_name=args.model,
                tokenizer=tokenizer,
                device=device,
                model=model,
                prompt_type=prompt_type,
                solution_dir=solution_dir,
                augment_tasks=augment_tasks,
                size=args.size,
                ablation=args.ablation
            )

if __name__ == "__main__":
    main()