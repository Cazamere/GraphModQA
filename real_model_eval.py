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

def eval(input_dir, model_name, tokenizer, device, model, augment_tasks, size, answer_dir, results_dir, checkpoint_dir, task):
    client = OpenAI()

    # Iterate over files
    score = 0

    #number_of_files_in_dataset = len([filename for filename in os.listdir(input_dir) if filename.endswith(".graphml")])

    #print(f"Number of files in dataset: {number_of_files_in_dataset}")

    #checkpoint_filename = "checkpoint.txt"

    #last_processed_index = load_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename)) # REMEMBER: to manually reset this, delete the checkpoint file

    #print(f"Last processed index: {last_processed_index}")

    #for i in range(last_processed_index, number_of_files_in_dataset):
    for i in range(1, 25):
        print('-----------------------------------')
        print(f"Evaluating on graph {i}")

        prompt_filename = f"{i}/{task}/prompt.txt"
        solution_filename = f"{i}/{task}/solution.txt"
        
       # Read prompt
        with open(os.path.join(input_dir, prompt_filename), "r") as prompt_file:
            prompt = prompt_file.read()

        if model_name in ['gpt-4o-mini', 'gpt-4o', 'gpt-4-0125-preview', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gpt-4-0613', 'o1-mini', 'o3-mini']:
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
            #print(f"Model output: {response.choices[0].message.content}")
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

            
        elif model_name in ['claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620', 'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking']:
            claude_client = anthropic.Anthropic()
            mt = 4096 if model_name == 'claude-3-haiku-20240307' else 8192

            if model_name == 'claude-3-7-sonnet-20250219-thinking':
                response = claude_client.beta.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=128000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 32000
                    },
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    betas=["output-128k-2025-02-19"]
                )
                print(response)
                sys.exit(0)
            else:
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
            #print(f"Model output: {output_string}")

            
        elif model_name in ['llama3.1', 'deepseek']:

            if model_name == 'llama3.1':
                router_model_name = "meta-llama/llama-3.1-405b-instruct"
            elif model_name == 'deepseek':
                router_model_name = "deepseek/deepseek-r1"

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
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

        
        with open(os.path.join(input_dir, solution_filename), "r") as solution_file:
            solution = solution_file.read()
        
        print()
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

        if task in ["node_count", "edge_count", "node_degree", "triangle_count"]:
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
            #sys.exit(0)
        elif task in ["overlapped_nodes", "overlapped_edges"]:
            final_answer = extract_final_answer(client=client, output_string=output_string, prompt_modifier="yes/no")
            if "Yes" in final_answer:
                    final_answer = "Yes"
            elif "No" in final_answer:
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
        elif task in ["isolated_nodes", "connected_nodes"]:
            final_answer = extract_final_answer(client=client, output_string=output_string, prompt_modifier="list_direct", solution=solution)
            if "Yes" in final_answer or "yes" in final_answer:
                print('Correct!')
                score += 1
                correct = True
            else:
                print('Incorrect!')
            #sys.exit(0)
        elif task in ["print_graph"]:
            correct = extract_final_answer_encoding(client=client, output_string=output_string, solution=solution)

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
        else:
            print('Invalid task type')
            return

        correct_print = "Correct" if correct else "Incorrect"

        with open(f"{answer_dir}/{i}.txt", "w") as answer_file:
            #answer_file.write(f"Task: {prompt}\n Solution: {solution}\n Model output: {final_answer}\n{correct_print}\nError: {error}\nn: {graph_n}\np: {graph_p}\n")
            answer_file.write(f"Task: {prompt}\n Solution: {solution}\n Model output: {final_answer}\n{correct_print}\nError: {error}")

        # Save the current index as the checkpoint
        #save_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename), i + 1)  # Save the next index to resume from

    # Calculate accuracy
    accuracy = get_answer_directory_accuracy(answer_directory=answer_dir)

    print(f"Accuracy: {accuracy}%")

    # Save the accuracy to a file
    with open(f"{results_dir}/results.txt", "w") as results_file:
        results_file.write(f"Accuracy: {accuracy}%")

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
    parser.add_argument("--model", choices=["llama2", "llama3", "opt", "openelm", "mistral", "mixtral", "phi", "gpt-4o-mini", "gpt-4o", "gpt-4-0125-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-0613", "o1-mini", "qwen2", "qwen2-math", "palm2", "gemma", "llama3.1", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620", "qwen2.5", "qwen2.5-math", "o3-mini", "deepseek", 'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking'], required=True, help="Specify the transformer model")
    parser.add_argument("--size", type=str, required=True, help="Specify the size of the model (OPT: 125m, 350m, 1.3b, 6.7b, OpenELM: 270M, 450M, 1_1B, 3B, Llama2: 7b, 13b, 70b, Llama3: 8B, 70B, Mistral: 7B, Phi: 4k, 128k)")
    parser.add_argument("--graph_size", required=True, choices=["small", "medium", "large"], help="bucket size")
    parser.add_argument("--task", required=True, choices=["node_count", "edge_count", "node_degree", "connected_nodes", "print_graph", "isolated_nodes", "triangle_count", "overlapped_nodes", "overlapped_edges"], help="Ablation study")
    args = parser.parse_args()

    # print all arguments
    args.ablation = True

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
    elif args.model == "claude-3-7-sonnet-20250219":
        model_name = "claude-3-7-sonnet-20250219"
        model = None
        tokenizer = None
    elif args.model == "claude-3-7-sonnet-20250219-thinking":
        model_name = "claude-3-7-sonnet-20250219-thinking"
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
    elif args.model == "deepseek":
        model_name = "deepseek"
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
    if args.model not in ["llama2", "llama3", "llama3.1", "phi", "mistral", "mixtral", "opt", "qwen2", "palm2", "gpt-4o-mini", "gpt-4o", "gemma", "gpt-4-0125-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-0613", "gemini-1.5-flash", "claude-3-haiku-20240307", "o1-mini", "claude-3-5-sonnet-20240620", "gemini-1.5-pro", "qwen2.5", "qwen2.5-math", "o3-mini", "deepseek", 'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking']:
        model = model.to(device)

    print(f"Model: {model_name}")
    print()
    
    graph_size = args.graph_size
    task = args.task

    #if modification == "mix" and chain_length == 5 and encoding == "adjacency_matrix":
    print(f'Evaluating on {graph_size} graphs')
    input_dir = f"data/coauth/{graph_size}/"

    answer_dir = f"answers/coauth/{graph_size}/{task}/{model_name}/"
    os.makedirs(answer_dir, exist_ok=True)

    results_dir = f"results/coauth/{graph_size}/{task}/{model_name}/"
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_dir = f"checkpoints/coauth/{graph_size}/{task}/{model_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    eval(
        input_dir=input_dir,
        model_name=args.model,
        tokenizer=tokenizer,
        device=device,
        model=model,
        augment_tasks=augment_tasks,
        size=args.size,
        answer_dir=answer_dir,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        task=task
    )

if __name__ == "__main__":
    main()
