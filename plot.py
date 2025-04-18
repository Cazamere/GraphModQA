import argparse
import argparse
import re
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import sem
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def compare_models():
        modification_tasks = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]#["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
        encodings = ["incident", "coauthorship"]
        final_tasks = ["print_graph"]
        models = ["claude-3-5-sonnet-20240620_70B", "llama3.1_405B", "o1-mini_70B", "gpt-4o-mini_70B"]

        assert len(modification_tasks) > 1, "At least two modification tasks are required to compare the models"
        
        for encoding_type in encodings:
            # Create a dictionary to store the accuracy lists for each model
            
            # Iterate from i = 1 to 5
            for final_task in final_tasks:
                modification_dict = {}
                for modification_task in modification_tasks:
                    # Iterate over the models
                    model_accuracy_dict = {}
                    for model_name in models:
                        accuracy_list = []
                        map_accuracy_list = []
                        
                        for i in range(1, 6):
                            if final_task == "print_graph" and encoding_type == "adjacency_matrix":
                                # get no print accuracy
                                file_path = f"results/encoding_chain_no_print/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                                try:
                                    with open(file_path, "r") as file:
                                        text = file.read()
                                        #print(f"text: {text}")
                                        # Extract the accuracy value using regular expression
                                        accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                        #print(f"accuracy: {accuracy}")
                                        if accuracy:
                                            accuracy_value = round(float(accuracy.group(1)), 2)
                                            accuracy_list.append(accuracy_value)
                                except FileNotFoundError:
                                    #print(f"File not found: {file_path}")
                                    #continue
                                    pass
                            else:
                                file_path = f"results/encoding_chain/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                                try:
                                    with open(file_path, "r") as file:
                                        text = file.read()
                                        # Extract the accuracy value using regular expression
                                        accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                        if accuracy:
                                            accuracy_value = round(float(accuracy.group(1)), 2)
                                            accuracy_list.append(accuracy_value)
                                except FileNotFoundError:
                                    continue
                        model_accuracy_dict[model_name] = accuracy_list


                        # Store the accuracy list for the current model in the dictionary
                        #accuracy_dict[modification_task] = accuracy_list

                        print(f"Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {accuracy_list}")
                    modification_dict[modification_task] = model_accuracy_dict
                    print()
                
                # Plot the accuracy values for each model on subplots for all modification tasks
                fig, axs = plt.subplots(1, len(modification_tasks), figsize=(6*len(modification_tasks), 6), sharex=True, sharey=True)
                i = 0
                for modification, model_accuracy_dict in modification_dict.items():
                    if modification == "add_edge":
                        axs[i].set_title("Modification Task: Add Edge")
                    elif modification == "remove_edge":
                        axs[i].set_title("Modification Task: Remove Edge")
                    elif modification == "add_node":
                        axs[i].set_title("Modification Task: Add Node")
                    elif modification == "remove_node":
                        axs[i].set_title("Modification Task: Remove Node")
                    elif modification == "mix":
                        axs[i].set_title("Modification Task: Mix")
                    #axs[i].set_title(f"Modification Task: {modification}")
                    for model_name, accuracy_list in model_accuracy_dict.items():
                        if model_name == "claude-3-5-sonnet-20240620_70B":
                            legend_model_name = "Claude 3.5"
                        elif model_name == "llama3.1_405B":
                            legend_model_name = "Llama 3.1"
                        elif model_name == "o1-mini_70B":
                            legend_model_name = "o1-mini"
                        elif model_name == "gpt-4o-mini_70B":
                            legend_model_name = "GPT-4o-mini"
                        axs[i].plot(range(1, len(accuracy_list) + 1), accuracy_list, label=legend_model_name, linewidth=2.5)
                    axs[i].set_xlabel('Number of modifications', fontsize=12)
                    if i == 0:
                        axs[i].set_ylabel('Accuracy', fontsize=12)
                    axs[i].set_xlim(1, 5)
                    axs[i].set_ylim(0, 100)
                    axs[i].set_xticks(range(1, 6))
                    axs[i].tick_params(right=False, top=False)  # Disable right vertical line and top horizontal line
                    axs[i].spines['top'].set_visible(False)  # Hide the top spine
                    axs[i].spines['right'].set_visible(False)  # Hide the right spine
                    axs[i].legend(fontsize=12)  # Show legend with model names
                    axs[i].tick_params(labelsize=12)  # Set tick label font size
                    axs[i].title.set_fontsize(12)  # Set title font size
                    axs[i].xaxis.label.set_fontsize(12)  # Set x-axis label font size
                    axs[i].yaxis.label.set_fontsize(12)  # Set y-axis label font size
                    for item in axs[i].get_xticklabels() + axs[i].get_yticklabels():
                        item.set_fontsize(10)  # Set tick label font size
                    i += 1
                plt.tight_layout()  # Adjust the spacing between subplots
                plt.show()
                #plt.title(f"Final Task: {final_task}")
                plot_file_dir = f"plots/encoding_chain/{final_task}/{encoding_type}"
                os.makedirs(plot_file_dir, exist_ok=True)
                plot_file_path = f"{plot_file_dir}/plot.png"
                plt.savefig(plot_file_path)
                #plt.clf()

                for ax in axs:
                    plt.figure(figsize=(10.5, 8.5))  # Set the figure size to 10 inches by 6 inches
                    #print(ax.get_lines())
                    linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
                    for m in range(len(models)):
                        plt.plot(ax.get_lines()[m].get_xdata(), ax.get_lines()[m].get_ydata(), label=models[m], linestyle=linestyle[m], linewidth=2.5)
                    #plt.plot(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata())
                    #plt.plot(ax.get_lines()[1].get_xdata(), ax.get_lines()[1].get_ydata())
                    #plt.plot(ax.get_lines()[2].get_xdata(), ax.get_lines()[2].get_ydata())
                    #plt.plot(ax.get_lines()[3].get_xdata(), ax.get_lines()[3].get_ydata())
                    plt.xlabel('Number of modifications', fontsize=30)
                    plt.ylabel('Accuracy', fontsize=30)
                    plt.title(ax.get_title(), fontsize=30)
                    legend_labels = [line.get_label() for line in ax.get_lines()]
                    plt.legend(legend_labels, fontsize=20)
                    plt.xlim(1, 5)
                    if not (final_task == "print_graph" and encoding_type == "adjacency_matrix"):
                        plt.ylim(0, 105)
                    plt.xticks(range(1, 6), fontsize=19)
                    plt.yticks(fontsize=19)
                    plt.tick_params(right=False, top=False)
                    # Hide the right and top spines
                    #ax.spines[['right', 'top']].set_visible(False)
                    plt.savefig(f"{plot_file_dir}/{ax.get_title().replace(' ', '_')}.png")
                    plt.clf()
                #return

        print("#----------------------#")

def in_context():
        modification_tasks = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
        encodings = ["adjacency_matrix"]#["adjacency_matrix", "incident", "coauthorship"]
        final_tasks = ["print_graph"]#["node_count", "edge_count", "node_degree", "connected_nodes", "print_graph"]
        models = ["claude-3-5-sonnet-20240620_70B", "llama3.1_405B", "o1-mini_70B", "gpt-4o-mini_70B"]
        
        for encoding_type in encodings:
            # Create a dictionary to store the accuracy lists for each model
            # Iterate from i = 1 to 5
            for final_task in final_tasks:
                modification_dict = {}
                for model_name in models:
                    # Plot the accuracy values for each model on subplots for all modification tasks
                    fig, axs = plt.subplots(1, len(modification_tasks), figsize=(6*len(modification_tasks), 6), sharex=True, sharey=True)
                    x = 0
                    for modification_task in modification_tasks:
                                        
                        accuracy_list = []
                        map_accuracy_list = []
                        cot_accuracy_dict = {}
                        for i in range(1, 6):
                            
                            # get no print accuracy
                            file_path = f"results/encoding_chain_no_print/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                            try:
                                with open(file_path, "r") as file:
                                    text = file.read()
                                    #print(f"text: {text}")
                                    # Extract the accuracy value using regular expression
                                    accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                    #print(f"accuracy: {accuracy}")
                                    if accuracy:
                                        accuracy_value = round(float(accuracy.group(1)), 2)
                                        accuracy_list.append(accuracy_value)
                            except FileNotFoundError:
                                #continue
                                pass

                            # get MAP accuracy
                            file_path = f"results/encoding_chain/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                            try:
                                with open(file_path, "r") as file:
                                    text = file.read()
                                    #print(f"text: {text}")
                                    # Extract the accuracy value using regular expression
                                    accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                    #print(f"accuracy: {accuracy}")
                                    if accuracy:
                                        accuracy_value = round(float(accuracy.group(1)), 2)
                                        map_accuracy_list.append(accuracy_value)
                            except FileNotFoundError:
                                #continue
                                pass

                            cot_accuracy_list = []
                            # get cot accuracy
                            for k in range(1, 4):
                                file_path = f"results/encoding_chain_cot/{model_name}/{final_task}/{modification_task}/{i}/{k}/{encoding_type}_big/results.txt"
                                try:
                                    with open(file_path, "r") as file:
                                        text = file.read()
                                        #print(f"text: {text}")
                                        # Extract the accuracy value using regular expression
                                        accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                        #print(f"accuracy: {accuracy}")
                                        if accuracy:
                                            accuracy_value = round(float(accuracy.group(1)), 2)
                                            cot_accuracy_list.append(accuracy_value)
                                except FileNotFoundError:
                                    #print(f"File not found: {file_path}")
                                    #continue
                                    pass
                            cot_accuracy_dict[i] = cot_accuracy_list

                        print(f"Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {accuracy_list}")
                        if final_task == "print_graph":
                            if len(map_accuracy_list) > 0:
                                print(f"MaP: Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {map_accuracy_list}")
                            if len(cot_accuracy_dict) > 0:
                                for j, cot_accuracy_list in cot_accuracy_dict.items():
                                    print(f"COT: Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Iteration: {j}, Accuracy (--- # examples --->): {cot_accuracy_list}")

                        cot_accuracies = [[], [], []]
                        for z in range(3):
                            for j, cot_accuracy_list in cot_accuracy_dict.items():
                                if len(cot_accuracy_list) > z:
                                    cot_accuracies[z].append(cot_accuracy_list[z])
                                else:
                                    cot_accuracies[z].append(0)

                        print(f"cot_accuracies: {cot_accuracies}")
                        

                        axs[x].plot(range(1, len(accuracy_list) + 1), accuracy_list, label="0-shot", linewidth=2.5)
                        if len(map_accuracy_list) > 0:
                            axs[x].plot(range(1, len(map_accuracy_list) + 1), map_accuracy_list, label="MAP", linestyle='dotted', linewidth=2.5)
                        if len(cot_accuracy_dict) > 0:
                            for j in range(3):
                                #print(range(1, 6))
                                #print(cot_accuracies[j])
                                axs[x].plot(range(1, 6), cot_accuracies[j], label=f"CoT, # examples: {j+1}", linestyle='--', linewidth=2.5)

                        # save plot
                        axs[x].set_xlabel('Number of modifications')
                        axs[x].set_ylabel('Accuracy')
                        if modification_task == "add_edge":
                            axs[x].set_title("Modification Task: Add Edge")
                        elif modification_task == "remove_edge":
                            axs[x].set_title("Modification Task: Remove Edge")
                        elif modification_task == "add_node":
                            axs[x].set_title("Modification Task: Add Node")
                        elif modification_task == "remove_node":
                            axs[x].set_title("Modification Task: Remove Node")
                        elif modification_task == "mix":
                            axs[x].set_title("Modification Task: Mix")
                        axs[x].set_xlim(1, 5)
                        #plt.ylim(0, 100)
                        axs[x].set_xticks(range(1, 6))
                        axs[x].tick_params(right=False, top=False)
                        axs[x].legend()
                        #plt.spines['top'].set_visible(False)
                        #plt.spines['right'].set_visible(False)
                        x += 1
                    plt.tight_layout()
                    plot_file_dir = f"plots/encoding_chain/{model_name}/{final_task}/"
                    os.makedirs(plot_file_dir, exist_ok=True)
                    plot_file_path = f"{plot_file_dir}/plot.png"
                    plt.savefig(plot_file_path)
                    #plt.clf()

                    for ax in axs:
                        plt.figure(figsize=(10.5, 8.5))  # Set the figure size to 10 inches by 6 inches
                        for k in range(5):
                            # get label
                            label = ax.get_lines()[k].get_label()
                            if label == "MAP":
                                linestyle = 'dotted'
                            elif "CoT" in label:
                                linestyle = '--'
                            else:
                                linestyle = 'solid'
                            plt.plot(ax.get_lines()[k].get_xdata(), ax.get_lines()[k].get_ydata(), label=label, linestyle=linestyle, linewidth=2.5)
                        #plt.plot(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata())
                        plt.xlabel('Number of modifications', fontsize=30)
                        plt.ylabel('Accuracy', fontsize=30)
                        plt.title(ax.get_title(), fontsize=30)
                        legend_labels = [line.get_label() for line in ax.get_lines()]
                        plt.legend(legend_labels, fontsize=20)
                        plt.xlim(1, 5)
                        plt.xticks(range(1, 6), fontsize=19)
                        plt.yticks(fontsize=19)
                        plt.tick_params(right=False, top=False)
                        plt.savefig(f"{plot_file_dir}/in_context_{ax.get_title().replace(' ', '_')}.png")
                        plt.clf()

                    """
                    for ax in axs:
                        plt.figure(figsize=(10.5, 8.5))  # Set the figure size to 10 inches by 6 inches
                        #print(ax.get_lines())
                        for m in range(len(models)):
                            plt.plot(ax.get_lines()[m].get_xdata(), ax.get_lines()[m].get_ydata())
                        #plt.plot(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata())
                        #plt.plot(ax.get_lines()[1].get_xdata(), ax.get_lines()[1].get_ydata())
                        #plt.plot(ax.get_lines()[2].get_xdata(), ax.get_lines()[2].get_ydata())
                        #plt.plot(ax.get_lines()[3].get_xdata(), ax.get_lines()[3].get_ydata())
                        plt.xlabel('Number of modifications', fontsize=30)
                        plt.ylabel('Accuracy', fontsize=30)
                        plt.title(ax.get_title(), fontsize=30)
                        legend_labels = [line.get_label() for line in ax.get_lines()]
                        plt.legend(legend_labels, fontsize=20)
                        plt.xlim(1, 5)
                        plt.xticks(range(1, 6), fontsize=19)
                        plt.yticks(fontsize=19)
                        plt.tick_params(right=False, top=False)
                        plt.savefig(f"{plot_file_dir}/{ax.get_title().replace(' ', '_')}.png")
                        plt.clf()
                    """

        print("#----------------------#")

        


        """
        # Plot the accuracy values for each model on the same figure
        for model_name, accuracy_list in accuracy_dict.items():
            plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label=model_name)

        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Values')
        plt.xlim(1, 5)
        plt.ylim(0, 100)
        plt.xticks(range(1, 6))
        plt.tick_params(right=False, top=False)  # Disable right vertical line and top horizontal line
        plt.legend()  # Show legend with model names
        plt.show()

        return

        # Save the plot for each model
        for model_name, accuracy_list in accuracy_dict.items():
            plot_file_dir = f"plots/encoding_chain/{model_name}"
            os.makedirs(plot_file_dir, exist_ok=True)
            plot_file_path = f"{plot_file_dir}/plot.png"
            plt.plot(range(1, len(accuracy_list) + 1), accuracy_list)
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Values')
            plt.xlim(1, 5)
            plt.ylim(0, 100)
            plt.xticks(range(1, 6))
            plt.tick_params(right=False, top=False)  # Disable right vertical line and top horizontal line
            plt.savefig(plot_file_path)
            plt.clf()
        """

def graph_types():
        modification_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
        encodings = ["adjacency_matrix"]#["adjacency_matrix", "incident", "coauthorship"]
        final_tasks = ["print_graph"]#["node_count", "edge_count", "node_degree", "connected_nodes", "print_graph"]
        models = ["claude-3-5-sonnet-20240620_70B", "llama3.1_405B", "o1-mini_70B"]#["claude-3-5-sonnet-20240620_70B", "llama3.1_405B", "o1-mini_70B"]
        
        for encoding_type in encodings:
            # Create a dictionary to store the accuracy lists for each model
            # Iterate from i = 1 to 5
            for final_task in final_tasks:
                modification_dict = {}
                for model_name in models:
                    # Plot the accuracy values for each model on subplots for all modification tasks
                    fig, axs = plt.subplots(1, len(modification_tasks), figsize=(6*len(modification_tasks), 6), sharex=True, sharey=True)
                    x = 0
                    for modification_task in modification_tasks:
                                        
                        accuracy_list = []
                        map_accuracy_list = []
                        graph_type_accuracy_dict = {
                            "star": [],
                            "path": [],
                            "complete": [],
                            "empty": []
                        }
                        for i in range(1, 4):
                            
                            # get no print accuracy
                            file_path = f"results/encoding_chain_no_print/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                            try:
                                with open(file_path, "r") as file:
                                    text = file.read()
                                    #print(f"text: {text}")
                                    # Extract the accuracy value using regular expression
                                    accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                    #print(f"accuracy: {accuracy}")
                                    if accuracy:
                                        accuracy_value = round(float(accuracy.group(1)), 2)
                                        accuracy_list.append(accuracy_value)
                            except FileNotFoundError:
                                #continue
                                pass

                            cot_accuracy_list = []
                            # get graph type accuracy
                            for graph_type in ["star", "path", "complete", "empty"]:
                                if modification_task == "add_edge" and graph_type == "complete":
                                    continue
                                elif modification_task == "remove_edge" and graph_type == "empty":
                                    continue
                                file_path = f"results/encoding_chain_graph_type/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/{graph_type}/results.txt"
                                try:
                                    with open(file_path, "r") as file:
                                        text = file.read()
                                        #print(f"text: {text}")
                                        # Extract the accuracy value using regular expression
                                        accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                        #print(f"accuracy: {accuracy}")
                                        if accuracy:
                                            accuracy_value = round(float(accuracy.group(1)), 2)
                                            graph_type_accuracy_dict[graph_type].append(accuracy_value)
                                except FileNotFoundError:
                                    #print(f"File not found: {file_path}")
                                    #continue
                                    pass
                            
                        """
                        print(f"Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {accuracy_list}")
                        if final_task == "print_graph":
                            if len(map_accuracy_list) > 0:
                                print(f"MaP: Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {map_accuracy_list}")
                            if len(cot_accuracy_dict) > 0:
                                for j, cot_accuracy_list in cot_accuracy_dict.items():
                                    print(f"COT: Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Iteration: {j}, Accuracy (--- # examples --->): {cot_accuracy_list}")
                        """
                        #cot_accuracies = [[], [], []]
                        #for z in range(3):
                        #    for j, cot_accuracy_list in cot_accuracy_dict.items():
                        #        if len(cot_accuracy_list) > z:
                        #            cot_accuracies[z].append(cot_accuracy_list[z])
                        #        else:
                        #            cot_accuracies[z].append(0)
                        #print(f"cot_accuracies: {cot_accuracies}")

                        for graph_type, graph_type_accuracy_list in graph_type_accuracy_dict.items():
                            print(f"Graph Type: {graph_type}, Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {graph_type_accuracy_list}")
                            axs[x].plot(range(1, len(graph_type_accuracy_list) + 1), graph_type_accuracy_list, label=graph_type, linewidth=2.5)

                        #return
                        

                        #axs[x].plot(range(1, len(accuracy_list) + 1), accuracy_list, label="0-shot", linewidth=2.5)
                        #if len(map_accuracy_list) > 0:
                        #    axs[x].plot(range(1, len(map_accuracy_list) + 1), map_accuracy_list, label="MaP", linestyle='dotted', linewidth=2.5)
                        #if len(cot_accuracy_dict) > 0:
                        #    for j in range(3):
                        #        axs[x].plot(range(1, 6), cot_accuracies[j], label=f"CoT, # examples: {j+1}", linestyle='--', linewidth=2.5)

                        # save plot
                        axs[x].set_xlabel('Number of modifications')
                        axs[x].set_ylabel('Accuracy')
                        if modification_task == "add_edge":
                            axs[x].set_title("Modification Task: Add Edge")
                        elif modification_task == "remove_edge":
                            axs[x].set_title("Modification Task: Remove Edge")
                        elif modification_task == "add_node":
                            axs[x].set_title("Modification Task: Add Node")
                        elif modification_task == "remove_node":
                            axs[x].set_title("Modification Task: Remove Node")
                        elif modification_task == "mix":
                            axs[x].set_title("Modification Task: Mix")
                        axs[x].set_xlim(1, 3)
                        axs[x].set_xticks(range(1, 3))
                        axs[x].tick_params(right=False, top=False)
                        axs[x].legend()
                        x += 1
                    plt.tight_layout()
                    plt.ylim(0, 100)
                    plot_file_dir = f"plots/encoding_chain_graph_type/{model_name}/{final_task}/"
                    os.makedirs(plot_file_dir, exist_ok=True)
                    plot_file_path = f"{plot_file_dir}/plot.png"
                    plt.savefig(plot_file_path)
                    #plt.clf()

                    for ax in axs:
                        plt.figure(figsize=(10.5, 8.5))
                        for k in range(4):
                            # get label
                            label = ax.get_lines()[k].get_label()
                            plt.plot(ax.get_lines()[k].get_xdata(), ax.get_lines()[k].get_ydata(), label=label, linewidth=2.5)
                        #plt.plot(ax.get_lines()[0].get_xdata(), ax.get_lines()[0].get_ydata())
                        plt.xlabel('Number of modifications', fontsize=30)
                        plt.ylabel('Accuracy', fontsize=30)
                        plt.title(ax.get_title(), fontsize=30)
                        legend_labels = [line.get_label() for line in ax.get_lines()]
                        plt.legend(legend_labels, fontsize=20)
                        plt.xlim(1, 3)
                        plt.ylim(0, 100)
                        plt.xticks(range(1, 4), fontsize=19)
                        plt.yticks(fontsize=19)
                        plt.tick_params(right=False, top=False)
                        plt.savefig(f"{plot_file_dir}/{ax.get_title().replace(' ', '_')}.png")
                        plt.clf()

        print("#----------------------#")

def stat():
        modification_tasks = ["add_edge", "remove_edge", "add_node", "remove_node", "mix"]
        encodings = ["adjacency_matrix"]#["adjacency_matrix", "incident", "coauthorship"]
        final_tasks = ["print_graph"]#["node_count", "edge_count", "node_degree", "connected_nodes", "print_graph"]
        models = ["claude-3-5-sonnet-20240620_70B", "llama3.1_405B", "o1-mini_70B", "gpt-4o-mini_70B"]
        
        for encoding_type in encodings:
            # Create a dictionary to store the accuracy lists for each model
            # Iterate from i = 1 to 5
            for final_task in final_tasks:
                modification_dict = {}
                for model_name in models:
                    # Plot the accuracy values for each model on subplots for all modification tasks
                    x = 0
                    for modification_task in modification_tasks:
                                        
                        accuracy_list = []
                        map_accuracy_list = []
                        for i in range(1, 6):
                            
                            # get no print accuracy
                            file_path = f"results/encoding_chain_no_print/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                            try:
                                with open(file_path, "r") as file:
                                    text = file.read()
                                    #print(f"text: {text}")
                                    # Extract the accuracy value using regular expression
                                    accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                    #print(f"accuracy: {accuracy}")
                                    if accuracy:
                                        accuracy_value = round(float(accuracy.group(1)), 2)
                                        accuracy_list.append(accuracy_value)
                            except FileNotFoundError:
                                #continue
                                pass

                            # get MAP accuracy
                            file_path = f"results/encoding_chain/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/results.txt"
                            try:
                                with open(file_path, "r") as file:
                                    text = file.read()
                                    #print(f"text: {text}")
                                    # Extract the accuracy value using regular expression
                                    accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                    #print(f"accuracy: {accuracy}")
                                    if accuracy:
                                        accuracy_value = round(float(accuracy.group(1)), 2)
                                        map_accuracy_list.append(accuracy_value)
                            except FileNotFoundError:
                                #continue
                                pass



                        if model_name == "claude-3-5-sonnet-20240620_70B":
                            legend_model_name = "Claude 3.5 Sonnet"
                        elif model_name == "llama3.1_405B":
                            legend_model_name = "Llama 3.1 405B"
                        elif model_name == "o1-mini_70B":
                            legend_model_name = "o1-mini"
                        elif model_name == "gpt-4o-mini_70B":
                            legend_model_name = "GPT-4o mini"
                        print(f"Model: {legend_model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {accuracy_list}")
                        if final_task == "print_graph":
                            if len(map_accuracy_list) > 0:
                                print(f"MaP: Model: {model_name}, Final Task: {final_task}, Encoding: {encoding_type}, Modification Task: {modification_task}, Accuracy: {map_accuracy_list}")
                        
                        threshold = 0.05

                        # Perform a two-sample t-test
                        t_stat, p_value = ttest_rel(map_accuracy_list, accuracy_list)  # Welch's t-test
                        
                        # Calculate means and standard errors
                        mean1, mean2 = np.mean(map_accuracy_list), np.mean(accuracy_list)
                        se1, se2 = sem(map_accuracy_list), sem(accuracy_list)
                        
                        # Compute the standard error of the difference
                        diff_se = np.sqrt(se1**2 + se2**2)
                        diff_mean = mean1 - mean2
                        
                        # Confidence interval range
                        ci_low = diff_mean - 1.96 * diff_se  # for 95% confidence, z = 1.96
                        ci_high = diff_mean + 1.96 * diff_se

                        # Calculate the standard deviations of the two groups
                        std1, std2 = np.std(map_accuracy_list, ddof=1), np.std(accuracy_list, ddof=1)  # Use ddof=1 for sample std
                        
                        # Pooled standard deviation
                        n1, n2 = len(map_accuracy_list), len(accuracy_list)
                        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                        
                        # Calculate Cohen's d
                        cohen_d = (mean1 - mean2) / pooled_std
                        
                        # Interpret the effect size
                        if abs(cohen_d) < 0.2:
                            effect_size = "negligible"
                        elif abs(cohen_d) < 0.5:
                            effect_size = "small"
                        elif abs(cohen_d) < 0.8:
                            effect_size = "medium"
                        else:
                            effect_size = "large"
                        
                        # Print the results
                        print(f"Cohen's d: {cohen_d:.4f} ({effect_size} effect)")


                        # Print the results
                        print(f"T-statistic: {t_stat}")
                        print(f"P-value: {p_value:.4e}")
                        print(f"Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")
                        
                        # Check if p-value is below the threshold
                        if p_value < threshold:
                            print("There is statistical significance.")
                        else:
                            print("There is no statistical significance.")
                            
                        print()
                        
        print("#----------------------#")

def p_n():
    modification_tasks = ["add_edge", "remove_edge", "add_node", "remove_node"]
    encoding_type = "adjacency_matrix"
    final_task = "print_graph"
    models = ["claude-3-5-sonnet-20240620_70B"]

    modification_dict = {}
    for model_name in models:
        # Plot the accuracy values for each model on subplots for all modification tasks
        x = 0
        for modification_task in modification_tasks:
            for n in [7, 10, 15, 20]:
                p_dict = {}
                print(f"Modification Task: {modification_task}, n: {n}")
                for p in [0.1, 0.5, 0.9]:
                
                    accuracy_list = []
                    for i in range(1, 6):
                        
                        # get no print accuracy
                        file_path = f"results/p/{model_name}/{final_task}/{modification_task}/{i}/{encoding_type}_big/{p}/{n}/results.txt"
                        try:
                            with open(file_path, "r") as file:
                                text = file.read()
                                #print(f"text: {text}")
                                # Extract the accuracy value using regular expression
                                accuracy = re.search(r"Accuracy: (\d+\.\d+)%", text)
                                #print(f"accuracy: {accuracy}")
                                if accuracy:
                                    accuracy_value = round(float(accuracy.group(1)), 2)
                                    accuracy_list.append(accuracy_value)
                        except FileNotFoundError:
                            #continue
                            pass
                    p_dict[p] = accuracy_list

                plt.figure(figsize=(10.5, 8.5))

                # plot each line from p_dict
                for p, accuracy_list in p_dict.items():
                    if p == 0.1:
                        linestyle = 'solid'
                    elif p == 0.5:
                        linestyle = 'dashed'
                    elif p == 0.9:
                        linestyle = 'dashdot'
                    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label=f"p={p}", linestyle=linestyle, linewidth=2.5)

                # save plot
                plt.xlabel('Number of modifications', fontsize=30)
                plt.ylabel('Accuracy', fontsize=30)
                if modification_task == "add_edge":
                    plt.title(f"Add Edge, Graph Size: {n} nodes", fontsize=30)
                elif modification_task == "remove_edge":
                    plt.title(f"Remove Edge, Graph Size: {n} nodes", fontsize=30)
                elif modification_task == "add_node":
                    plt.title(f"Add Node, Graph Size: {n} nodes", fontsize=30)
                elif modification_task == "remove_node":
                    plt.title(f"Remove Node, Graph Size: {n} nodes", fontsize=30)

                plt.xlim(1, 5)
                plt.ylim(0, 100)
                plt.xticks(range(1, 6), fontsize=19)
                plt.yticks(fontsize=19)
                plt.tick_params(right=False, top=False)
                plt.legend(fontsize=20)
                plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))
                plot_file_dir = f"plots/p/{model_name}/{final_task}/{n}"

                os.makedirs(plot_file_dir, exist_ok=True)
                plot_file_path = f"{plot_file_dir}/{modification_task}.png" 
                plt.savefig(plot_file_path)
                plt.clf()
                
            
    print("#----------------------#")

def main():
    # Parse the arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_name", type=str, choices=["claude", "llama", "o1", "qwen", "gpt", "claude-h", "llama-70", "llama-8"], help="Name of the model")
    #args = parser.parse_args()
    """
    if args.model_name == "claude":
        model_name = "claude-3-5-sonnet-20240620_70B"
    elif args.model_name == "llama":
        model_name = "llama3.1_405B"
    elif args.model_name == "o1":
        model_name = "o1-mini_70B"
    elif args.model_name == "qwen":
        model_name = "qwen2.5_72B"
    elif args.model_name == "gpt":
        model_name = "gpt-4o-mini_70B"
    elif args.model_name == "claude-h":
        model_name = "claude-3-haiku-20240307_70B"
    elif args.model_name == "llama-70":
        model_name = "llama3.1_70B"
    elif args.model_name == "llama-8":
        model_name = "llama3.1_8B"
    """
    # Call the function to process the files
    #compare_models()
    #stat()
    #graph_types()
    p_n()

if __name__ == "__main__":
    main()