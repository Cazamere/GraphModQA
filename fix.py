import os


score = 0
sizes = ['small', 'medium', 'large']
models = ["gpt-4o-mini", 'o1-mini', 'claude-3-5-sonnet-20240620', 'llama3.1', 'o3-mini', 'claude-3-7-sonnet-20250219']
tasks = ["node_count", "edge_count", "node_degree", "connected_nodes", "print_graph", "isolated_nodes", "triangle_count", "overlapped_nodes", "overlapped_edges"]

for model in models:
    for task in tasks:
        for size in sizes:

            filename = f'coauth/{size}/{task}/{model}/results.txt'
            filepath = os.path.join('results', filename)
            
            if os.path.isfile(filepath):
                # Do something with the file
                with open(filepath, 'r') as file:
                    contents = file.read()
                    print(f"model: {model}, task: {task}, size: {size}")
                    print(contents)
                    print()
        print('---------------------')
    print('---------------------------------')
    