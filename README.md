# DyGraphQA: Evaluating LLMs on Real-World Graph Reasoning

This repository contains code for generating and evaluating real-world graph reasoning tasks using large language models (LLMs), as described in our research paper.

---

## 🔧 Setup

1. **Download the dataset**

   Download the real-world coauthorship graph from the DBLP project:

   🔗 https://projects.csail.mit.edu/dnd/DBLP/

   Download `dblp_coauthorship.json.gz` and place it in the root directory of this repo.

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Generate Graphs

Run the following script to generate graphs from the real-world coauthorship network:

```bash
python3 graph_generator.py --n_graphs 250 --experimentType "real"
```

> 🔢 Use any value for `--n_graphs`, but we use `250` in our paper.

---

## 🤖 Evaluate Models on Graph Tasks

Once graphs are generated, evaluate an LLM on a specific graph reasoning task:

```bash
python3 real_model_eval.py \
  --model $MODEL \
  --size 405B \
  --graph_size "large" \
  --task "triangle_count"
```

### ✅ Valid options:

- **`--task`**  
  - `node_count`  
  - `edge_count`  
  - `node_degree`  
  - `connected_nodes`  
  - `print_graph`  
  - `isolated_nodes`  
  - `triangle_count`  
  - `overlapped_nodes`  
  - `overlapped_edges`

- **`--graph_size`**  
  - `small`  
  - `medium`  
  - `large`

- **`--model`**  
  - `gpt-4o-mini`  
  - `o1-mini`  
  - `o3-mini`  
  - `claude-3-7-sonnet-20250219`  
  - `llama3.1`

---

## 📄 Citation

If you use this code in your work, please cite our paper (coming soon).
