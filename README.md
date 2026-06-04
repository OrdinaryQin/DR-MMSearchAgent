# DR-MMSearchAgent
## 📥 Models & Datasets 
"Almost all our code, models, and real datasets are now open-source. Feel free to contact us with any questions or bug reports 🐛 (and please bear with us if we reply slowly due to our busy, multi-threaded schedules 😵‍💫🔥)."

* **🤖  (Model):** [Shengqina/DR-MMSearchAgent](https://huggingface.co/Shengqina/DR-MMSearchAgent)
* **📊  (Dataset):** [Shengqina/BridgeVQA](https://huggingface.co/datasets/Shengqina/BridgeVQA)

DR-MMSearchAgent: Deepening Reasoning in Multimodal Search Agents


📌 Introduction
DR-MMSearchAgent is a novel framework designed to enhance the multi-step reasoning capabilities of multimodal search agents. By moving beyond simple correctness, our approach focuses on the depth and quality of the reasoning trajectory.

The framework leverages structural proximity to derive advantage signals from entire batch rollout trajectories. This encourages the model to generate diverse reasoning paths of varying lengths, even when they lead to the same correct answer. Furthermore, we introduce Differentiated Gaussian Rewards to dynamically calibrate interaction tolerance, effectively balancing information reliability with the need to reduce redundancy in multimodal search.


# MMSearch & VERL Integration

## 📖 Core Components

### 1. Multi-modal Search (MMSearch)
MMSearch is the core module of the project, implementing the following functionalities:
- **Custom Dataset Processing:** `CustomRLHFDataset`
- **Reward Scoring:** `compute_score` calculation logic.
- **Multi-modal Preprocessing:** Support for interleaved image and text data.
- **Tool Integration:** Management of tool invocation.

### 2. VERL Reinforcement Learning Framework
VERL is a general-purpose Reinforcement Learning framework providing:
- **PPO Trainer:** Robust implementation.
- **Distributed Training:** Scalable training support.
- **Parallelism Strategies:** Data Parallelism, Model Parallelism, etc.
- **Agent Loop:** The core multi-turn interaction logic is located in `Train/verl/experimental/agent_loop`.

### 3. Supported Tools
The project integrates external tools to enhance model capabilities:
- `web_text_search`: Network text search.
- `web_image_to_image_search`: Visual similarity search.

---

## 🚀 Quick Start

### 1. Environment Setup

**Create Conda Environment**
```bash
1.conda create -n verl python==3.10.12
conda activate verl
Install DependenciesExecute the install.sh script located in the Train directory.Note: Flash-attention must be installed manually.Bash# Download and install Flash Attention
wget -nv [https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
requirements.txt
```
### 2. Data Preparation

**Data path**
```bash
DATASET_TRAIN consists of Fvqa_train and our BridgeVQA.
DATASE_VAL are composed of Fvqa_test
```
**Local service**
```bash
text searh: https://github.com/petergriffinjin/search-r1
image search: https://github.com/EvolvingLMMs-Lab/multimodal-search-r1
```

### 3. Training with local search
**Command**
```bash
bash run_mmsearch_grpo.sh
```
### 4. Eval
**Command**
```bash
bash run_fvqa_test.sh
```
### 4. Notes
**Detail**
```bash
1.Data PreparationTraining requires specific parquet files.
⚠️ Important: The training set is a mixture of internal data and FVQA. The images column format has changed and is no longer a list.
2.Training Set: /inspire/hdd/project/public/datasets.parquet Validation Set: fvqa_test.parquet (Download: HuggingFace FVQA) Search Cache: Current image-to-image search uses a local cache. (Download: HuggingFace Cache)
3. Deploy Local Search ServiceRefer to the setup instructions in Search-R1. Resources: E5 model and wiki25 database/index.
4. Start TrainingUse the provided script to launch the training job:
5.Environment Variables:Ensure the following variables are set correctly in the script before running: WANDB_API_KEY: (Optional) WandB API Key. SAVE_CHECKPOINT_DIR: Directory to save model checkpoints. DATASET_TRAIN: Path to the training dataset. DATASET_VAL: Path to the validation dataset. REF_MODEL_PATH: Path to the reference model.
```

### 5. More details
**Key Parts**
```bash
Tool list: mm_search_tool_config.yaml
Prompt and reward: mmsearch.py
Data and experimental parameter configuration：run_mmsearch_grpo.sh
```

**Updating......**


# Acknowledgments
**We would like to make improvements based on the following. Thank you very much**
* [MMSearch-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
* [Verl](https://github.com/verl-project/verl)
* [Search-R1](https://github.com/petergriffinjin/search-r1)
* [SenseNova-MARS](https://github.com/OpenSenseNova/SenseNova-MARS)

  
# Citation

Please cite this work if you find it useful.
