# DR-MMSearchAgent
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
conda create -n verl python==3.10.12
conda activate verl
Install DependenciesExecute the install.sh script located in the Train directory.Note: Flash-attention must be installed manually.Bash# Download and install Flash Attention
wget -nv [https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
2. Data PreparationTraining requires specific parquet files.⚠️ Important: The training set is a mixture of internal data and FVQA. The images column format has changed and is no longer a list.Training Set: /inspire/hdd/project/public/datasets.parquetValidation Set: fvqa_test.parquet (Download: HuggingFace FVQA)Search Cache: Current image-to-image search uses a local cache. (Download: HuggingFace Cache)3. Deploy Local Search ServiceRefer to the setup instructions in Search-R1.Startup Script: /inspire/hdd/project/continuinglearningtheory/public/miror/search/re.shResources: E5 model and wiki25 database/index are located at /inspire/hdd/project/continuinglearningtheory/public/wiki25.4. Start TrainingUse the provided script to launch the training job:Bashcd Train
bash run_mmsearch_grpo.sh
Environment Variables:Ensure the following variables are set correctly in the script before running:WANDB_API_KEY: (Optional) WandB API Key.SAVE_CHECKPOINT_DIR: Directory to save model checkpoints.DATASET_TRAIN: Path to the training dataset.DATASET_VAL: Path to the validation dataset.REF_MODEL_PATH: Path to the reference model

## We are actively curating the remaining assets of this project and working through the necessary licensing approvals.




# Citation
## NOTE：Our work and acceptance by ICML 2026
Please cite this work if you find it useful:
