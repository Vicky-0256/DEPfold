# DEPfold: RNA Secondary Structure Prediction as Dependency Parsing

A deep learning framework that treats RNA secondary structure prediction as a dependency parsing problem.

## Overview

DEPfold is a novel approach to RNA secondary structure prediction that leverages techniques from natural language processing, specifically dependency parsing with biaffine attention. The model can effectively predict both canonical base pairs and pseudoknots, achieving competitive performance on standard RNA structure benchmarks.

## Installation

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DEPfold.git
cd DEPfold
```

2. Create and activate the environment using the provided YAML file:
```bash
conda env create -f environment.yaml
conda activate depfold
```

## Usage

### Training

To train the model:

```bash
python run_parser.py --mode train \
                    --train_path /path/to/training/data \
                    --eval_path /path/to/validation/data \
                    --test_path /path/to/test/data \
                    --embedding roberta-base \
                    --output_dir ./output
```

### Prediction

To make predictions with a trained model:

```bash
python run_parser.py --mode predict \
                    --predict /path/to/predict/data \
                    --predict_save /path/to/save/results \
                    --path /path/to/model.pt
```

## Key Features

- **Dependency Parsing Framework**: Models RNA secondary structure prediction as a dependency parsing problem
- **Multiple Embedding Options**: Supports RNA-FM and RoBERTa embeddings
- **Pseudoknot Support**: Can predict complex RNA structures including pseudoknots
- **Contact Map Visualization**: Generates detailed contact maps for structural analysis
- **Tree-Constraint Decoding**: Can enforce tree constraints during structure prediction

## Command Line Options

### General Options
- `--seed`: Random seed (default: 66)
- `--mode`: 'train' or 'predict' (default: 'train')
- `--output_dir`: Directory to save model and results (default: './output')
- `--cache_data`: Path to cache processed data (default: './data/bp_')

### Model Options
- `--embedding`: Embedding type ('one-hot', 'RNA-fm', 'roberta-base')
- `--finetune`: Whether to finetune the embedding model
- `--tree`: Use tree constraints for decoding
- `--proj`: Use projectivity constraints for decoding
- `--loss`: Loss function ('cross_entropy', 'focal_loss')
- `--is_pse`: Enable pseudoknot prediction

### Training Options
- `--train_path`, `--eval_path`, `--test_path`: Paths to datasets
- `--per_gpu_train_batch_size`: Training batch size (default: 3)
- `--num_train_epochs`: Number of training epochs (default: 100)
- `--early_stop`: Patience for early stopping (default: 8)
- `--lr`: Learning rate (default: 5e-5)

### Prediction Options
- `--predict`: Path to data for prediction
- `--predict_save`: Directory to save prediction results
- `--path`: Path to trained model for prediction
- `--beta`: Beta coefficient for stem map scores (default: 0.0)

## Output Files

- Contact maps (`.txt` files): RNA structure contact maps
- Dot-bracket notation (in `predict.txt`): Standard RNA structure representation
- Sequence information (in `seq.txt`): Original RNA sequences

Note: RNA-FM embeddings require additional setup. Follow the instructions in the RNA-FM repository and place the pretrained model in the `pretrained/RNA-FM_pretrained.pth` path.


## Citation
```bibtex
@inproceedings{wangdepfold,
  title={DEPfold: RNA Secondary Structure Prediction as Dependency Parsing},
  author={WANG, KE and Cohen, Shay B.},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
