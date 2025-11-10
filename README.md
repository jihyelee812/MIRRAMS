# MIRRAMS
The presence of missing values often reflects variations in data collection policies, which may shift across time or locations, even when the underlying feature distribution remains stable. Such shifts in the missingness distribution between training and test inputs pose a significant challenge to achieving robust predictive performance. In this study, we propose a novel deep learning framework designed to address this challenge, particularly in the common yet challenging scenario where the test-time dataset is unseen. We begin by introducing a set of mutual information-based conditions, called MI robustness conditions, which guide the prediction model to extract label-relevant information. This promotes robustness against distributional shifts in missingness at test-time. To enforce these conditions, we design simple yet effective loss terms that collectively define our final objective, called MIRRAMS. Importantly, our method does not rely on any specific missingness assumption such as MCAR, MAR, or MNAR, making it applicable to a broad range of scenarios. Furthermore, it can naturally extend to cases where labels are also missing in training data, by generalizing the framework to a semi-supervised learning setting. Extensive experiments across multiple benchmark tabular datasets demonstrate that MIRRAMS consistently outperforms existing state-of-the-art baselines and maintains stable performance under diverse missingness conditions. Moreover, it achieves superior performance even in fully observed settings, highlighting MIRRAMS as a powerful, off-the-shelf framework for generalpurpose tabular learning.

# Run the Experiments
Set your dataset directory path in `data_open.py`:
```python
def dataset_open(ds_name):
    path = '/home/user/data' # <-- Edit this to your actual data location
```

To run the training script, execute the following commands in your terminal:
```bash
# Run standard training
python ./MIRRAMS/train.py --ds_name adult --epochs 100
# With semi-supervised learning
python ./MIRRAMS/train.py --ds_name adult --epochs 100 --ssl
```


# Code Files Description
| File / Folder  | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| `train.py`     | Main script to train MIRRAMS and evaluate model performance               |
| `datasets.py` | Logic to load raw tabular datasets                   |
| `data_open.py`   | Prepares the dataset and splits it into train, validation, and test sets  |
| `missing.py`   | Implements missingness simulation mechanisms (MCAR, MAR, MNAR)            |
| `utils.py`     | Utility functions for training and evaluation |
| `models/`      | Contains model components |


# Requirements
This project was developed and tested in the following environment:

- **Python 3.10**  
- **PyTorch 2.6.0**  

> 💡 For full package details, please refer to [`requirements.txt`](./requirements.txt).
```bash
pip install -r requirements.txt
```

# Dataset
We use ten publicly available benchmark datasets for experiments.
These datasets are mainly sourced from:

* UCI Machine Learning Repository
* AutoML Benchmark Repository

Supported datasets include: adult, htru2, qsar_bio, and more (see code for full list).


# arXiv
[Click here to try](https://arxiv.org/abs/2507.08280)
