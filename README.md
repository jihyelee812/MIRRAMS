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