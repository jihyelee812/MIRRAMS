import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def dataset_open(ds_name):
    '''
    The target column should be the last column.
    '''
    print(f'Dataset: {ds_name}')
    # path = '/home/user/data'
    path = '/home/jihye812/data'

    colnames = False
    if ds_name == 'adult':
        data_dir = f'{path}/adult/adult.data'
        dataset = pd.read_table(data_dir,sep=',',header=None)
        categorical_indicator = [False, True, False, True, False, True, True, True, True, True, False, False, False, True] 
        colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']                
        dataset.columns = colnames
        target = colnames[-1]


    elif ds_name == 'bank':
        data_dir = f'{path}/bank/bank-full.csv'
        dataset = pd.read_csv(data_dir, sep=';')
        colnames = ["age", "job", "marital", "education", "default", "balance",
                    "housing", "loan", "contact", "day", "month", "duration",
                    "campaign", "pdays", "previous", "poutcome", "y"]
        categorical_indicator = [False, True, True, True, True, False, True, True, True, False, True, False, False, False, False, True]
        target = colnames[-1]


    elif ds_name == 'cheating':
        data_dir = f'{path}/cheating/response_data_231110.csv'
        dataset = pd.read_csv(data_dir, sep=';')
        categorical_indicator = [False, True, True, True, True, False, True, True, True, False, True, False, False, False, False, True]        
        target = colnames[-1]


    elif ds_name == 'churn':
        data_dir = f'{path}/churn/Churn_Modelling.csv'
        dataset = pd.read_csv(data_dir,sep=',')
        dataset = dataset.drop(columns=['RowNumber','CustomerId'])
        colnames = dataset.columns.tolist()
        target = colnames[-1]
        categorical_indicator = [True, False, True, True, False, False, False, False, True, True, False]


    elif ds_name == 'covertype':
        # data_dir = f'{path}/covertype/covtype.data.gz'
        # dataset = pd.read_csv(data_dir,sep=',',header=None)
        # # categorical_indicator = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, 
        # #     True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
        # #     True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        # unique_counts = pd.DataFrame(dataset.nunique()).reset_index()
        # unique_counts.columns = ['column', 'unique_count']
        # category_threshold = 7
        # categorical_indicator = unique_counts['unique_count'] <= category_threshold
        # categorical_indicator = categorical_indicator.tolist()
        # categorical_indicator = categorical_indicator[:-1] # with out label
        # colnames = dataset.columns.tolist()
        # target = colnames[-1]
        data_dir = f'{path}/covertype/covtype_le.csv'
        dataset = pd.read_csv(data_dir)
        categorical_indicator = [False] * (dataset.shape[1] - 1)
        target = 'Cover_Type'
        colnames = dataset.columns.tolist()
        # task='multiclass'


    elif ds_name == 'dota2':
        data_dir = f'{path}/dota2/dota2Train.csv'
        dataset = pd.read_csv(data_dir, sep=',', header=None)
        categorical_indicator = [True] * 116
        colnames = dataset.columns.tolist()
        # The target is not at the last position
        dataset = dataset[dataset.columns[1:].tolist() + [dataset.columns[0]]]
        target = colnames[-1]


    elif ds_name == 'htru2':
        data_dir = f'{path}/htru2/HTRU_2.csv'
        dataset = pd.read_csv(data_dir, header=None)
        categorical_indicator = [False, False, False, False, False, False, False, False]
        colnames = dataset.columns.tolist()
        target = colnames[-1]


    elif ds_name == 'qsar_bio':
        data_dir = f'{path}/qsar_bio/biodeg.csv'
        dataset = pd.read_csv(data_dir,sep=';', header=None)
        categorical_indicator = [False] * 41
        colnames = dataset.columns.tolist()
        target = colnames[-1]


    elif ds_name == 'shoppers':
        data_dir = f'{path}/shoppers/online_shoppers_intention.csv'
        dataset = pd.read_csv(data_dir)
        colnames = dataset.columns.tolist()
        categorical_indicator = [False] * 10 + [True] * 7
        colnames = dataset.columns.tolist()
        target = colnames[-1]

    elif ds_name == 'mnist':
        from torchvision import datasets
        from torchvision.transforms import ToTensor

        dataset_vision = datasets.MNIST('/home/jihye812/data', train=True, download=True)

        X = dataset_vision.data.numpy()
        n, a, b = dataset_vision.data.shape
        X = X.reshape(-1, a*b)
        y = dataset_vision.targets.numpy().reshape(-1, 1)
        dataset = pd.DataFrame(np.concatenate([X, y], axis=1))

        colnames = dataset.columns.tolist()
        categorical_indicator = [False] * (dataset.shape[1] - 1)
        target = colnames[-1]


    # SAINT Dataset
    elif ds_name == 'arcene':
        dataset = pd.read_csv(f'{path}/arcene/arcene.csv')
        colnames = dataset.columns.tolist()
        categorical_indicator = [False] * (dataset.shape[1] - 1)
        target = colnames[-1]


    elif ds_name == 'arrhythmia':
        # import scipy.io
        # matdata = scipy.io.loadmat('./arrhythmia/arrhythmia.mat')
        # train = matdata['X']
        # train  = np.append(train,matdata['y'],axis=1)
        # dataset = pd.DataFrame(train)
        # dataset = dataset.loc[:, dataset.std() > 0.08]
        # dataset.drop(columns=[18], axis=1, inplace=True)
        # dataset.to_csv('./arrhythmia.csv',sep=',',index = False)
        data_dir = f'{path}/arrhythmia/arrhythmia.csv'
        dataset = pd.read_csv(data_dir,header=None,skiprows=1)
        colnames = dataset.columns.tolist()
        categorical_indicator = [False] * (dataset.shape[1] - 1)
        target = colnames[-1]


    elif ds_name == 'blastchar':
        dataset = pd.read_csv(f'{path}/blastchar/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        dataset.drop(columns=['customerID'], axis=1, inplace=True) # drop ID column
        dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
        colnames = dataset.columns.tolist()
        categorical_indicator = [True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, False]
        target = 'Churn'

        
    elif ds_name == 'spambase':
        dataset = pd.read_csv(f'{path}/spambase/spambase.data', header=None)
        colnames = dataset.columns.tolist()
        categorical_indicator = [False] * (dataset.shape[1] - 1)
        target = dataset.shape[1]


    else:
        raise ValueError("Unsupported data.\n")
    print("File load")

    return dataset, categorical_indicator, colnames, target