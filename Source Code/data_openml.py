import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import sys

### To fit in user dataset, use id=1 ###
### 1. Change X, y data file root
### 2. Change col name and category and continuous col type manually
### 3. Change Raha detection result file root
### 4. Change Raha dirty shreshold manually
### 5. Change y_dim (classification class number) in train.py or train_robust.py file
### 6. Change blank value filling strategy by switching Model 1/2/3/4 (optional)


def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    np.random.seed(seed)
    if ds_id == 1:
        ## Hospital
        # csv_X_data = pd.read_csv('./rahaDatasetHospital/dirty.csv', header=0, encoding='gb18030')
        # X = pd.DataFrame(csv_X_data)
        # csv_y_data = pd.read_csv('./rahaDatasetHospital/y.csv', header=0, encoding='gb18030')
        # y = csv_y_data['measure_code']
        # categorical_indicator = [True, True, True, True, True, True, True, True, True, True, True, True, True, True,
        #                          True, True, False, True, True]  # hospital
        # attribute_names = ["index", "provider_number", "name", "address_1", "address_2", "address_3", "city", "state",
        #                    "zip", "county", "phone", "type", "owner", "emergency_service", "condition", "measure_name",
        #                    "score", "sample", "state_average"]
        # csv_raha_detect = pd.read_csv("./rahaDatasetHospital/raha_final.csv", header=0, encoding='gb18030')

        ## Flight
        csv_X_data = pd.read_csv('./rahaDatasetFlight/dirty.csv', header=0, encoding='gb18030')
        X = pd.DataFrame(csv_X_data)
        csv_y_data = pd.read_csv('./rahaDatasetFlight/y.csv', header=0, encoding='utf-8')
        y = csv_y_data['flight']
        categorical_indicator = [True, True, True, True, True, True]                ## flight
        attribute_names = ["tuple_id", "src", "sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]
        csv_raha_detect = pd.read_csv("./rahaDatasetFlight/raha_final.csv", header=0, encoding='gb18030')

        ## Beer
        # categorical_indicator = [True, True, True, False, False, True, True, True, True]            ## beer

        raha_detect = []
        for row in range(csv_raha_detect.shape[0]):
            dirty = 0
            for col in range(csv_raha_detect.shape[1]):
                if csv_raha_detect.iat[row, col] == 1:
                    dirty = dirty + 1
            raha_detect.append(dirty)
        for row in range(len(raha_detect)):
            if raha_detect[row] >= 2:
                raha_detect[row] = 1
            else:
                raha_detect[row] = 0

    else:
        dataset = openml.datasets.get_dataset(ds_id)

        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    ## Change Input Workflow here ##

    # X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    # saveDataset = input("Do you want to save original dataset? ([y] for yes, others for no) ")
    # if saveDataset == 'y':
    #     print("Saving original dataset...")
    #     X.to_csv("./dataset{}/X.csv".format(ds_id))
    #     y.to_csv("./dataset{}/y.csv".format(ds_id))
    #     with open("./dataset{}/categorical_indicator.txt".format(ds_id), 'w') as f:
    #         print(categorical_indicator, file=f)
    #     with open("./dataset{}/attribute_names.txt".format(ds_id), 'w') as f:
    #         print(attribute_names, file=f)
    #     print("Original dataset saved.")
    #     sys.exit()

    if ds_id == 42178:
        categorical_indicator = [True, False, True,True,False,True,True,True,True,True,True,True,True,True,True,True,True,False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp ]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728,42705,42729,42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    raha_detect_train = []

    ## LQ: modify to split raha detect
    dataSplitResult = X["Set"].values.tolist()
    for row in range(len(dataSplitResult)):
        if dataSplitResult[row] == "train":
            raha_detect_train.append(raha_detect[row])

    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")

        # ## model 1 ##
        # if type(X[col][0]) == float:
        #     X[col] = X[col].fillna(0.0)
        # else:
        X[col] = X[col].fillna("MissingValue")

        ## model 2 ##
        # ColList = X[col].tolist()
        # FreqDict = {}               # record frequency dict
        # for idx in range(len(ColList)):
        #     if ColList[idx] in FreqDict.keys():
        #         FreqDict[ColList[idx]] = FreqDict[ColList[idx]] + 1
        #     else:
        #         FreqDict[ColList[idx]] = 1
        # FreqDict = sorted(FreqDict.items(), key=lambda x: x[1], reverse=True)
        # MaxFreq = FreqDict[0][0]
        # X[col] = X[col].fillna(MaxFreq)

        ## model Raha + SAINT ##


        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        ## model 3 ##
        X[col].fillna(0.0,inplace=True)
        for row in range(len(X[col])):
            if isinstance(X[col][row], str):
                X[col][row] = float(0.0)

        ## model 4 ##
        # X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    ## LQ: raha detection matrix is added!!!!!!!!!!!!!!!!

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, raha_detect_train




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

