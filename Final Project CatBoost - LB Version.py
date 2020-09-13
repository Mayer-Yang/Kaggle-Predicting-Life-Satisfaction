#### Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


#### import datasets
df_train_raw = pd.read_csv('../input/eurosat-w2020/train_raw.csv')
df_train = pd.read_csv('../input/eurosat-w2020/train.csv')
df_test_raw = pd.read_csv('../input/eurosat-w2020/test_raw.csv')
df_test = pd.read_csv('../input/eurosat-w2020/test.csv')
df_ss = pd.read_csv('../input/eurosat-w2020/sample_submission.csv')

#### EDA

# dtype dictionary
col_types = {}
for cname in df_train.columns:
    if df_train[cname].dtype not in col_types:
        col_types[df_train[cname].dtype] = 1
    else:
        col_types[df_train[cname].dtype] += 1
for tp in col_types:
    print(tp, col_types[tp])
    
# Categorical cname list
cat_cols = [cname for cname in df_train.columns if df_train[cname].dtype == "object"]

num_cat_col = len(cat_cols)

# Categorical cname list with unique values <= 10
cat_cols10 = [cname for cname in df_train.columns 
            if df_train[cname].nunique() <= 10 
            and df_train[cname].dtype == "object"]

num_cat_col10 = len(cat_cols10)

# Categorical cname list with unique values > 10
cat_cols10_plus = [cname for cname in df_train.columns 
            if df_train[cname].nunique() > 10 
            and df_train[cname].dtype == "object"]

num_cat_col10_plus = len(cat_cols10_plus)

# Numeric cname list
num_cols = [cname for cname in df_train.columns if 
                df_train[cname].dtype in ['int64', 'float64']]

num_num_col = len(num_cols)

#########################################################################
# <= 10 object: 134 /// > 10 object: 93 /// num: 46 /// total: 273
#########################################################################

## Missing data
num_tot = df_train.isnull().sum().sort_values(ascending=False)
perc = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing = pd.concat([num_tot, perc], axis=1, keys=['Total', 'Percent'])
missing_real = missing[missing['Total'] > 0]
missing_real


# list of cnames to drop which has obs <= 1000 
drop_list = ['v207','v195','v262','v89','v206','v194','v88','v261','v205','v193','v87','v260','v204','v86','v192','v259','v270','v203','v215','v97','v202','v96','v269','v214','v95','v213','v201','v268']
       
# rest cols needed to impute
rest_cols = []
for cname in missing_real.index:
    if cname not in drop_list:
        rest_cols.append(cname)

        
# drop cnames
drop_list.append('v17')
drop_list.append('v25')
drop_list.append('v78')
df_train = df_train.drop(drop_list,1)
df_test = df_test.drop(drop_list, 1)

# Impute missing values
def miss_imputer(col_list, special_value = ['.a', '.b', '.c'], strategy = 'median'):
    for temp in col_list:
        for i in special_value:
            df_train[temp] = df_train[temp].replace(i, np.nan)
            df_test[temp] = df_test[temp].replace(i, np.nan)
        imp = SimpleImputer(strategy= strategy)
        #df_train[temp] = imp.fit_transform(df_train[[temp]])
        #df_test[temp] = imp.fit_transform(df_test[[temp]])
        total = pd.concat([df_train[[temp]], df_test[[temp]]])
        imp.fit(total)
        df_train[temp] = imp.transform(df_train[[temp]])
        df_test[temp] = imp.transform(df_test[[temp]])

# rest_cols split
most_frequent_list = ['v123', 'v132']        
a_list = ['v126', 'v127', 'v131', 'v130']
abc_list = ['v256', 'v104', 'v67', 'v61', 'v60', 'v66', 'v59', 'v58', 'v112', 'v115', 'v114', 'v117', 'v116', 'v118', 'v149', 'v148', 'v146', 'v145', 'v144', 'v143', 'v142', 'v141', 'v140', 'v139', 'v138', 'v137', 'v147', 'v136', 'v113']
abcd_list = ['v68', 'v63', 'v62']

miss_imputer(most_frequent_list, special_value = [], strategy = 'most_frequent')
miss_imputer(a_list, special_value = ['.a'])
miss_imputer(abc_list)
miss_imputer(abcd_list, special_value = ['.a', '.b', '.c', '.d'])

# double check there is no more missing values
df_train.isnull().sum()

###################################################################################

###################################################################################

# Call cols dtype code chunck again
# Categorical cname list
cat_cols = [cname for cname in df_train.columns if df_train[cname].dtype == "object"]

num_cat_col = len(cat_cols)

# Categorical cname list with unique values <= 10
cat_cols10 = [cname for cname in df_train.columns 
            if df_train[cname].nunique() <= 10 
            and df_train[cname].dtype == "object"]

num_cat_col10 = len(cat_cols10)

# Categorical cname list with unique values > 10
cat_cols10_plus = [cname for cname in df_train.columns 
            if df_train[cname].nunique() > 10 
            and df_train[cname].dtype == "object"]

num_cat_col10_plus = len(cat_cols10_plus)

# Numeric cname list
num_cols = [cname for cname in df_train.columns if 
                df_train[cname].dtype in ['int64', 'float64']]

num_num_col = len(num_cols)


# LabelEncoder
def lencoder(col_list):
    for temp in col_list:
        le = LabelEncoder()
        total = pd.concat([df_train[temp], df_test[temp]])
        le.fit(total)
        df_train[temp] = le.transform(df_train[[temp]])
        df_test[temp] = le.transform(df_test[[temp]])
        
le_list = ['v20', 'v154', 'v155', 'v161', 'cntry']
lencoder(le_list)


# specially created for some testing cols
def miss_imputer_test(col_list, special_value = ['.a', '.b', '.c'], strategy = 'median'):
    for temp in col_list:
        for i in special_value:
            df_test[temp] = df_test[temp].replace(i, np.nan)
        imp = SimpleImputer(strategy= strategy)
        df_test[temp] = imp.fit_transform(df_test[[temp]])
miss_imputer_test(['v239'], special_value = ['.a', '.b', '.c', '.d'])

def cols10_processor(col_list):
    for temp in col_list:
        special_list = []
        for val in df_train[temp].unique():
            if str(val) == '.a':
                special_list.append('.a')
            elif str(val) == '.b':
                special_list.append('.b')
            elif str(val) == '.c':
                special_list.append('.c')
            elif str(val) == '.d':
                special_list.append('.d')
        #print(temp)
        miss_imputer([temp], special_value = special_list, strategy = 'median')

cols10_processor(cat_cols10)
cols10_processor(cat_cols10_plus)


## Feature Engineering
# Original 0.9404 /// 0.8926
# v98*v224 0.9442 /// 0.8929
# drop original v98 & v224 0.9529 /// 0.8747
# v98*v224 with learning rate = 0.05  0.9415 /// 0.8932
# avg(v98, v224, v101) 0.9472 /// 0.8935
#pairplot
#df_train['v98'] = df_train['v98'] / df_train['v98'].std()
sns.set()
cols = ['v224', 'v101']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

# create new col to take average of v98, v224 and v101
df_train['avg98and224and101'] = (df_train['v98'] + df_train['v224'] + df_train['v101'])
df_test['avg98and224and101'] = (df_test['v98'] + df_test['v224'] + df_test['v101'])


###################################################################################
###################################################################################
###################################################################################

#### Model

from catboost import CatBoostClassifier, Pool, cv

from sklearn.model_selection import KFold
import gc
from sklearn.metrics import roc_auc_score
def cat_model(df_train, n_folds = 5):
    
    # drop id and satisfied
    df_train_id = df_train['id']
    df_train_label = df_train['satisfied'].values
    df_train = df_train.drop(['id', 'satisfied'], 1)

    # feature importance & out of fold empty array
    fi = np.zeros(len(df_train.columns))
    oof = np.zeros(df_train.shape[0])

    # valid & train score empty list
    vs = []
    ts = []
    
    pred = np.zeros(df_test.shape[0])

    # define kfold and loop through each iteration
    kf = KFold(n_splits = n_folds, shuffle = True, random_state = 1)
    for ti, vi in kf.split(df_train):
        # split train and valid dataframe
        train_features, train_labels = df_train.iloc[ti], df_train_label[ti]
        valid_features, valid_labels = df_train.iloc[vi], df_train_label[vi]

        # fit lightgbm
        # 0.894842 // better than 0.894822 without using eval_set
        # 0.895669 // with cat_features
        cat_params = {
                'iterations': 2000,
                'early_stopping_rounds': 500,
                'depth': 10,
                'learning_rate': 0.01,
                'loss_function': 'Logloss',
                'verbose': False,
                'eval_metric': 'AUC'
        }
        
        model = CatBoostClassifier(**cat_params)
        model.fit(train_features, train_labels, eval_set=(valid_features, valid_labels), use_best_model=True)

        pred += model.predict_proba(df_test.drop(['id'], 1))[:, 1] / kf.n_splits
        #fi += model.get_feature_importance / kf.n_splits
        fi = pd.DataFrame(list(zip(train_features.dtypes.index, model.get_feature_importance(Pool(train_features, train_labels, 
                                                            cat_features=None)))),
                          columns=['Feature','Score'])
        oof[vi] = model.predict_proba(valid_features)[:, 1]
        #vs.append(model.best_score_['valid']['auc'])
        #ts.append(model.best_score_['train']['auc'])

        # Call garbage collector
        gc.enable()
        del model, train_features, valid_features, train_labels, valid_labels
        gc.collect()


    # feature importance
    col_names = list(df_train.columns)
    #feature_importances = pd.DataFrame({'feature': col_names, 'importance': fi})

    # metrics
    vs.append(roc_auc_score(df_train_label, oof))
    #ts.append(np.mean(ts))
    metrics = pd.DataFrame({'valid': vs}) 
    
    submission = pd.DataFrame({'Id': df_test['id'], 'Predicted': pred})
    
    return fi, metrics, submission

fi, metrics, submission = cat_model(df_train)

'''
fi['importance'] = fi['importance']/fi['importance'].sum()
fi = fi.sort_values(by='importance', ascending = False).reset_index(drop=True)
fi
'''

#submission.to_csv('/Users/yangzibin/Desktop/STAT 441 Final Project/eurosat-w2020/submission.csv', index = False)




















