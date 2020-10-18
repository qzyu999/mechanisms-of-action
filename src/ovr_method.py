# ovr_method.py
import os
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import config
import copy

def run(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the target and and convert to numpy
    x_train = df_train.drop(['target', 'kfold'], axis=1).values
    y_train = df_train.target.values

    # Repeat for validation data
    x_valid = df_valid.drop(['target', 'kfold'], axis=1).values
    y_valid = df_valid.target.values

    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    x_train[:,7:] = sc.fit_transform(x_train[:,7:])
    x_valid[:,7:] = sc.transform(x_valid[:,7:])

    # Intiialize the classifier
    model = linear_model.LogisticRegression(random_state=0, max_iter=1e10)

    # Fit the model
    model.fit(x_train, y_train)

    # Create predictions
    y_pred_probs = model.predict_proba(x_valid)
    y_preds = model.predict(x_valid)

    # Calculate and print the accuracy
    log_loss_score = metrics.log_loss(y_valid, y_pred_probs,
        labels=model.classes_)
    auc = metrics.roc_auc_score(y_valid, y_preds,
        labels=model.classes_)
    accuracy = metrics.accuracy_score(y_valid, y_preds)
    
    print(f"Fold={fold}, Log-Loss={log_loss_score}, AUC={auc}, Accuracy={accuracy}")

if __name__ == "__main__":
    ### Create OVR target vectors
    y = pd.read_csv(config.TARGETS_FILE)
    class_counts = y.iloc[:,1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)
    class_counts_sub = class_counts.head(12)
    retained_classes = class_counts_sub.index.values
    y2 = y.iloc[:,1:]

    ### The following creates 'c' binary target vectors saved in a list: 'binary_vector_list'
    class_index_list = [] # Save indices that contain the class
    for c in retained_classes:
        c_indices = y2.loc[:,c][y2.loc[:,c] == 1].index.values
        class_index_list.append([c, c_indices])

    binary_vector_list = []
    n = len(y)
    for i in class_index_list: # Loop through class/index pairs
        zeros = [0] * n
        for j in range(n): # Loop through all rows
            # Check if the index should be one instead
            if j in i[1]:
                zeros[j] = 1
        binary_vector_list.append(pd.DataFrame({i[0]: zeros}))

    ### Loop through OVR classes
    for i in binary_vector_list:
        y_temp = copy.deepcopy(i)
        class_name = y_temp.columns[0]
        X = pd.read_csv(config.FEATURES_FILE)
        X.drop(X.columns[0], axis=1, inplace=True)
        
        # Save the column names
        X_col_names = X.columns.tolist()

        cat_cols = ['cp_type', 'cp_time', 'cp_dose'] # Identify categorical columns

        ohe = OneHotEncoder() # Load OHE

        # Get the column names after OHE
        # Reference: https://stackoverflow.com/questions/54570947/feature-names-from-onehotencoder
        _ = ohe.fit_transform(X[cat_cols])
        ohe_names = ohe.get_feature_names(cat_cols)
        ohe_names = ohe_names.tolist()

        # Fix new column names to include OHE names and normal feature names
        X_col_names = [col for col in X_col_names if col\
            not in cat_cols]
        ohe_names.extend(X_col_names)

        # Transform the data with OHE on the indices of the cat variables
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), list(range(0,3)))],
            remainder='passthrough')
        X = pd.DataFrame(ct.fit_transform(X))
        X.columns = ohe_names
        
        y_temp.columns = ['target']
        
        df = pd.concat([X, y_temp], axis=1) # Recombine into single df

        df['kfold'] = -1 # Create k-folds column

        df = df.sample(frac=1).reset_index(drop=True) # Randomize the dataset

        y = df.target.values # Subset the target column

        # Initialize the stratified k-fold module from sklearn
        kf = StratifiedKFold(n_splits=5)

        # Fill the 'kfold' column with the assigned folds
        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'kfold'] = f
            
        print(class_name)
        for fold_ in range(5):
            run(fold_)

# def run(fold):
#     # Read the data
#     df = pd.read_csv(config.TRAINING_FILE)

#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     df_valid = df[df.kfold == fold].reset_index(drop=True)

#     # Drop the target and and convert to numpy
#     x_train = df_train.drop(['target', 'kfold'], axis=1).values
#     y_train = df_train.target.values

#     # Repeat for validation data
#     x_valid = df_valid.drop(['target', 'kfold'], axis=1).values
#     y_valid = df_valid.target.values

#     # Apply feature scaling to the numeric attributes
#     sc = StandardScaler()
#     x_train[:,7:] = sc.fit_transform(x_train[:,7:])
#     x_valid[:,7:] = sc.transform(x_valid[:,7:])

#     # Intiialize the classifier
#     model = linear_model.LogisticRegression(random_state=0, max_iter=1e10)

#     # Fit the model
#     model.fit(x_train, y_train)

#     # Create predictions
#     y_pred_probs = model.predict_proba(x_valid)
#     y_preds = model.predict(x_valid)

#     # Calculate and print the accuracy
#     log_loss_score = metrics.log_loss(y_valid, y_pred_probs,
#         labels=model.classes_)
#     auc = metrics.roc_auc_score(y_valid, y_preds,
#         labels=model.classes_)
#     accuracy = metrics.accuracy_score(y_valid, y_preds)
    
#     print(f"Fold={fold}, Log-Loss={log_loss_score}, AUC={auc}, Accuracy={accuracy}")

# if __name__ == "__main__":
#     # Create the binary OVR target vectors
#     y = pd.read_csv('../input/train_targets_scored.csv')
#     class_counts = y.iloc[:,1:].sum(axis=0)
#     class_counts = class_counts.sort_values(ascending=False)
#     class_counts_sub = class_counts.head(12)
#     retained_classes = class_counts_sub.index.values
#     y2 = y.iloc[:,1:]

#     ### The following creates 'c' binary target vectors saved in a list: 'binary_vector_list'
#     class_index_list = [] # Save indices that contain the class
#     for c in retained_classes:
#         c_indices = y2.loc[:,c][y2.loc[:,c] == 1].index.values
#         class_index_list.append([c, c_indices])

#     binary_vector_list = []
#     n = len(y)
#     for i in class_index_list: # Loop through class/index pairs
#         zeros = [0] * n
#         for j in range(n): # Loop through all rows
#             # Check if the index should be one instead
#             if j in i[1]:
#                 zeros[j] = 1
#         binary_vector_list.append(pd.DataFrame({i[0]: zeros}))

#     ### Test the OVR method on each of the OVR classes
#     for i in binary_vector_list:
#         y_temp = copy.deepcopy(i)
#         class_name = y_temp.columns[0]
#         X = pd.read_csv('../input/train_features.csv')
#         X.drop(X.columns[0], axis=1, inplace=True)
        
#         # Save the column names
#         X_col_names = X.columns.tolist()

#         cat_cols = ['cp_type', 'cp_time', 'cp_dose'] # Identify categorical columns

#         ohe = OneHotEncoder() # Load OHE

#         # Get the column names after OHE
#         # Reference: https://stackoverflow.com/questions/54570947/feature-names-from-onehotencoder
#         _ = ohe.fit_transform(X[cat_cols])
#         ohe_names = ohe.get_feature_names(cat_cols)
#         ohe_names = ohe_names.tolist()

#         # Fix new column names to include OHE names and normal feature names
#         X_col_names = [col for col in X_col_names if col\
#             not in cat_cols]
#         ohe_names.extend(X_col_names)

#         # Transform the data with OHE on the indices of the cat variables
#         ct = ColumnTransformer(
#             transformers=[('encoder', OneHotEncoder(), list(range(0,3)))],
#             remainder='passthrough')
#         X = pd.DataFrame(ct.fit_transform(X))
#         X.columns = ohe_names
        
#         y_temp.columns = ['target']
        
#         df = pd.concat([X, y_temp], axis=1) # Recombine into single df

#         df['kfold'] = -1 # Create k-folds column

#         df = df.sample(frac=1).reset_index(drop=True) # Randomize the dataset

#         y = df.target.values # Subset the target column

#         # Initialize the stratified k-fold module from sklearn
#         kf = StratifiedKFold(n_splits=5)

#         # Fill the 'kfold' column with the assigned folds
#         for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
#             df.loc[v_, 'kfold'] = f
            
#         print(class_name)
#         for fold_ in range(5):
#             run(fold_)