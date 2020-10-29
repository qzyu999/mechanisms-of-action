# ovr_fe.py
# Extend the OVR method that has been tuned using grid search on various
# models and parameters to try out feature engineering techniques.

import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import itertools
import copy
import csv
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import dask.dataframe as dd
import config

### FE1 sub-functions
def percentile_10(x):
    return np.percentile(x, 10)

def percentile_60(x):
    return np.percentile(x, 60)

def percentile_90(x):
    return np.percentile(x, 90)

def quantile_5(x):
    return np.percentile(x, 5)

def quantile_95(x):
    return np.percentile(x, 95)

def quantile_99(x):
    return np.percentile(x, 99)


def add_FE3(x_train, x_valid, orig_colnames):
    """Add the FE3 based off categorical level counts.
    """
    cat_feat_cols = orig_colnames[0:19] # Subset cat feature column names

    def calc_cat_feat_counts(X_train, orig_colnames, cat_feat_cols):
        """Calculate the counts of each level for each categorical variable
            from the training set.
        """
        # Make into pd df, then subset the OHE vars
        X_train = pd.DataFrame(X_train, columns=orig_colnames)
        ohe_vars = X_train.loc[:,cat_feat_cols]
        cat_feat_counts = ohe_vars.apply(lambda col: sum(col), axis=0)

        return cat_feat_counts

    # Initialize training categorical level counts
    cat_feat_counts = calc_cat_feat_counts(X_train=x_train,
                                           orig_colnames=orig_colnames,
                                           cat_feat_cols=cat_feat_cols)

    def cp_type_mapping(row, training_counts=cat_feat_counts):
        if (row[0] == 1):
            return training_counts[0]
        elif (row[1] == 1):
            return training_counts[1]

    def cp_time_mapping(row, training_counts=cat_feat_counts):
        if (row[0] == 1):
            return training_counts[2]
        elif (row[1] == 1):
            return training_counts[3]
        elif (row[2] == 1):
            return training_counts[4]

    def cp_dose_mapping(row, training_counts=cat_feat_counts):
        if (row[0] == 1):
            return training_counts[5]
        elif (row[1] == 1):
            return training_counts[6]

    def new_feature_mapping(row, training_counts=cat_feat_counts):
        if (row[0] == 1):
            return training_counts[7]
        elif (row[1] == 1):
            return training_counts[8]
        elif (row[2] == 1):
            return training_counts[9]
        elif (row[3] == 1):
            return training_counts[10]
        elif (row[4] == 1):
            return training_counts[11]
        elif (row[5] == 1):
            return training_counts[12]
        elif (row[6] == 1):
            return training_counts[13]
        elif (row[7] == 1):
            return training_counts[14]
        elif (row[8] == 1):
            return training_counts[15]
        elif (row[9] == 1):
            return training_counts[16]
        elif (row[10] == 1):
            return training_counts[17]
        elif (row[11] == 1):
            return training_counts[18]

    # Gather mapping functions into a list
    mapping_fun_list = [cp_type_mapping, cp_time_mapping, cp_dose_mapping, new_feature_mapping]

    def cv_cat_mapping(X_mat, cat_feat_cols=cat_feat_cols, orig_colnames=orig_colnames):
        """Map the OHE categorical variables to their counts within CV.
        """
        X_df = pd.DataFrame(X_mat) # Make matrix into pd.df
        X_df.columns = orig_colnames

        # Create indices for each OHE cat variable
        cat_feat_1 = cat_feat_cols[0:2]
        cat_feat_2 = cat_feat_cols[2:5]
        cat_feat_3 = cat_feat_cols[5:7]
        cat_feat_3 = cat_feat_cols[5:7]
        cat_feat_4 = cat_feat_cols[7:19]
        cat_feat_col_list = [cat_feat_1, cat_feat_2, cat_feat_3, cat_feat_4]

        cat_mapping_list = [] # Map each OHE variable to their counts
        for cat_idx in range(len(cat_feat_col_list)):
            # Apply to X_df
            ohe_var = X_df.loc[:,cat_feat_col_list[cat_idx]]
            mapping_fun = mapping_fun_list[cat_idx]
            cat_mapping_list.append(ohe_var.apply(lambda x: mapping_fun(x), axis=1))

        cat_counts_df = pd.DataFrame(cat_mapping_list).T # Put results into a df
        cat_counts_df.columns=['type_counts', 'time_counts', 'dose_counts', 'new_feature_counts']

        X_df = dd.concat([X_df, cat_counts_df], axis=1) # Combine with X_df
        X_df = X_df.compute() # Return Dask to Pandas

        return X_df

    X_train = cv_cat_mapping(X_mat=x_train, cat_feat_cols=cat_feat_cols, orig_colnames=orig_colnames)
    X_valid = cv_cat_mapping(X_mat=x_valid, cat_feat_cols=cat_feat_cols, orig_colnames=orig_colnames)

    return X_train, X_valid


def add_PCA(x_train, x_valid, orig_colnames, batch_size=10):
    """Uses PCA to add the first two PC's to the train/valid sets.
    """
    # Convert X-matrix back to pd df
    X_train = pd.DataFrame(x_train, columns=orig_colnames)
    X_valid = pd.DataFrame(x_valid, columns=orig_colnames)

    # Subset the cell and gene attributes
    cell_attributes = [c for c in X_train.columns if 'c-' in str(c)]
    gene_attributes = [g for g in X_train.columns if 'g-' in str(g)]
    cell_gene_list = [cell_attributes, gene_attributes]

    # Apply to X_train
    # Loop through the cell/gene attributes and construct their PC's
    cell_gene_pc_list_train = []
    for cell_gene in cell_gene_list:
        # Subset the cell or gene attributes
        cg_sub = X_train.loc[:,cell_gene]

        # Scale the data
        temp_scaler = StandardScaler()
        temp_scaler.fit(cg_sub)
        cg_sub = temp_scaler.transform(cg_sub)

        # Fit the PC's and save the first two
        pca = PCA(n_components=2, random_state=0)
        cg_pcs = pca.fit_transform(cg_sub)
        cell_gene_pc_list_train.append(pd.DataFrame(cg_pcs))

    # Extend the PC to X_valid
    X_valid_copy = copy.deepcopy(X_valid) # Create copy for safety
    cell_gene_pc_list_valid = []

    # Reference: https://stackoverflow.com/questions/41868890/how-to-loop-through-a-python-list-in-batch
    # Batch through X_valid and concat them to an X_train_copy to save the
    # PC's in batches
    for valid_row in range(0, X_valid.shape[0], batch_size):
        # Deepcopy the X_valid and X_train
        X_valid_copy2 = copy.deepcopy(X_valid_copy)
        X_train_copy = copy.deepcopy(X_train)

        # Get batch rows from X_valid and concatenate them to X_train_copy
        temp_rows = copy.deepcopy(X_valid_copy2.iloc[valid_row:valid_row+batch_size,:])
        X_train_copy = pd.concat([X_train_copy, pd.DataFrame(temp_rows)], axis=0)

        # Loop through cell/gene attributes in X_train_copy
        temp_cell_gene_concat_list = []
        for cell_gene in cell_gene_list:
            # Construct PC's
            cg_sub = X_train_copy.loc[:,cell_gene]

            # Get a smaller sample
            sample_size = round(cg_sub.shape[0] / 3)
            cg_sub = cg_sub.iloc[np.random.randint(cg_sub.shape[0], size=sample_size), :]

            # Scale the data
            temp_scaler = StandardScaler()
            temp_scaler.fit(cg_sub)
            cg_sub = temp_scaler.transform(cg_sub)

            pca = PCA(n_components=2, random_state=0)
            cg_pcs = pca.fit_transform(cg_sub)

            # Save the batch rows (it varies) of the 2 PC's to a list
            temp_cell_gene_concat_list.append(cg_pcs[-temp_rows.shape[0]:,:])
        cell_gene_batch_combine = np.concatenate(temp_cell_gene_concat_list, axis=1)
        cell_gene_pc_list_valid.append(cell_gene_batch_combine)

    # Combine cell/gene PC's and convert to df
    cell_gene_pc_df_train = pd.concat(cell_gene_pc_list_train, axis=1)
    X_train = pd.concat([X_train, cell_gene_pc_df_train], axis=1)

    # Combine cell/gene valid PC with original df
    cell_gene_pc_df_valid = pd.DataFrame(np.concatenate(cell_gene_pc_list_valid, axis=0))
    X_valid = pd.concat([X_valid, cell_gene_pc_df_valid], axis=1)

    return X_train, X_valid


def class_wise_log_loss(y_true, y_hat):
    """Calculate the log-loss for just a chosen class."""
    dummy_zero = 1 * 10 ** (-15)  # Compensate for 0's and 1's predictions
    y_hat[y_hat == 0] = dummy_zero
    y_hat[y_hat == 1] = 1 - dummy_zero
    class_log_loss = y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
    class_log_loss = -np.mean(class_log_loss)

    return class_log_loss


def preprocess_data(num_retained_classes=12):
    """Preprocess the data."""
    X = pd.read_csv(config.FEATURES_FILE)
    X.drop(X.columns[0], axis=1, inplace=True)
    y = pd.read_csv(config.TARGETS_FILE)
    y.drop(y.columns[0], axis=1, inplace=True)

    ### Add features before scaling
    ### FE 1
    # Add the row summary statistics
    # List of cell and gene attributes
    cell_attributes = [c for c in X.columns if c[0:2] == 'c-']
    gene_attributes = [g for g in X.columns if g[0:2] == 'g-']
    cell_gene_list = [cell_attributes, gene_attributes]

    # Reference: https://stackoverflow.com/questions/11736407/apply-list-of-functions-on-an-object-in-python
    # Calculate min, max, sum, and mean for cell/gene groups per row
    row_statistics = X.apply(lambda row:
        list(itertools.chain.from_iterable([[f(row[cg]) for f in \
        [min, max, sum, np.mean, np.std, np.var, np.ptp,
         percentile_10, percentile_60, percentile_90,
         quantile_5, quantile_95, quantile_99]] \
        for cg in cell_gene_list])), axis=1)

    # Reference: https://stackoverflow.com/questions/45901018/convert-pandas-series-of-lists-to-dataframe
    # Turn row statistics into a dataframe
    row_summary = pd.DataFrame.from_dict(dict(zip(row_statistics.index, row_statistics.values))).T
    
    X = dd.concat([X, row_summary], axis=1) # Combine with X
    X = X.compute() # Return Dask to Pandas
    
    ### FE 2
    # Combine the categorical variables into a single new feature
    new_feature = (X.cp_type.astype(str) + "_" +
        X.cp_time.astype(str) + "_" +
        X.cp_dose.astype(str))
    ### Hard-coded insert
    X.insert(3, 'new_feature', new_feature)
    ### End FE

    # Add hidden class
    zero_class_indices = y[y.iloc[:, 1:].apply(sum, axis=1) == 0].index
    y["hidden_class"] = 0
    y.loc[zero_class_indices, "hidden_class"] = 1

    class_counts = y.iloc[:, 1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)
    class_counts_sub = class_counts.head(num_retained_classes + 1)
    chosen_classes = class_counts_sub.index.values

    # Save the column names
    X_col_names = X.columns.tolist()
    cat_cols = ["cp_type", "cp_time", "cp_dose", "new_feature"]  # Identify categorical columns
    ohe = OneHotEncoder()  # Load OHE

    _ = ohe.fit_transform(X.loc[:,cat_cols])
    ohe_names = ohe.get_feature_names(cat_cols)
    ohe_names = ohe_names.tolist()

    # Fix new column names to include OHE names and normal feature names
    X_col_names = [col for col in X_col_names if col not in cat_cols]
    ohe_names.extend(X_col_names)

    # Transform the data with OHE on the indices of the cat variables
    ct = ColumnTransformer(
        ### Hard coded to account for 'new_features'
        transformers=[("encoder", OneHotEncoder(), list(range(0, 4)))],
        remainder="passthrough",
    )
    X = pd.DataFrame(ct.fit_transform(X))
    X.columns = ohe_names

    train_idx_list = []
    valid_idx_list = []
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, valid_index in mskf.split(X, y):
        train_idx_list.append(train_index)
        valid_idx_list.append(valid_index)

    return X, y, train_idx_list, valid_idx_list, chosen_classes

def binary_msfk_fun(y_df, chosen_class):
    """Create a OVR binary vector for a set ofchosen classes."""
    y_df_copy = copy.deepcopy(y_df)  # Create copy of target df
    y_df_copy.reset_index(drop=True, inplace=True)  # Reset row indices
    c_indices = y_df_copy[y_df_copy.loc[:, chosen_class] == 1].loc[:, chosen_class]
    n = y_df.shape[0]
    zeros = [0] * n  # Can't do this actually

    for j in range(n):  # Loop through all rows
        # Check if the index should be one instead
        if j in c_indices:
            zeros[j] = 1

    binary_target = pd.DataFrame({chosen_class: zeros})

    return binary_target


def run_cv(
    fold,
    X,
    y,
    train_idx_list,
    valid_idx_list,
    chosen_class,
    fold_log_loss_list,
    model,
):
    """Run the cross-validation."""
    train_idx = train_idx_list[fold]
    valid_idx = valid_idx_list[fold]

    ### These have shifted row names
    x_train = X.iloc[train_idx, :].values
    y_train = y.iloc[train_idx, :]
    x_valid = X.iloc[valid_idx, :].values
    y_valid = y.iloc[valid_idx, :]

    ### Add features in-between CV folds
    ### FE 3
    # Save counts of categorical variables from X_train to X_valid
    orig_colnames = X.columns
    x_train, x_valid = add_FE3(x_train=x_train, x_valid=x_valid, orig_colnames=orig_colnames)
    
    ### FE 4
    # Save batches of PC's from X_train to X_valid
    orig_colnames = x_train.columns # Update col names
    x_train, x_valid = add_PCA(x_train, x_valid, orig_colnames, batch_size=100)
    ### End adding features
    
    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    x_train.iloc[:, 7:] = sc.fit_transform(x_train.iloc[:, 7:])
    x_valid.iloc[:, 7:] = sc.transform(x_valid.iloc[:, 7:])

    ### Need a non-scored df of dimensions equal to validation set
    non_scored_y_valid = copy.deepcopy(y_valid)
    non_scored_y_valid.replace(1, 0, inplace=True)

    y_temp = binary_msfk_fun(y_df=y_train, chosen_class=chosen_class)

    class_name = y_temp.columns[0]

    # Fit the model
    model.fit(x_train, y_temp.values.ravel())

    # Create predictions
    y_pred_probs = model.predict_proba(x_valid)[:, 1]

    # Calculate the class-log-loss and save it to the list
    class_log_loss_score = class_wise_log_loss(
        y_true=y_valid.loc[:, class_name], y_hat=y_pred_probs
    )
    fold_log_loss_list.append(class_log_loss_score)


if __name__ == "__main__":
    # Get tuned parameters for logistic regression
    stat_dict_csv = pd.read_csv(config.OUTPUT_STAT_DICT)
    log_reg_results = stat_dict_csv.loc[:, "log_reg"]
    eta_list = []
    for i in log_reg_results:
        eta_list.append(float(i.split(",")[1].split(")")[0].split(" ")[1]))

    # Preprocess the data
    X, y, train_idx_list, valid_idx_list, chosen_classes = preprocess_data(
        num_retained_classes=12
    )

    result_dict = {}  # Create a dictionary to store results
    for class_ in chosen_classes:
        result_dict[class_] = []

    # For each class fit the tuned model on the engineered features
    chosen_classes_list = chosen_classes.tolist()
    for class_idx in range(len(chosen_classes)):  # Loop through classes
        ith_class = chosen_classes_list[class_idx]
        print(
            f"Class: {ith_class}, index {class_idx + 1} out of {len(chosen_classes)}"
        )
        
        model_ = LogisticRegression(
            C=eta_list[class_idx],
            random_state=0,
            max_iter=1e10,
        )
        
        fold_log_loss_list_ = [] # Do CV
        for fold_ in range(5):
            run_cv(
                fold=fold_,
                X=X,
                y=y,
                train_idx_list=train_idx_list,
                valid_idx_list=valid_idx_list,
                chosen_class=ith_class,
                fold_log_loss_list=fold_log_loss_list_,
                model=model_,
            )
            
        mean_log_loss = np.mean(fold_log_loss_list_) # Save CV average score
        result_dict[ith_class].append(mean_log_loss)

    # Save results to csv
    result_dict_list = [dict(class_name=i, clf_result=j) for i, j in result_dict.items()]
    fieldnames = ["class_name", "clf_result"]
    with open(config.OUTPUT_FE_DICT, "w") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(result_dict_list)
