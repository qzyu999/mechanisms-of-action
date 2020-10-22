# clf_param_search.py
# The goal here is to build upon model_tuning_cv.py, by allowing for multiple
# classifiers and parameters to be grid searched.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import itertools
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import copy
import config


def multilabel_log_loss(y_valid, y_pred):
    """Calculate the log-loss for the multilabel case."""
    N, M = y_valid.shape  # Create temp matrix to store values
    zero_mat = np.zeros((N, M))

    dummy_zero = 1 * 10 ** (-15)  # Compensate for 0's and 1's predictions
    y_pred.replace(0, dummy_zero, inplace=True)
    y_pred.replace(1, 1 - dummy_zero, inplace=True)

    for m in range(M):  # Calculate log-loss per index
        for n in range(N):
            y_true = y_valid.iloc[n, m]
            y_hat = y_pred.iloc[n, m]
            temp_log_loss = y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
            zero_mat[n, m] = temp_log_loss

    log_loss_score = -zero_mat.mean(axis=0).mean()

    return log_loss_score


def preprocess_data():
    """Preprocess the data."""
    X = pd.read_csv(config.FEATURES_FILE)
    X.drop(X.columns[0], axis=1, inplace=True)
    y = pd.read_csv(config.TARGETS_FILE)
    y.drop(y.columns[0], axis=1, inplace=True)

    # Add hidden class
    zero_class_indices = y[y.iloc[:, 1:].apply(sum, axis=1) == 0].index
    y["hidden_class"] = 0
    y.loc[zero_class_indices, "hidden_class"] = 1

    class_counts = y.iloc[:, 1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)
    class_counts_sub = class_counts.head(13)
    chosen_classes = class_counts_sub.index.values

    # Save the column names
    X_col_names = X.columns.tolist()
    cat_cols = ["cp_type", "cp_time", "cp_dose"]  # Identify categorical columns
    ohe = OneHotEncoder()  # Load OHE
    _ = ohe.fit_transform(X[cat_cols])
    ohe_names = ohe.get_feature_names(cat_cols)
    ohe_names = ohe_names.tolist()

    # Fix new column names to include OHE names and normal feature names
    X_col_names = [col for col in X_col_names if col not in cat_cols]
    ohe_names.extend(X_col_names)

    # Transform the data with OHE on the indices of the cat variables
    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), list(range(0, 3)))],
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


def binary_msfk_fun(y_df, chosen_classes):
    """Create a OVR binary vector for a set of chosen classes."""
    y_df_copy = copy.deepcopy(y_df)
    y_df_copy.reset_index(drop=True, inplace=True)
    chosen_classes = chosen_classes

    ### The following creates 'c' binary target vectors saved in a list: 'binary_vector_list'
    class_index_list = []  # Save indices that contain the class
    for c in chosen_classes:
        # These are row-indices
        c_indices = y_df_copy[y_df_copy.loc[:, c] == 1].loc[:, c]
        class_index_list.append([c, c_indices])

    # For each class, generate a binary target vector
    binary_vector_list = []
    n = y_df.shape[0]

    for i in class_index_list:  # Loop through class/index pairs
        zeros = [0] * n  # Can't do this actually
        for j in range(n):  # Loop through all rows
            # Check if the index should be one instead
            if j in i[1]:
                zeros[j] = 1
        binary_vector_list.append(pd.DataFrame({i[0]: zeros}))

    return binary_vector_list


def run_cv(
    fold,
    X,
    y,
    train_idx_list,
    valid_idx_list,
    chosen_classes,
    log_loss_list,
    param_combo_,
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

    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    x_train[:, 7:] = sc.fit_transform(x_train[:, 7:])
    x_valid[:, 7:] = sc.transform(x_valid[:, 7:])

    ### Need a non-scored df of dimensions equal to validation set
    non_scored_y_valid = copy.deepcopy(y_valid)
    non_scored_y_valid.replace(1, 0, inplace=True)

    # Do OVR encoding on the training targets
    ovr_targets = binary_msfk_fun(y_df=y_train, chosen_classes=chosen_classes)

    for i in ovr_targets:  # Loop through the OVR targets and fit a model
        y_temp = copy.deepcopy(i)
        class_name = y_temp.columns[0]

        ### Is this a bug? There seems to be a class name included in the fit...
        # Fit the model
        model.fit(x_train, y_temp.values.ravel())

        # Create predictions
        y_pred_probs = model.predict_proba(x_valid)

        # Fill the non-scored y's
        non_scored_y_valid.loc[:, class_name] = y_pred_probs[:, 1]

    # Go through each row and find the column with the larget value
    chosen_classes_per_row = non_scored_y_valid.iloc[:, 1:].idxmax(axis=1)

    for index, row in non_scored_y_valid.iterrows():
        max_class = chosen_classes_per_row[index]  # Subset the selected class
        row[row.index.isin([max_class]) == False] = 0
        non_scored_y_valid.loc[index, :] = row

    # drop the hidden_class column
    non_scored_y_valid.drop(["hidden_class"], axis=1, inplace=True)
    y_valid = y_valid.drop(["hidden_class"], axis=1)

    log_loss_score = multilabel_log_loss(y_valid=y_valid, y_pred=non_scored_y_valid)

    # print(f"Fold={fold}, Log-Loss={log_loss_score}")
    log_loss_list.append(log_loss_score)


def set_model_params(clf_name, params):
    """Set the parameters for a model during grid search."""
    if clf_name == "log_reg":
        model = LogisticRegression(
            penalty=params[0],
            C=params[1],
            random_state=0,
            max_iter=1e10,
        )
    elif clf_name == "svm":
        model = SVC(
            C=params[0], gamma=params[1], class_weight=params[2], probability=params[3]
        )
    elif clf_name == "rf":
        model = RandomForestClassifier(
            n_estimators=params[0],
            max_depth=params[1],
            min_samples_split=params[2],
            min_samples_leaf=params[3],
            max_features=params[4],
        )
    elif clf_name == "dt":
        model = DecisionTreeClassifier(
            max_depth=params[0],
            min_samples_split=params[1],
            min_samples_leaf=params[2],
            max_features=params[3],
        )
    elif clf_name == "knn":
        model = KNeighborsClassifier(n_neighbors=params[0], p=params[1])
    elif clf_name == "nb":
        model = GaussianNB()
    elif clf_name == "xgb":
        model = XGBClassifier(
            learning_rate=params[0],
            gamma=params[1],
            max_depth=params[2],
            min_child_weight=params[3],
            subsample=params[4],
            colsample_bytree=params[5],
            reg_lambda=params[6],
            reg_alpha=params[7],
        )

    return model


if __name__ == "__main__":
    # clf_list = ["log_reg", "svm", "rf", "df", "knn", "nb"]
    clf_list = ["svm"]
    param_grid = {
        "log_reg": {
            "Penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1],
        },
        "svm": {
            "C": [round((0.1) * ((0.1) ** (n - 1)), 5) for n in reversed(range(-1, 2))],
            "gamma": ["auto"],
            "class_weight": ["balanced"],
            "probability": [True],
        },
        "rf": {
            "n_estimators": [120, 800],
            "max_depth": [5, 25],
            "min_samples_split": [1, 100],
            "min_samples_leaf": [1, 10],
            "max_features": ["log2"],
        },
        "df": {
            "max_depth": [5, 25],
            "min_samples_split": [1, 10],
            "min_samples_leaf": [1, 10],
            "max_features": ["log2"],
        },
        "knn": {
            "n_neighbors": [round((2) * ((2) ** (n - 1)), 5) for n in range(1, 3)],
            "p": [2, 3],
        },
        "nb": {"dummy_param": [None, None]},
        "xgb": {
            "eta": [0.01, 0.015, 0.025, 0.05, 0.1],
            "gamma": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "max_depth": [3, 5, 7, 9, 12, 15, 17, 25],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "lambda": [0.01, 0.1, 1.0],
            "alpha": [0, 0.1, 0.5, 1.0],
        },
    }
    ### Note: In the future this should allow for feature engineering.
    # Preprocess the data
    X, y, train_idx_list, valid_idx_list, chosen_classes = preprocess_data()

    historic_log_loss_list = []
    for clf_idx in clf_list:  # Loop through models
        print(f"Classifier: {clf_idx}")
        clf_param_grid = param_grid[clf_idx]

        param_names = [
            key for key in clf_param_grid.keys()
        ]  # Create parameter combinations
        param_combos = itertools.product(
            *(clf_param_grid[p_name] for p_name in param_names)
        )
        param_combos_list = list(param_combos)
        total_param_combos = len(param_combos_list)

        for p_combo_idx in range(total_param_combos):  # Loop through parameters
            # Intiialize the classifier
            print(
                f"Parameter combination index: {p_combo_idx + 1} out of {total_param_combos}"
            )
            param_combo_ = param_combos_list[p_combo_idx]
            model_ = set_model_params(clf_name=clf_idx, params=param_combo_)

            log_loss_list = []  # Do CV on the gridsearch combo
            for fold_ in range(5):
                run_cv(
                    fold_,
                    X,
                    y,
                    train_idx_list,
                    valid_idx_list,
                    chosen_classes,
                    log_loss_list,
                    param_combo_,
                    model_,
                )

            mean_log_loss = np.mean(log_loss_list)
            print(f"CV average={mean_log_loss}, Params={param_combo_}")
            historic_log_loss_list.append(
                [clf_idx, param_combo_, log_loss_list, mean_log_loss]
            )
