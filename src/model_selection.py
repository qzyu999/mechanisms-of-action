# model_selection.py
# Go through each of the top K most populated classes and identify the
# model + parameter combination that leads to the most optimal log-loss for
# that k'th class.

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
import csv
import itertools
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import copy
import config


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

    # Add hidden class
    zero_class_indices = y[y.iloc[:, 1:].apply(sum, axis=1) == 0].index
    y["hidden_class"] = 0
    y.loc[zero_class_indices, "hidden_class"] = 1

    class_counts = y.iloc[:, 1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)
    class_counts_sub = class_counts.head(num_retained_classes+1)
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
    # clf_list = ["log_reg", "svm", "rf", "dt", "knn", "nb", "xgb"]
    clf_list = ["log_reg", "rf", "dt", "knn", "nb", "xgb"]

    param_grid = {
        "log_reg": {
            "Penalty": ["l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        "svm": {
            "C": [round((0.1) * ((0.1) ** (n - 1)), 5) for n in reversed(range(-3, 4))],
            "gamma": ["auto"],
            "class_weight": ["balanced", None],
            "probability": [True],
        },
        "rf": {
            "n_estimators": [120],
            "max_depth": [8],
            "min_samples_split": [10],
            "min_samples_leaf": [10],
            "max_features": ["log2"],
        },
        "dt": {
            "max_depth": [8],
            "min_samples_split": [10],
            "min_samples_leaf": [10],
            "max_features": ["log2"],
        },
        "knn": {
            "n_neighbors": [round((2) * ((2) ** (n - 1)), 5) for n in range(1, 3)],
            "p": [2],
        },
        "nb": {"dummy_param": [None]},
        "xgb": {
            "eta": [0.01],
            "gamma": [0.05],
            "max_depth": [8],
            "min_child_weight": [1],
            "subsample": [0.6],
            "colsample_bytree": [0.6],
            "lambda": [0.01],
            "alpha": [0.1],
        },
    }

    ### Note: In the future this should allow for feature engineering.
    # Preprocess the data
    X, y, train_idx_list, valid_idx_list, chosen_classes = preprocess_data(num_retained_classes=12)

    result_dict = {}  # Create a dictionary to store results
    for class_ in chosen_classes:
        temp_clf_dict = {}
        for clf_ in clf_list:
            temp_clf_dict[clf_] = []
        result_dict[class_] = temp_clf_dict

    # For each class fit a model using grid search
    chosen_classes_list = chosen_classes.tolist()
    for class_idx in chosen_classes: # Loop through classes
        ith_class = chosen_classes_list.index(class_idx)
        print(f"Class: {class_idx}, index {ith_class + 1} out of {len(chosen_classes_list)}")
        for clf_idx in clf_list:  # Loop through models
            print(f"Classifier: {clf_idx}")

            # Create parameter combinations
            clf_param_grid = param_grid[clf_idx]
            param_names = [key for key in clf_param_grid.keys()]
            param_combos = itertools.product(
                *(clf_param_grid[p_name] for p_name in param_names)
            )
            param_combos_list = list(param_combos)
            total_param_combos = len(param_combos_list)

            # 3) Loop through parameters
            for p_combo_idx in range(total_param_combos):  # Loop through parameters
                print(
                    f"Parameter combination index: {p_combo_idx + 1} out of {total_param_combos}"
                )
                param_combo_ = param_combos_list[p_combo_idx]
                model_ = set_model_params(clf_name=clf_idx, params=param_combo_)

                # 4) Run CV on the parameter
                # Calculate the log-loss for clf_idx-class_idx-p_combo_idx
                fold_log_loss_list_ = []
                for fold_ in range(5):
                    run_cv(
                        fold=fold_,
                        X=X,
                        y=y,
                        train_idx_list=train_idx_list,
                        valid_idx_list=valid_idx_list,
                        chosen_class=class_idx,
                        fold_log_loss_list=fold_log_loss_list_,
                        param_combo_=param_combo_,
                        model=model_,
                    )

                mean_log_loss = np.mean(fold_log_loss_list_)
                result_dict[class_idx][clf_idx].append([mean_log_loss, param_combo_])

    # Organize the results into best parameters per model and best overall
    stat_dict = {}
    best_dict = {}  # Initialize dictionaries
    for class_ in chosen_classes:
        temp_clf_dict = {}
        for clf_ in clf_list:
            temp_clf_dict[clf_] = []
        stat_dict[class_] = temp_clf_dict
        best_dict[class_] = []

    # Fill the dictionaries showing best parameters per model (stat_dict)
    # and best model per class (best_dict)
    for class_idx in chosen_classes:  # Loop through classes
        for clf_idx in clf_list:  # Loop through classifiers
            # Find best parameter (index) per model
            temp_class_clf_list = result_dict[class_idx][clf_idx]
            cv_score_list = [cv_score[0] for cv_score in temp_class_clf_list]
            best_clf_score = min(cv_score_list)
            best_idx = cv_score_list.index(best_clf_score)
            stat_dict[class_idx][clf_idx] = [
                temp_class_clf_list[best_idx][1],
                best_clf_score,
            ]

        # Find best model/param combo per class
        # Reference: https://stackoverflow.com/questions/34249441/finding-minimum-value-in-dictionary
        best_clf = min(stat_dict["hidden_class"].items(), key=lambda x: x[1][1])
        best_dict[class_idx] = [best_clf]

    # Reference: https://stackoverflow.com/questions/29771895/save-nested-dictionary-with-differing-number-of-dictionaries
    # Save results to csv
    best_dict_list = [dict(class_name=i, clf_result=j) for i, j in best_dict.items()]
    fieldnames = ["class_name", "clf_result"]
    with open(config.OUTPUT_BEST_DICT, "w") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(best_dict_list)

    nested_dict_keys = []  # Set fieldnames for stat_dict
    for idx_i, idx_j in stat_dict.items():
        for idx_k, idx_l in idx_j.items():
            nested_dict_keys.append(idx_k)
    nested_dict_keys = list(set(nested_dict_keys))
    fieldnames = ["class_name"] + nested_dict_keys

    # Reference: https://stackoverflow.com/questions/29400631/python-writing-nested-dictionary-to-csv
    # Save results to csv
    with open(config.OUTPUT_STAT_DICT, "w") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        for key in stat_dict:
            w.writerow(
                {field: stat_dict[key].get(field) or key for field in fieldnames}
            )
