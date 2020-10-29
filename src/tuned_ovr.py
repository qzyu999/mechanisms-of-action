# tuned_ovr.py
# Fit a model using the tuned model/parameters from model_selection.py.
import os
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import config
import copy


def preprocess_data(num_retained_classes=12):
    """Preprocess the data."""
    X = pd.read_csv(config.FEATURES_FILE)
    # X = pd.read_csv("../input/train_features.csv")
    X.drop(X.columns[0], axis=1, inplace=True)
    # X_test = pd.read_csv("../input/test_features.csv")
    X_test = pd.read_csv(config.TESTING_FILE)
    X_test.drop(X_test.columns[0], axis=1, inplace=True)

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
    X_test = pd.DataFrame(ct.transform(X_test))
    X_test.columns = ohe_names

    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    X.iloc[:, 7:] = sc.fit_transform(X.iloc[:, 7:])
    X_test.iloc[:, 7:] = sc.transform(X_test.iloc[:, 7:])

    return X, X_test


def generate_OVR_targets(num_retained_classes=12):
    """Generate the list of binary OVR target vectors that will be tested."""
    y = pd.read_csv(config.TARGETS_FILE)

    # Add hidden class
    zero_class_indices = y[y.iloc[:, 1:].apply(sum, axis=1) == 0].index
    y["hidden_class"] = 0
    y["hidden_class"].iloc[zero_class_indices] = 1

    class_counts = y.iloc[:, 1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)

    class_counts_sub = class_counts.head(num_retained_classes + 1)
    retained_classes = class_counts_sub.index.values
    y2 = y.iloc[:, 1:]

    ### The following creates 'c' binary target vectors saved in a list: 'binary_vector_list'
    class_index_list = []  # Save indices that contain the class
    for c in retained_classes:
        c_indices = y2.loc[:, c][y2.loc[:, c] == 1].index.values
        class_index_list.append([c, c_indices])

    binary_vector_list = []
    n = len(y)
    for i in class_index_list:  # Loop through class/index pairs
        zeros = [0] * n
        for j in range(n):  # Loop through all rows
            # Check if the index should be one instead
            if j in i[1]:
                zeros[j] = 1
        binary_vector_list.append(pd.DataFrame({i[0]: zeros}))

    return binary_vector_list


if __name__ == "__main__":
    # Load the tuned log-reg eta list
    stat_dict_csv = pd.read_csv("../output/stat_dict.csv")
    log_reg_results = stat_dict_csv.loc[:, "log_reg"]
    eta_list = []
    for i in log_reg_results:
        eta_list.append(float(i.split(",")[1].split(")")[0].split(" ")[1]))

    # Preprocess data and load OVR targets
    X_train, X_test = preprocess_data(num_retained_classes=12)
    ovr_targets = generate_OVR_targets(num_retained_classes=12)

    # Load test submission file
    nonscored_targets = pd.read_csv(config.SAMPLE_SUBMISSION)
    nonscored_targets.replace(0.5, 0, inplace=True)
    nonscored_targets["hidden_class"] = 0

    # Loop through the ovr_targets and fit a model for each class
    for ovr_idx in range(len(ovr_targets)):
        y_temp = copy.deepcopy(ovr_targets[ovr_idx])
        class_name = y_temp.columns[0]

        # Intiialize the classifier
        ### Need to later check the correct model for a given feature
        # model = linear_model.LogisticRegression(random_state=0, max_iter=1e10)
        model = LogisticRegression(
            C=eta_list[ovr_idx],
            random_state=0,
            max_iter=1e10,
        )

        # Fit the model
        model.fit(X_train, y_temp.values.ravel())

        # Create predictions
        y_pred_probs = model.predict_proba(X_test)

        # Update predicted probabilities
        nonscored_targets.loc[:, class_name] = y_pred_probs[:, 1]

    # drop the hidden_class column
    nonscored_targets.drop(["hidden_class"], axis=1, inplace=True)

    nonscored_targets.to_csv(config.OUTPUT_FILE, index=False)
