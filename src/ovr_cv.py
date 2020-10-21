# ovr_cv.py
# The goal is to develop some CV scheme to allow local testing of models before
# submitting a final (tuned) model to the leaderboard.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import pandas as pd
import numpy as np
import copy
import config


def preprocess_data():
    """The preprocess_X() function will do the initial preprocessing for the
    dataset features.
    """
    X = pd.read_csv(config.FEATURES_FILE)
    y = pd.read_csv(config.TARGETS_FILE)

    # Add hidden class
    zero_class_indices = y[y.iloc[:, 1:].apply(sum, axis=1) == 0].index
    y["hidden_class"] = 0
    y.loc[zero_class_indices, "hidden_class"] = 1

    class_counts = y.iloc[:, 1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)

    ### Hard coded # of classes

    class_counts_sub = class_counts.head(13)
    chosen_classes = class_counts_sub.index.values

    X.drop(X.columns[0], axis=1, inplace=True)

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

    # Reverse the OHE labels
    y = y.iloc[:, 1:].idxmax(axis=1)
    y = pd.DataFrame(y)
    y.columns = ["target"]

    df = pd.concat([X, y], axis=1)  # Recombine into single df

    df["kfold"] = -1  # Create k-folds column

    df = df.sample(frac=1).reset_index(drop=True)  # Randomize the dataset

    y = df.target.values  # Subset the target column

    # Initialize the stratified k-fold module from sklearn
    kf = StratifiedKFold(n_splits=5)

    # Fill the 'kfold' column with the assigned folds
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    return df, chosen_classes


def y_arr_to_df(y_arr):
    """Change the y array into a y dataframe."""
    y_df_template = pd.read_csv(config.SAMPLE_SUBMISSION)
    y_df_template.replace(0.5, 0, inplace=True)
    y_df_template["hidden_class"] = 0
    y_df_template.drop(["sig_id"], axis=1, inplace=True)

    n_rows = y_arr.shape[0]
    m_rows = y_df_template.shape[0]
    new_rows = n_rows - m_rows

    keys = y_df_template.columns
    values = [[0] * new_rows] * len(y_df_template.columns)
    extra_rows_dict = dict(zip(keys, values))
    extra_rows_df = pd.DataFrame(extra_rows_dict)

    y_df_template = y_df_template.append(extra_rows_df, ignore_index=True)

    for col_idx in range(len(y_arr)):
        y_df_template.loc[col_idx, y_arr[col_idx]] = 1

    return y_df_template


def binary_vector_fun(y_df, chosen_classes):
    """Create a OVR binary vector for a set of chosen classes."""
    ### The following creates 'c' binary target vectors saved in a list
    class_index_list = []  # Save indices that contain the class
    for c in chosen_classes:
        c_indices = y_df.loc[:, c][y_df.loc[:, c] == 1].index.values
        class_index_list.append([c, c_indices])

    binary_vector_list = []
    n = y_df.shape[0]
    for i in class_index_list:  # Loop through class/index pairs
        zeros = [0] * n
        for j in range(n):  # Loop through all rows
            # Check if the index should be one instead
            if j in i[1]:
                zeros[j] = 1
        binary_vector_list.append(pd.DataFrame({i[0]: zeros}))

    return binary_vector_list


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


def run_cv(fold, df, chosen_classes, log_loss_list):
    """Run the cross-validation."""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Drop the target and and convert to numpy
    x_train = df_train.drop(["target", "kfold"], axis=1).values
    y_train = df_train.target.values
    y_train_df = y_arr_to_df(y_arr=y_train)

    # Do OVR encoding on the training targets
    ovr_targets = binary_vector_fun(y_df=y_train_df, chosen_classes=chosen_classes)

    # Repeat for validation data
    x_valid = df_valid.drop(["target", "kfold"], axis=1).values
    y_valid = df_valid.target.values
    y_valid = y_arr_to_df(y_arr=y_valid)

    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    x_train[:, 7:] = sc.fit_transform(x_train[:, 7:])
    x_valid[:, 7:] = sc.transform(x_valid[:, 7:])

    ### Need a non-scored df of dimensions equal to validation set
    non_scored_y_valid = copy.deepcopy(y_valid)
    non_scored_y_valid.replace(1, 0, inplace=True)

    ### So interestingly, this needs to be repeated for each of the chosen classes.
    for i in ovr_targets:
        y_temp = copy.deepcopy(i)
        class_name = y_temp.columns[0]

        # Intiialize the classifier
        model = linear_model.LogisticRegression(random_state=0, max_iter=1e10)

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
        row[row.index.isin([max_class, "sig_id"]) == False] = 0
        non_scored_y_valid.iloc[index, :] = row

    # drop the hidden_class column
    non_scored_y_valid.drop(["hidden_class"], axis=1, inplace=True)
    y_valid.drop(["hidden_class"], axis=1, inplace=True)

    log_loss_score = multilabel_log_loss(y_valid=y_valid, y_pred=non_scored_y_valid)

    print(f"Fold={fold}, Log-Loss={log_loss_score}")
    log_loss_list.append(log_loss_score)


if __name__ == "__main__":
    df, chosen_classes = preprocess_data()

    log_loss_list = []
    for fold_ in range(5):
        run_cv(fold_, df, chosen_classes, log_loss_list)

    print(f"CV average={np.mean(log_loss_list)}")
