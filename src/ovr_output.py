# ovr_output.py
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

def preprocess_X():
    """The preprocess_X() function will do the initial preprocessing for the
        dataset features.
    """
    X = pd.read_csv("../input/train_features.csv")
    X_test = pd.read_csv("../input/test_features.csv")
    X.drop(X.columns[0], axis=1, inplace=True)
    X_test.drop(X_test.columns[0], axis=1, inplace=True)

    # Save the column names
    X_col_names = X.columns.tolist()
    cat_cols = ['cp_type', 'cp_time', 'cp_dose'] # Identify categorical columns
    ohe = OneHotEncoder() # Load OHE
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
    ### Unsure if this is correct ct.transform()
    X_test = pd.DataFrame(ct.transform(X_test))
    X_test.columns = ohe_names

    # Apply feature scaling to the numeric attributes
    sc = StandardScaler()
    X = X.values
    X_test = X_test.values
    X[:,7:] = sc.fit_transform(X[:,7:])
    X_test[:,7:] = sc.transform(X_test[:,7:])
    
    return X, X_test

def generate_OVR_targets():
    """Generate the list of binary OVR target vectors that will be tested.
    """
    y = pd.read_csv("../input/train_targets_scored.csv")
    class_counts = y.iloc[:,1:].sum(axis=0)
    class_counts = class_counts.sort_values(ascending=False)
    
    ### Hard coded # of classes

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

    return binary_vector_list

if __name__ == "__main__":
    ovr_targets = generate_OVR_targets()

    # Edit to use it for sample submission
    nonscored_targets = pd.read_csv(config.SAMPLE_SUBMISSION)
    nonscored_targets.replace(0.5, 0, inplace=True)

    ### Can OHE and standard scale the X_train first
    X_train, X_test = preprocess_X()

    for i in ovr_targets:
        y_temp = copy.deepcopy(i)
        class_name = y_temp.columns[0]

        # Intiialize the classifier
    ### Need to later check the correct model for a given feature
        model = linear_model.LogisticRegression(random_state=0, max_iter=1e10)

        # Fit the model
        model.fit(X_train, y_temp.values.ravel())

        # Create predictions
        y_pred_probs = model.predict_proba(X_test)

        # Update predicted probabilities
        nonscored_targets.loc[:,class_name] = y_pred_probs[:,0]

    # Go through each row and find the column with the larget value
    chosen_classes_per_row = nonscored_targets.iloc[:,1:].idxmax(axis=1)

    # Reference: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    for index, row in nonscored_targets.iterrows():
        max_class = chosen_classes_per_row[index] # Subset the selected class
        row[row.index.isin([max_class, 'sig_id']) == False] = 0
        nonscored_targets.iloc[index,:] = row

    nonscored_targets.to_csv('../output/submission.csv', index=False)