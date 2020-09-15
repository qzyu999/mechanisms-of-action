import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Load training data
    train_features = pd.read_csv('../input/train_features.csv')
    train_labels_ohe = pd.read_csv('../input/train_targets_scored.csv')

    # Reverse the OHE labels
    y = train_labels_ohe.iloc[:,1:].idxmax(axis=1)
    y = pd.DataFrame(y)
    y.columns = ['target']

    df = pd.concat([train_features, y], axis=1) # Recombine into single df

    df['kfold'] = -1 # Create k-folds column

    df = df.sample(frac=1).reset_index(drop=True) # Randomize the dataset

    y = df.target.values # Subset the target column

    # Initialize the stratified k-fold module from sklearn
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Fill the 'kfold' column with the assigned folds
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    
    df.to_csv('../input/train_folds.csv', index=False) # Save to csv