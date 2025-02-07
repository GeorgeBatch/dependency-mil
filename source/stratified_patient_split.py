import pandas as pd
from sklearn.model_selection import train_test_split


def group_split(df, groups, test_size=0.2, random_state=42, shuffle=True):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=shuffle)

    if groups is not None:
        # get the set of unique groups in the train set and test set
        train_groups = set(groups[df_train.index])
        test_groups = set(groups[df_test.index])
        # for the intersecting groups, move them from the test set to the train set
        intersecting_groups = train_groups.intersection(test_groups)

        for group in intersecting_groups:
            # get the index of all rows in the test set that belong to this group
            idx = df_test.index[groups[df_test.index] == group]
            # move these rows from the test set to the train set
            df_train = pd.concat([df_train, df_test.loc[idx]])
            # remove these rows from the test set
            df_test = df_test.drop(idx)

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def create_stratify_column(dataframe, feature_col_names):
    """
    Create a new column for stratification based on the combination of specified feature_col_names and add it to the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        feature_col_names (tuple): Tuple of feature column names to be combined for stratification.

    Returns:
        pd.DataFrame: The DataFrame with the stratify column added.
    """
    dataframe_copy = dataframe.copy()  # Create a copy to avoid modifying the original DataFrame
    dataframe_copy['stratify_column'] = dataframe_copy[list(feature_col_names)].apply(
        lambda row: '_'.join(row.astype(str)), axis=1)
    # train_test_split errors if there is only one value in any of the groups in the stratify_column
    # print(dataframe_copy['stratify_column'].value_counts())
    return dataframe_copy


def split_dataset_by_patient(dataframe, groups_col_name, feature_col_names, test_size=0.2, random_state=42):
    """
    Split a dataset into train and test sets while ensuring all records for a specific patient are either in the train or test set.
    Stratification is performed based on the combination of specified feature_col_names.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        feature_col_names (tuple): Tuple of feature column names for stratification.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Seed for random number generation (optional).

    Returns:
        pd.DataFrame, pd.DataFrame: Returns train and test DataFrames.
    """
    dataframe_with_stratify = create_stratify_column(dataframe, feature_col_names)
    patient_stratify_mapping = dataframe_with_stratify.groupby('patient_id')['stratify_column'].last()

    train_patients, test_patients = train_test_split(patient_stratify_mapping.index, stratify=patient_stratify_mapping,
                                                     test_size=test_size, random_state=random_state)

    train_set = dataframe_with_stratify[dataframe_with_stratify['patient_id'].isin(train_patients)].copy()
    test_set = dataframe_with_stratify[dataframe_with_stratify['patient_id'].isin(test_patients)].copy()

    # Drop the 'stratify_column' as it was only used for splitting
    train_set.drop(columns=['stratify_column'], inplace=True)
    test_set.drop(columns=['stratify_column'], inplace=True)

    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('../experiment-label-files/OUH_LUAD_vs_rest.csv')
    # X = df['path']
    # y = df['label']
    groups_series = df['patient_id']

    # Test the group_split function
    df_train, df_test = group_split(df=df, groups=groups_series, test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Print the train and test sets
    print("Training Set:")
    print(df_train.head())
    print("")

    print("Test Set:")
    print(df_test.head())
    print("")

    # ipdb.set_trace()

    # Check that each group is fully placed in either train or test set
    train_groups = set(groups_series[df_train.index])
    test_groups = set(groups_series[df_test.index])

    # assert len(train_groups.intersection(test_groups)) == 0, \
    #     (f"Each group should be fully placed in either train or test set."
    #      f"\n Intersection: {train_groups.intersection(test_groups)}")

    y = df['label']
    y_train = df_train['label']
    y_test = df_test['label']
    # Calculate class percentages in train and test sets
    all_class_percentages = y.value_counts(normalize=True) * 100
    train_class_percentages = y_train.value_counts(normalize=True) * 100
    test_class_percentages = y_test.value_counts(normalize=True) * 100

    print("Class Percentages in All Data:")
    print(all_class_percentages)
    print("Total number of samples:", len(y))
    print("")

    print("Class Percentages in Training Set:")
    print(train_class_percentages)
    print("Total number of samples:", len(y_train))
    print("")

    print("Class Percentages in Test Set:")
    print(test_class_percentages)
    print("Total number of samples:", len(y_test))
    print()
