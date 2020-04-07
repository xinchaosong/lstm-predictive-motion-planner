import pandas as pd


def generate_data_set(source_df, output_file_path, from_proportion, to_proportion):
    """
    Makes the data set in the given proportion of the original data.
    :param source_df: the pandas data frame of the data source
    :param output_file_path: the path of the output data file
    :param from_proportion:
    :param to_proportion:
    :return: None
    """

    print("Making the data set...")

    from_index = int(round(source_df.shape[0] * from_proportion))
    to_index = int(round(source_df.shape[0] * to_proportion))

    data_set = source_df[from_index: to_index]
    data_set.to_csv(output_file_path, index=False, header=False)

    print("The size of the generated data set:", data_set.shape)
    print("Task Done.")


if __name__ == '__main__':
    raw_data_name = 'raw_data_102'
    input_file_path = '../data/raw/' + raw_data_name + '.csv'
    print("Loading the data...")
    df = pd.read_csv(input_file_path, header=None)
    print("Data loaded successfully.")
    print(df.shape)
    df = df.sample(frac=1)
    print(df.shape)
    boundary_0 = 100000 / df.shape[0]
    boundary_1 = 110000 / df.shape[0]
    boundary_end = df.shape[0] / df.shape[0]

    generate_data_set(
        source_df=df,
        output_file_path='../data/data_set/' + raw_data_name + '_train.csv',
        from_proportion=0,
        to_proportion=boundary_0
    )

    generate_data_set(
        source_df=df,
        output_file_path='../data/data_set/' + raw_data_name + '_valid.csv',
        from_proportion=boundary_0,
        to_proportion=boundary_1
    )

    generate_data_set(
        source_df=df,
        output_file_path='../data/data_set/' + raw_data_name + '_test.csv',
        from_proportion=boundary_1,
        to_proportion=boundary_end
    )
