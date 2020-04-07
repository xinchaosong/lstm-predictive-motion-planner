import pandas as pd


class DataRecorder:
    def __init__(self, column=None, index=None):
        self.column = column
        self.index = index
        self.df = pd.DataFrame(columns=self.column, index=self.index)

    def append(self, data_list):
        self.df.loc[self.df.shape[0]] = data_list

    def save_to_csv(self, output_path, index=False):
        self.df.to_csv(output_path, index=index)
