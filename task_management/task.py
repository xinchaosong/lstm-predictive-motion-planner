import pandas as pd


class Task:
    def __init__(self, task_id, data_set_csv, data_description, input_size, learning_rate, weight_decay,
                 sparsity_weight, hidden_size, num_epochs, gpu_index, date, task_name=None):
        self.task_cvs_path = '../task_management/task_info.csv'
        self.df = pd.read_csv(self.task_cvs_path)
        self.task_id = task_id
        self.data_set_csv = data_set_csv
        self.input_size = input_size
        self.data_description = data_description
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.sparsity_weight = sparsity_weight
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.gpu_index = gpu_index
        self.date = date

        if task_name is None:
            self.task_name = 'task_' + str(self.task_id)
        else:
            self.task_name = task_name

    def save(self):
        data_row = [self.task_id, self.data_description, self.lr, self.weight_decay, self.sparsity_weight,
                    self.hidden_size, self.num_epochs, self.gpu_index, self.date]

        self.df.loc[self.df.shape[0]] = data_row
        self.df.to_csv(self.task_cvs_path, index=False)

    def has_done(self):
        if self.task_id in list(self.df.task_id):
            return True

    def __str__(self):
        return "Task ID: " + str(self.task_id) + "; Data Set: " + self.data_description \
               + "\nLearning Rate: " + str(self.lr) + "; Weight Decay: " + str(self.weight_decay) \
               + "; Sparsity Weight: " + str(self.sparsity_weight) + "; Hidden Size: " + str(self.hidden_size) \
               + "\nEpochs: " + str(self.num_epochs) + "; GPU Index: " + str(self.gpu_index) \
               + "; Date: " + str(self.date)
