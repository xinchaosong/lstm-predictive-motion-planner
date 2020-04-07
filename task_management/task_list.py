from task_management.task import Task

tasks = {}

tasks[104] = Task(task_id=104, data_set_csv='raw_data_101', data_description='rd101:100k:5k:5k', input_size=1,
                  learning_rate=0.001, weight_decay=0, sparsity_weight=0, hidden_size=128,
                  num_epochs=400, gpu_index=0, date="2019-12-07")
tasks[105] = Task(task_id=105, data_set_csv='raw_data_102', data_description='rd102:100k:10k:10k', input_size=1,
                  learning_rate=0.001, weight_decay=0, sparsity_weight=0, hidden_size=128,
                  num_epochs=600, gpu_index=0, date="2019-12-09")
tasks[106] = Task(task_id=106, data_set_csv='raw_data_102', data_description='rd102:100k:10k:10k', input_size=1,
                  learning_rate=0.001, weight_decay=0, sparsity_weight=0, hidden_size=128,
                  num_epochs=200, gpu_index=0, date="2019-12-09")