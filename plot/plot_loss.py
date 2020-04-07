import matplotlib.pyplot as plt
import pandas as pd
from task_management.task_list import tasks


def plot(task_name, csv_path, task=None, show_fig=True, save_fig=False):
    plt.style.use('ggplot')

    df_loss = pd.read_csv(csv_path)

    fig, (ax_1, ax_2) = plt.subplots(2, 1)
    fig.set_size_inches(12, 8)

    if task is not None:
        ax_1.set_title(str(task))
    else:
        ax_1.set_title('Validation Loss of ' + task_name)

    ax_1.set_xlabel('Epoch')
    ax_1.set_ylabel('Validation Reconstruction Loss')
    ax_1.plot(range(df_loss.shape[0]), df_loss['vali_reco_loss'])

    ax_2.set_ylabel('Validation Prediction Loss')
    ax_2.plot(range(df_loss.shape[0]), df_loss['vali_pred_loss'])

    if save_fig:
        plt.savefig(task_name + '.jpg')

    if show_fig:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    task = tasks[105]
    path = '../data/loss/' + task.task_name + '_loss.csv'
    plot(task_name=task.task_name, csv_path=path, task=task, show_fig=True, save_fig=True)
