import matplotlib.pyplot as plt
import pandas as pd
from task_management.task_list import tasks


def plot(task_name, csv_path, show_fig=True, save_fig=False):
    plt.style.use('ggplot')

    df_loss = pd.read_csv(csv_path)

    fig, (ax_1, ax_2) = plt.subplots(2, 1)
    fig.set_size_inches(12, 8)

    df_loss = df_loss.iloc[:50, :]

    ax_1.set_title('Reconstruction Loss')
    ax_1.set_xlabel('Epoch')
    ax_1.set_ylabel('Mean Squared Error')
    ax_1.plot(df_loss['train_reco_loss'], c='red')
    ax_1.plot(df_loss['vali_reco_loss'], c='blue')
    ax_1.legend(["Training", "Validation"])

    ax_2.set_title('Prediction Loss')
    ax_2.set_xlabel('Epoch')
    ax_2.set_ylabel('Mean Squared Error')
    ax_2.plot(df_loss['train_pred_loss'], c='red')
    ax_2.plot(df_loss['vali_reco_loss'], c='blue')
    ax_2.legend(["Training", "Validation"])

    if save_fig:
        plt.savefig(task_name + '.jpg')

    if show_fig:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    task = tasks[105]
    path = '../data/loss/' + task.task_name + '_loss.csv'
    plot(task_name=task.task_name, csv_path=path, show_fig=True, save_fig=True)
