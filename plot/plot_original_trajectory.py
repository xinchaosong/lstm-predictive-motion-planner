import matplotlib.pyplot as plt
import pandas as pd


def plot_original_trajectory(csv_path):
    plt.style.use('ggplot')

    print("Loading the data...")
    df = pd.read_csv(csv_path, header=None)
    print("Data loaded successfully. The dimension is:")
    print(df.shape)

    print("Plotting...")

    i = 400

    plt.plot(df.iloc[i, :], 'red')

    plt.show()


if __name__ == '__main__':
    name = 'raw_data_102'
    path = '../data/raw/' + name + '.csv'

    plot_original_trajectory(path)
