import pandas as pd

frames = []

for i in range(10):
    df = pd.read_csv('../data/raw/raw_data_101_%s.csv' % i, header=None)
    frames.append(df)

df = pd.concat(frames).sample(frac=1)
print(df.shape)

df.to_csv("../data/raw/raw_data_101.csv", index=False, header=False)
