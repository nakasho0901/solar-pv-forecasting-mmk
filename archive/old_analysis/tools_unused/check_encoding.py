import pandas as pd

file = r"C:\Users\nakas\my_project\solar-kan\dataset_PV\raw_sec\20150101-20150112SecCsv.csv"

# cp932 (Shift-JIS) で読み込んで、先頭3行だけ表示
df = pd.read_csv(file, encoding="cp932", nrows=3, header=None)
print(df.head(3))
