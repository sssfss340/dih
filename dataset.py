import sys
from multiprocessing import Pool, cpu_count
from os import path,replace,remove
import csv
from tempfile import NamedTemporaryFile

dir_name = path.dirname(path.realpath(__file__))
datasets=[
    "Wednesday-14-02-2018.csv",
    "Thursday-15-02-2018.csv",
    "Friday-16-02-2018.csv",
    "Tuesday-20-02-2018.csv",
    "Wednesday-21-02-2018.csv",
    "Thursday-22-02-2018.csv",
    "Friday-23-02-2018.csv",
    "Wednesday-28-02-2018.csv",
    "Thursday-01-03-2018.csv",
    "Friday-02-03-2018.csv",
]

def process_file(name):
    file_path=f"{dir_name}\\{name}"
    if not path.isfile(file_path):
        return f"{name} 不存在"
    try:
        with  open(file_path,mode="r",encoding="utf-8") as src_file, \
              NamedTemporaryFile(mode='w',delete=False, dir=dir_name, newline='', encoding='utf-8') as temp_file:
            reader = csv.reader(src_file)
            writer = csv.writer(temp_file)
            for row in reader:
                if row[90]!="-1":
                    row[89]="BENIGN"
                writer.writerow(row)
        replace(temp_file.name,file_path)
    except:
        remove(temp_file.name)
        raise
    return f"{name} 處理完成"

if __name__ == "__main__":
    modify=input("修改前請確保所有csv檔都與此python檔在同資料夾且未曾改名\n開始修改? (y/n) ")
    if modify!="y":
        sys.exit()

    with Pool(processes=min(len(datasets), cpu_count())) as pool:
        for result in pool.imap_unordered(process_file, datasets):
            print(result)
