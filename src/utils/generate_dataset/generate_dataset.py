import shutil, os, re
from tqdm import tqdm


def create_label_folder(directory):
    path = f'./data/processed/ip102_v1.1/images/{directory}'

    x = []
    f = open("./data/processed/classes.txt", "r")
    try:
        for text in f:
            if text.strip():
                s = re.sub(r"\t", "", text).rstrip()
                label = s.title()
                list_label = label.splitlines()
                for folder in list_label:
                    x.append(folder)
                    os.makedirs(os.path.join(path,folder), exist_ok=True)
    finally:
        f.close()
    return x


def move_file(filename):
    if filename == "val.txt":
        path = "./data/processed/ip102_v1.1/images/val"
        file = open("./data/processed/ip102_v1.1/val.txt", "r")
    elif filename == "train.txt":
        path = "./data/processed/ip102_v1.1/images/train"
        file = open("./data/processed/ip102_v1.1/train.txt", "r")
    elif filename == "test.txt":
        path = "./data/processed/ip102_v1.1/images/test"
        file = open("./data/processed/ip102_v1.1/test.txt", "r")
    
    try:
        for data in tqdm(file, desc=str(file)):
            if data.strip():
                reg = re.sub(r"\n", "", data)
                res = reg.split(" ")
                for kelas, folder in zip(range(102), x):
                    if res[1] == f"{kelas}":
                        shutil.move(f'./data/processed/ip102_v1.1/images/{res[0]}', f'{path}/{folder}/')
    finally:
        file.close()


dir_list = ["train", "test", "val"]
for directory in dir_list:
    os.makedirs(f'./data/processed/ip102_v1.1/images/{directory}', exist_ok=True)
    x = create_label_folder(directory)
    move_file(f"{directory}.txt")