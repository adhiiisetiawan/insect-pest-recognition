import tarfile
from tqdm import tqdm

tar = tarfile.open("./data/raw/ip102.tar")
tqdm(tar.extractall(path="./data/processed"))
