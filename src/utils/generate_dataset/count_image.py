import os

check_file_path = "./data/processed/ip102_v1.1/images/val/102 Cicadellidae"
print(f"Train class 102 is {len(os.listdir(check_file_path))}")