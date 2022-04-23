import gdown

# https://drive.google.com/file/d/1VRBiCVJvCUoFImkUWjKWYfPCCi3Vl42L/view?usp=sharing
id = "1VRBiCVJvCUoFImkUWjKWYfPCCi3Vl42L"
output = "../../data/ip102.tar"
gdown.download(id=id, output=output, quiet=False)