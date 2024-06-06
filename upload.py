from huggingface_hub import upload_folder

# 本地保存模型的文件夹路径
local_folder = './experiments/240602_3'

# 在Hugging Face上创建的仓库ID，格式为：username/repo_name
repo_id = "harrym111/animeface_gender"

# 上传文件夹
upload_folder(
    folder_path=local_folder,
    path_in_repo="",
    repo_id=repo_id,
    repo_type="model"
)