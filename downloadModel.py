from huggingface_hub import snapshot_download

dir1 = "./models/pythia-70m"
dir2 = "./models/pythia-2.8b"
dir3 = "./models/pythia-410m"
print("下载70M模型")
snapshot_download(
    repo_id="EleutherAI/pythia-70m",
    local_dir=dir1,
    local_dir_use_symlinks=False,
)
print("下载2.8B模型")
snapshot_download(
    repo_id="EleutherAI/pythia-2.8b",
    local_dir=dir2,
    local_dir_use_symlinks=False,
)
print("下载410m模型")
snapshot_download(
    repo_id="EleutherAI/pythia-410m",
    local_dir=dir3,
    local_dir_use_symlinks=False,
)

print(f"模型下载完成")