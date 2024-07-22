import os


if __name__ == "__main__":
    root_cluster = "/netscratch/franzreb/accent_info/"
    root_local = "/Users/cafr02/repos/accent_info/"
    ckpt_dir = "checkpoints"
    ckpt_file = "last.ckpt"
    folders = [
        "logs/asr/train/version_8",
        "logs/asr/train/version_9",
        "logs/ensemble/train/binary/b15/DAT/version_19",
        "logs/ensemble/train/binary/b11/DAT/version_2",
    ]

    for folder in folders:
        local_dir = os.path.join(root_local, folder, ckpt_dir)
        os.makedirs(local_dir, exist_ok=True)
        cluster_file = os.path.join(root_cluster, folder, ckpt_dir, ckpt_file)
        local_file = os.path.join(local_dir, ckpt_file)
        print(f"\n\nscp {local_file} dfki-hpc:{cluster_file}")
