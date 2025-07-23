import os
import shutil
from tqdm import tqdm

def flatten_mnistm_split(source_dir, dest_dir, label_file_path):
    os.makedirs(dest_dir, exist_ok=True)

    # Count total files for progress bar
    total_files = sum(len(files) for _, _, files in os.walk(source_dir))

    with open(label_file_path, 'w') as label_file, tqdm(total=total_files, desc=f"Processing {source_dir}") as pbar:
        for label in sorted(os.listdir(source_dir)):
            class_dir = os.path.join(source_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                src = os.path.join(class_dir, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy(src, dst)
                label_file.write(f"{fname} {label}\n")
                pbar.update(1)

# ✅ Set correct dataset root
mnist_m_root = '../data/MNIST-M'

# 🔄 Flatten training set
flatten_mnistm_split(
    source_dir=os.path.join(mnist_m_root, 'training'),
    dest_dir='../data/mnist_m/mnist_m_train',
    label_file_path='../data/mnist_m/mnist_m_train_labels.txt'
)

# 🔄 Flatten test set
flatten_mnistm_split(
    source_dir=os.path.join(mnist_m_root, 'testing'),
    dest_dir='../data/mnist_m/mnist_m_test',
    label_file_path='../data/mnist_m/mnist_m_test_labels.txt'
)
