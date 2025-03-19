import torch
import clip
from PIL import Image
import os
import math
import random
import time
import os
import sys

# TreeNode和ReadOnlyPathORAM类定义保持不变
class TreeNode:
    def __init__(self):
        self.data = None

class ReadOnlyPathORAM:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_depth = math.ceil(math.log2(capacity + 1))
        self.tree_size = 2 ** (self.tree_depth + 1) - 1
        self.tree = [TreeNode() for _ in range(self.tree_size)]
        self.position_map = dict()

    def _get_path(self, leaf):
        path = []
        node_idx = leaf + self.tree_size // 2
        while node_idx >= 0:
            path.append(node_idx)
            node_idx = (node_idx - 1) // 2
        return path

    def _is_leaf(self, idx):
        return idx >= self.tree_size // 2

    def _get_random_leaf(self):
        return random.randint(0, self.capacity - 1)

    def access(self, file_path):
        leaf = self.position_map[file_path]
        path = self._get_path(leaf)

        retrieved_file = None
        for node_idx in path:
            node_data = self.tree[node_idx].data
            if node_data == file_path:
                retrieved_file = node_data
                break

        if retrieved_file is None:
            print(f"Warning: could not retrieve file {file_path}")

        return retrieved_file

# 初始化和填充ReadOnlyPathORAM的函数
def init_path_oram(file_paths, capacity):
    oram = ReadOnlyPathORAM(capacity)
    for i, file_path in enumerate(file_paths):
        #leaf = oram._get_random_leaf()
        leaf = i % oram.capacity
        oram.position_map[file_path] = leaf
        node_idx = leaf + oram.tree_size // 2
        oram.tree[node_idx].data = file_path
    return oram

def main():
    os.chdir(sys.path[0])#使用当前目录作为根目录
    print("当前目录是~",sys.path[0])
    # CLIP模型和设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B-32.pt", device=device) #加载模型

    # 目标标签
    labels_COCO = ["airplane", "apple", "banana", "bear", "bicycle", "bird", "bus", "car", "chair", "cow"]  # ... [其他标签]

    # 根目录和初始化ORAM 修改路径
    rootdir="./COCO_10/"
    imagelist_dir = os.listdir(rootdir)
    imagelist_dir.sort()

    file_paths = []
    for j in range(10):  # 仅处理前10个目录
        subdir = imagelist_dir[j]
        files = os.listdir(os.path.join(rootdir, subdir))[:100]  # 每个类别取前100个文件
        file_paths.extend([os.path.join(rootdir, subdir, file) for file in files])

    oram = init_path_oram(file_paths, len(file_paths))

    # 评估每个标签
    num = []
    start_time = time.time()

    for j in range(len(labels_COCO)):  # 仅处理前10个标签
        text_list = ["this is a " + labels_COCO[j], "this is a dog", "this is a table"]
        count = 0

        for i in range(100):  # 每个类别处理前100个图像
            #file_path = oram.access(file_paths[j * 100 + i])  # 计算在file_paths中的索引
            file_path = file_paths[j * 100 + i] # 计算在file_paths中的索引
            retrieved_file_path = oram.access(file_paths[j * 100 + i])

            if retrieved_file_path is None:
                print(f"File not found in ORAM: {file_paths[j * 100 + i]}")
                continue  # Skip this file

            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            text = clip.tokenize(text_list).to(device)

            with torch.no_grad():
                # image_features = model.encode_image(image)
                # text_features = model.encode_text(text)
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            list_probs = probs[0].tolist()
            confidence = max(list_probs)
            if confidence > 0.5:
                if list_probs.index(max(list_probs)) == 0:
                    count += 1

        num.append(count)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(num)
    print(f"Program running time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
