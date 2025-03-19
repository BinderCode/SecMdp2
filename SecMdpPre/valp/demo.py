import torch
import clip.clip
from PIL import Image
import sys
import os   
import numpy as np
os.chdir(sys.path[0])#使用当前目录作为根目录
print("当前目录是~",sys.path[0])
# print(torch.cuda.device_count())#查看GPU数量
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B-32.pt", device=device) #加载模型
import time
start_time = time.time()
labels_cifar = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
          "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
          "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
          "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
          "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
          "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
          "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
          "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
          "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
          "possum", "rabbit", "raccoon", "ray", "road", "rocket" "rose", "sea", "seal",
          "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
          "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
          "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
          "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
labels_VOC=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
          "chair", "cow"]
labels_COCO=["airplane", "apple", "banana", "bear", "bicycle", "bird", "bus","car",  
"chair", "cow"]
num=[]#用于不同的类别计数
import time
start_time = time.time()
for j in range(len(labels_COCO[:10])):    #list1[start:stop:step]
    text_list=["this is a "+labels_COCO[j], "this is a dog", "this is a table"]
    #rootdir="./VOC_10/"
    rootdir="./COCO_10/"
    #rootdir="./test-10/"
    imagelist_dir = os.listdir(rootdir) #文件夹名称每一类一个文件夹
    imagelist_dir.sort()    #文件夹按顺序排序   不然会乱序  与lable不对应
    print("imagelist_dir=",imagelist_dir)  
    imagelist=os.listdir(rootdir+'/'+imagelist_dir[j])#图片名称
    count=0#用于每类数据计数
    for i in range(len(imagelist[:100])):
        image = preprocess(Image.open(rootdir+'/'+imagelist_dir[j]+'/'+imagelist[i])).unsqueeze(0).to(device)
        text = clip.tokenize(text_list).to(device)
        #下午找数据集，测试，计算准确率
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        list_probs = probs[0].tolist()  #list化
        confidence=max(list_probs)      #取最大的置信度    求最大值
        if confidence>0.5 :  #求最大值的索引
            if list_probs.index(max(list_probs))==0:
                count=count+1
    num.append(count)
print(num)
end_time = time.time()
elapsed_time = end_time - start_time   #4000S  一小时才能完成
print(f"Program running time: {elapsed_time:.2f} seconds") 
