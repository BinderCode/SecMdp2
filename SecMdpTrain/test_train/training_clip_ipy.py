from fastai.vision.all import *
from fastai import *
import os
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
#如果是多GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #

os.chdir(sys.path[0])#使用当前目录作为根目录
print('sys.path[0]===',sys.path[0])
#os.chdir('..') #使用当前目录上一级作为根目录
print("当前绝对路径1=========",os.path.abspath('training_clip_ipy.py'))#查看当前绝对路径
#sys.path.append("..") #返回上一级目录
from self_supervised.multimodal.clip import *
import clip
from zero_optimizer import ZeroRedundancyOptimizer # only works with multi-gpu / distributed training
#torch.distributed.init_process_grop('nccl',init_method='file:///myfile',work_size=1,rank=0)
dist.is_initialized()
# if dist.is_available():
#     parser.add_argument("--backend", type=str, help="Distributed backend",
#                         choices=[dist.Backend.GLOO,
#                                     dist.Backend.NCCL, dist.Backend.MPI],
#                         default=dist.Backend.GLOO)
trainpath = Path("/host/CLIP_data/train2014/")               ########----------数据集需要
validpath = Path("/host/CLIP_data/val2014/")
annospath = Path("/host/CLIP_data/annotations/") 
#获取文件名称前就要加入ORAM写回一个假的文件
class TreeNode:
    def __init__(self):
        self.data = None

class MM_PathORAM:
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
        #print(f"Accessing file {file_path}")
        leaf = self.position_map[file_path]
        path = self._get_path(leaf)
        #print(f"Access path: {path}")

        retrieved_file = None
        for node_idx in path:
            node_data = self.tree[node_idx].data
            #print(f"Node {node_idx} data: {node_data}")
            if node_data == file_path:
                retrieved_file = node_data
                break

        if retrieved_file is None:
            print(f"Warning: could not retrieve file {file_path}")

        return retrieved_file
train_images = get_image_files(trainpath)  #递归地获取' path '中的图像文件，如果指定，仅在' folders '中。
valid_images = get_image_files(validpath)
len(train_images), len(valid_images)
print(len(train_images), len(valid_images))
caption_paths = annospath.ls().filter(lambda o: 'captions' in o.name)
import time
start_time = time.time()
fn2captions = {}        #读取标签的json文件
for p in caption_paths:
    caps = json.loads(open(p).read())
    id2fn = {o['id']: o['file_name'] for o in caps['images']}
    fn2cap = {id2fn[o['image_id']]: o['caption'] for o in caps['annotations']}
    fn2captions.update(fn2cap)

assert len(fn2captions) == (len(train_images) + len(valid_images))  #assert判断是否相等，不等就报错
all_images = train_images[:10000] + valid_images[:5000]; len(all_images),len(fn2captions)
oram = MM_PathORAM(len(all_images))
def read_image(fn): 
    #return PILImage.create(fn) 
    retrieved_file = oram.access(fn)
    if retrieved_file is not None and os.path.isfile(retrieved_file):  #能找到路径  然后返回路径图片
        return PILImage.create(retrieved_file)   
    else:
        print(f"Warning: file {retrieved_file} does not exist")
        return None
def read_image1(fn): #不调用ORAM
    return PILImage.create(fn) 
def read_text(fn): return fn2captions[fn.name]
def dummy_targ(o): return 0 
# 加载文件路径到PathORAM
for i, file_path in enumerate(all_images):
    leaf = i % oram.capacity
    oram.position_map[file_path] = leaf
    oram.tree[leaf + oram.tree_size // 2].data = file_path

clip_stats = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
clip_tokenizer = ClipTokenizer()    #将文本输入转化为数值型的输入
size,bs = 224,16 #batchsize------------------------------------

split_func = lambda fn: True if "val2014" in str(fn) else False  #判断是否是val2014数据集
dsets = Datasets(all_images, tfms=[read_image1, read_text, dummy_targ], 
                n_inp=2, 
                splits=FuncSplitter(split_func)(all_images))
item_tfms = [RandomResizedCrop(size, min_scale=0.9), clip_tokenizer, ToTensor()]

batch_tfms = [IntToFloatTensor, Normalize.from_stats(*clip_stats)]
train_dl = TfmdDL(dsets.train, shuffle=True, bs=bs, after_item=item_tfms, after_batch=batch_tfms, drop_last=True,num_workers=0)#更改num_workers=0只有一个主进程
# print("t",train_dl)             #t <fastai.data.core.TfmdDL object at 0x7f40a226c7c0>
# print("type",type(train_dl))    #type <class 'fastai.data.core.TfmdDL'>
valid_dl = TfmdDL(dsets.valid, shuffle=False, bs=bs*2, after_item=item_tfms, after_batch=batch_tfms,num_workers=0)#更改num_workers=0只有一个主进程，去除多进程
dls = DataLoaders(train_dl, valid_dl, device=default_device())  #数据集   DataLoaders存放数据集

clip_trainer_cb = CLIPTrainer()
cbs = [clip_trainer_cb]
opt_func = ranger
arch = 'vitb32'
do_finetune = True
use_grad_check = True
grad_check_nchunks = 2
finetune_modelname = '/host/ViT-B-32.pt'    #-------------------------预训练模型需要放进去

vitb32_config_dict = vitb32_config(size, clip_tokenizer.context_length, clip_tokenizer.vocab_size)#返回模型参数信息（键值对）
clip_model = CLIP(**vitb32_config_dict, checkpoint=use_grad_check, checkpoint_nchunks=grad_check_nchunks)
#把vitb32_config_dict这个dict字典的所有key-value用关键字参数传入到函数的**kwargs
if do_finetune:
    clip_pretrained_model, _ = clip.load(finetune_modelname, jit=False)
    
    clip_model.load_state_dict(clip_pretrained_model.state_dict())  #为模型添加参数  jit？？？

learner = Learner(dls, clip_model, loss_func=noop, cbs=clip_trainer_cb, opt_func=opt_func)#,metrics=[tuple_accuracy]
learner.to_fp16()

lr,wd,epochs=1e-5,1e-6,5  #学习率  wd训练时权重衰减 迭代次数

csv_logger = CSVLogger(fname="metrics.csv")
learner.fit_flat_cos(epochs, lr,wd=wd, pct_start=0.25,cbs=[csv_logger]) #训练  ORAM-4m38.4s  5m11.3s  5min23.8  5min12.3    正常~4m44.3s

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Program running time: {elapsed_time:.2f} seconds")  #265.52 seconds    264.99 seconds
