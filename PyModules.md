- [Python内置库](#python内置库)
  - [contextlib](#contextlib)
  - [itertools](#itertools)
  - [natsort](#natsort)
  - [pathlib](#pathlib)
- [深度学习常用库](#深度学习常用库)
  - [tensorflow](#tensorflow)
  - [numpy](#numpy)
  - [pickle](#pickle)
  - [tensorboard](#tensorboard)
  - [einops](#einops)
- [研究方向相关](#研究方向相关)
  - [art](#art)
  - [dlib](#dlib)
  - [opencv-python (cv2)](#opencv-python-cv2)
- [接口类/工具库](#接口类工具库)
  - [dataclasses](#dataclasses)
  - [hydra](#hydra)
  - [bs4 (Beautiful Soup)](#bs4-beautiful-soup)
  - [fire](#fire)
- [多进程加速库](#多进程加速库)
  - [multiprocessing (mp)](#multiprocessing-mp)
  - [concurrent.futures](#concurrentfutures)

# Python内置库

## contextlib

* 上下文管理器通过`__enter__`和`__exit__`两个方法来实现，Python标准库`contextlib`提供了更简便的使用方法
* decorator：`contextlib.contextmanager`

## itertools

* `list(islice(cycle(colors), int(max(label_list) + 1)))`：`cycle`用于重复，`islice`用于截取
* `accumulate(data)`：用于获取前缀和

## natsort

* windows资源管理器，显示路径（自然顺序），[python实现](https://www.cnblogs.com/wan-deuk/p/15640552.html)

```python
from natsort import natsorted
from natsort.ns_enum import ns
path_list = natsorted(path_list, ns.PATH)
```

## pathlib

* 完美替代`os.path`，[面向对象使用文件](https://docs.python.org/zh-cn/3/library/pathlib.html#pure-paths)

```Python
from pathlib import Path

# 1. join, dirname, basename
file = Path("/home/user/demo/foo.py")
dir_ = file.parent   # file.parents[0]
filename = file.name
new_file = dir_ / "new_foo.py"
new_file = dir_.joinpath('new', 'foo.py')
file.with_name("FOO.txt")   # /home/user/demo/FOO.txt
file.with_stem("FOO")       # /home/user/demo/FOO.py
file.with_suffix(".txt")    # /home/user/demo/foo.txt

# 2. exists, mkdirs, rmdir
file.exists()
file.parent().mkdir(parents=True, exist_ok=True)
file.parent().rmdir()   # 必须是空文件夹
file.absolute()         # 绝对路径
file.resolve()          # 绝对路径，解析符号链接

# 3. glob, match, listdir
dir_.rglob("*.py")          # dir_.glob("**/*.py")
dir_.match("demo/*.py")     # 遵循大小写
for child in dir_.iterdir(): print(child)

# 4. open, touch, write
with file.open(mode='r', encoding='utf8') as fout:
    ...
file.touch(exist_ok=True)
file.write_text(data='foo', encoding='utf8')
```

# 深度学习常用库

## tensorflow

* 模型的保存和恢复：[Not Found Error]，用`pywrap_tensorflow`读取`.ckpt`文件变量名与模型变量名对比
  + `selfname.ckpt.data-0000-of-0001、selfname.ckpt.meta、selfname.ckpt.index`，只需输入`selfname.ckpt`，不用输入后缀；如果有`checkpoint`文件，里面包含了最近一次的模型信息
  + `Saver, Restore`，

## numpy

* `np.array`和`np.asarray`的区别：两者都可以将数据转换为ndarray，但当数据源仍是ndarray时，`np.array`会产生一个副本，而`np.asarray`只产生一个引用
* `np.savez()`

```python
# 非结构化数据保存
str_ = 'abc'
arr_ = np.array([[1, 2], [3, 4]])
dict_ = {'a' : 1, 'b': 2}
np.savez('SAVE_PATH/filename.npz', st= str_, ar = arr_, dic= dict_)
# load
data = np.load('SAVE_PATH/filename.npz')
_str = [data['st'], data['st'][()]]		# equal
_arr = [data['ar'], data['ar'][()]]		# equal
_arr0 = [data['ar'][0], data['ar'][()]][0]		# equal
_dict = [data['dic'], data['dic'][()]]			# equal
_dicta = [data['dic']['a'], data['dic'][()]['a']]	# error, correct
```

* 高斯概率分布函数及其反函数
  + `scipy.stats.norm(0, 1).cdf()`
  + `scipy.stats.norm(0, 1).ppf()`
* `import numpy.ma as ma`
  + `MaskedArray`的作用：对于mask掉的值不参与运算
  + 定义：`ma.array(data=val_, mask=mask_)` / `ma.masked_array()data=val_, mask=mask_`

## pickle

* 把Python对象直接保存到文件里，而不需要先把它们转化为字符串再保存，也不需要用底层的文件访问操作，直接把它们写入到一个二进制文件里
* `pickle.dump(OBJ, open(f, 'wb'))` & `pickle.load(open(f, 'rb'))`
* `pickle.dumps()` -> bytes字符串 -> `pickle.loads()`
* serializing & de-serializing：反序列化时，需要能够读到其数据类型，e.g. 自定义类

## tensorboard

* 基本用法

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=LOG_DIR)
writer.add_scalar('train/loss', LOSS_VALUE, global_step=CUR_ITER)
# 同一个main_tag的scalar，绘制在一个figure里面
writer.add_scalars(MAIN_TAG, TAG_SCALAR_DICT, global_step=CUR_ITER)
# 将values先展成一维向量，然后再用tensorflow默认的bins，绘制直方图
writer.add_histogram(TAG, VALUES, bins=BINS,global_step=CUR_ITER)
```

* `BINS -> [tensorflow, auto, ...]`, [参考](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges)
* 启动：`tensorboard --logdir LOG_DIR --port 9999`，服务器需要在VSCode终端上启动（由于本地端口和服务器端口需要通过ssh映射）

## einops

* [官方文档](https://einops.rocks/)，[知乎](https://zhuanlan.zhihu.com/p/342675997)
* 提供易读可靠、灵活强大的张量操作符，同时支持`numpy, pytorch, tensorflow`
* 主要提供`rearrange, reduce, repeat`三个方法，进而实现`stacking, reshape, transposition, squeeze/unsqueeze, repeat, tile, concatenate, view`等操作

# 研究方向相关

## art

* 如果将`metaclass=input_filter`，改为`input_filter`，则只会调用`abc.ABCMeta.__new__ `进行类实例化，而不会实例化input_filter，即不会进入`input_filter.__init__`

```python
class input_filter(abc.ABCMeta):
    def __init__(cls, name, bases, clsdict):
        def make_replacement(fdict, func_name):
        # ...
        replacement_list = ["generate", "extract"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)

class Attack(abc.ABC, metaclass=input_filter):
```

## [dlib](http://dlib.net/)

* 开源C++ Toolkit：包含许多深度学习算法和工具，e.g. 人脸识别、人脸检测
* 人脸的68点的关键点分布如下：

```json
{
    IdxRange jaw;       // [0 , 16]
    IdxRange rightBrow; // [17, 21]
    IdxRange leftBrow;  // [22, 26]
    IdxRange nose;      // [27, 35]
    IdxRange rightEye;  // [36, 41]
    IdxRange leftEye;   // [42, 47]
    IdxRange mouth;     // [48, 59]
    IdxRange mouth2;    // [60, 67]
}
```

* 默认安装使用CUDA，使用cpu安装dlib：`python setup.py install --no DLIB_USE_CUDA`

## opencv-python (cv2)

* 读取到的维度顺序是`(H, W, C)`，其中通道按照`BGR`
* 读取视频

```python
video_frames = []
cap = cv2.VideoCapture(path)
while(cap.isOpened()):
    ret, frame = cap.read() # (H, W, C), BGR
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        video_frames.append(frame)
    else:
        break
cap.release()
```

* 写入视频

```python
video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(path, video_fourcc, fps, (height, width))
for roi in mouth_roi_list:
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    video_writer.write(roi)
video_writer.release()
```

# 接口类/工具库

## [dataclasses](https://zhuanlan.zhihu.com/p/60009941)

* 提供一个装饰器`@dataclass`和一些函数，用于自动添加特殊方法，如`__init__()`
* 使用`filed`来添加额外字段的附加信息
* 下面代码来自`fairseq.dataclass.configs.py`

```python
@dataclass
class DistributedTrainingConfig(FairseqDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    # ...

@dataclass
class FairseqConfig(FairseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None
```

## [hydra](https://hydra.cc/docs/intro/)

* 层次化的配置，方便从命令行覆盖，`fairseq`开源工具中使用到（但还没看懂）

## bs4 (Beautiful Soup)

* [bs4官方文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* `html`的基本结构，参见[菜鸟教程](https://www.runoob.com/html/html-tutorial.html)：
  + `<html>, <head>, <body>, <title>`，`<!DOCTYPE html>`
  + `<h1>, ..., <h6>, <p>, <div>, <span>`：标题，段落，小节（块级），内联元素
  + `<ul>, <ol>, <li>`：无序列表，有序列表，项目
  + `<a href="http...">`：链接，除此之外还有`<img>, <table>, <script>`等
  + 属性包括：`href, class, id, style`
* bs4解析html

```python
from bs4 import BeautifulSoup

with open('./ISCA Archive.html', 'rb') as frb:
    html = frb.read()
    bs = BeautifulSoup(html, 'html.parser')
```

* `find_all()`，[参见](https://blog.csdn.net/YZXnuaa/article/details/97002132)

```python
tags = bs.find_all('div', attrs={"class":"w3-card w3-round w3-white"})
```

* `bs4.element.Tag`一些方法及属性：
  + `tag.contents, tag.children, tag.descendants`
  + `tag.string, tag.text`
* 获得文本内容：`tag.text`；获得属性内容：e.g. `tag[href]`

## fire

* [子命令、命令组、属性访问三大功能](https://zhuanlan.zhihu.com/p/100459723)

```Python
import fire

def add(x, y): return x + y
def multiply(x, y): return x * y
if __name__ == "__main__":
    fire.Fire({'add': add, 'mul': multiply})
# python demo.py mul 10 20 -> 200

class Calculator:
    def __init__(self, offset=0):
        self._offset = offset
    def add(self, x, y): return x + y + self._offset
    def multiply(self, x, y): return x * y
if __name__ == "__main__":
    fire.Fire(Calculator)
# python demo.py add 10 20 --offset=1 -> 31
```

```Python
import fire

class Attacker:
    def generate(self): return 'Generating AEs ...'
class Defenser:
    def run(self, name): return f'Defenser: {name}'

class Pipeline:
    def __init__(self):
        self.attack = Attacker()
        self.defense = Defenser()
    def run(self):
        self.attack.run(); self.defense.run()

if __name__ == '__main__':
    fire.Fire(Pipeline)
# python demo.py run
# python demo.py defense run lmd
```

# 多进程加速库

## multiprocessing (mp)

* 多进程情况下，子进程出错，但主进程不能结束；[参考](https://segmentfault.com/q/1010000017216079)

```python
import signal
def throw_error_exit_all(ex):
    # 由于开启的多个进程属于一个进程组，kill进程组即可
    print(ex.__cause__)
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

pool = mp.Pool(njobs)
data = [list(range(10)) for _ in range(njobs)]
for i in range(njobs * 2):  # 任务数和进程池大小，不一定要一样
    pool.apply_async(
        fn, args=(data[i % njobs],)
        error_callback=throw_error_exit_all # 子进程出错时，执行回调函数
    )
```

* `Process`开启多进程

```python
worker_list = []
for j in range(njobs):
    p = Process(target=fn, args=(data[i],))
    p.daemon = True
    worker_list.append(p)
    p.start()
for p in worker_list:
    p.join()
```

* [多进程基础知识](https://www.jianshu.com/p/a7a2ee7c5463)
  + `Process`是自己管理进程；`Pool`是开启一个固定大小的进程池，将所有任务放在一个进程池中，由系统调度
  + `Process`参数：`daemon=True`表示主进程结束后强制结束；`p.join()`表示阻塞直到运行结束；`p.is_alive()`可以用来判断子进程是否结束
  + `Pool`参数：`apply_async`表示异步非阻塞；`apply`表示阻塞，一个个运行

## concurrent.futures

* [Python官方文档](https://docs.python.org/zh-cn/3.11/library/concurrent.futures.html)、[博客](https://www.cnblogs.com/huchong/p/7459324.html)
* `concurrent.futures`提供`ThreadPoolExecutor`和`ProcessPoolExecutor`用于创建线程池、进程池
* `wespeaker`代码示例（使用`map`），`examples/voxconverse/v1/diar/cluster.py`

```python
import concurrent.futures as cf
# cf.ProcessPoolExecutor(max_workers=N)
with cf.ProcessPoolExecutor() as executor, open(args.output, 'w') as f:
    for (subsegs, labels) in zip(subsegs_list,
                                    executor.map(cluster, embeddings_list)):
        [print(subseg, label, file=f) for (subseg, label) in zip(subsegs, labels)]
```

* 代码示例（不使用`map`），[参考博客](https://www.cnblogs.com/huchong/p/7459324.html)

```python
import concurrent.futures as cf
executor = cf.ProcessPoolExecutor()
obj_list = []
for data in data_list:
    obj = executor.submit(fn, data) # 返回一个future实例，用obj.result()获取结果
    # 添加回调函数：输入是futuer示例，用in_obj.result()得到结果
    # executor.submit(fn, data).add_done_callback(callback_fn)
    obj_list.append(obj)
executor.shut_down()    # 等同于multiprocessing中，close和join一起使用
results = [o.result() for o in obj_list]
```
