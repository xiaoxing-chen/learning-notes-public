- [1. 基础及语法](#1-基础及语法)
  - [a. 规范\&基础](#a-规范基础)
  - [b. 进阶概念](#b-进阶概念)
  - [c. 区别性概念](#c-区别性概念)
  - [d. 特殊语法](#d-特殊语法)
- [2. 奇技淫巧](#2-奇技淫巧)
- [3. 设计模式](#3-设计模式)
  - [创建型模式，类的创建](#创建型模式类的创建)
  - [结构型模式，多个类如何组织](#结构型模式多个类如何组织)
  - [行为型模式，类的方法实现](#行为型模式类的方法实现)

# 1. 基础及语法

## a. 规范&基础

* [官网高质量文档](https://docs.python.org/zh-cn/3/library/index.html)
* 字符串前缀`f、r、u、b`：分别代表格式化、原生字符串(无转义效果)、采用unicode编码、Python3中默认的str是Unicode类。
* [下划线各种用法](https://zhuanlan.zhihu.com/p/36173202)
  + `_foo`：仅提示作用，限类/方法内部使用，不能import出去；
  + `foo_`：仅规范作用，避免与关键字重复的命名方法；
  + `__foo`：解释器强制执行，避免被子类重写；
  + `__foo__`：Python特殊方法；
  + `_`：临时变量，或上个表达式的值。

* windows手动安装pip：从`https://pypi.org/project/pip/#files`下载`pip-21.1.1.tar.gz`解压，然后`python setup.py install`
* Conda
  + Linux用户组概念：方便管理权限，`/etc/groups`
  + conda安装在`/usr/local/anaconda3`目录下：
    - 流程：root权限、创建用户组、将`/usr/local/anaconda3`加入用户组、给该目录赋权限、添加用户进入、重启
    - 每个用户`~/.condarc`文件依然有效、或者修改`/etc/profile`作用于所有用户
  + [Windows Terminal启动，不能激活conda虚拟环境](https://www.jianshu.com/p/4e6d00d55506)，

## b. 进阶概念

* Python启动时，会分配`-5`到`256`的数值对象，——>通过创建小整数池的方式来避免小整数频繁的申请和销毁内存空间
* Python运算优先级：`== `高于`not`， `==`与`not in`相同
* `raise NotImplementedError`父类要求子类一定要实现的方法，否则不能调用。
* `for`循环工作原理：
  + 执行`in`后面对象的`dict.__iter__()`方法，得到一个迭代器对象`iter_dict`
  + 执行`next(iter_dict)`，将得到的值赋给`k`，然后执行循环体
  + 重复上一步，直至捕捉到异常`StopIteration`，结束循环
* `iterable, iterator, yield`与`iter(), next()`：
  + `[x**2 for x in range(5)]`得到列表，而`(x**2 for x in range(5))`得到iterator；
  + 包含`__iter__()`方法的对象属于iterable，而包含`__next__()`的iterable是iterator，e.g. 列表、元组、字典等属于iterable，而包含`yield`关键字的函数属于iterator；
  + 针对**iterable**对象：`iter(OBJ)`等价于`OBJ.__iter__`，会返回一个迭代器（`ITER`）；当`next(ITER)`时，才会去执行`__iter__()`方法；
  + 针对**iterator**对象：`iter(OBJ)`必须返回一个iterator（self or other iterator）；`next(ITER)`则是直接执行`__next__()`方法；
  + 针对**yield**函数（iterator）：`iter(FUN())`与`FUN()`是相同的iterator，`next(ITER)`是执行到第一个yield关键字，下次next则紧接着yield开始执行。
* python导入包的执行顺序：`from art.attacks.attack import Attack`
  + 因为`art`还未被导入，所以先去执行`art.__init__.py`，然后是`art.attacks.__init__.py`，依次往下，直至无`__init__.py`，则执行`attack.py`；
  + 各个文件，**“从高往下，先到先得”**，之前导入过的不需要再次导入。
  + `import`将后面的包、函数、变量名称加载到**当前环境**，作用域仅限当前环境。
* 遍历字典注意事项：
  + `for key in dict_:`等价于`for key in dict_.keys():`
  + `for val in dict_.values():`
  + 有序字典创建：`ordered_dict = collections.OrderDict()`，然后正常遍历
  + Python3.6 版本以后的 dict 是有序的
  + 注意：这里的有序，仅指按照插入循序，如果定义时赋值，则无序
* 格式化字符串：`fmt = '{:<10s}|' + '|'.join(['{:*^5.2f}'] * 10) + '\n'`，`fmt.format('header', *list(range(10)))`，其中`<,^,>`分别表示向左/居中/向右对齐

## c. 区别性概念

* [函数与方法区别](https://segmentfault.com/a/1190000009157792)：函数通过**FunctionObject**实现，方法通过**PyMethodObject**实现；其中方法还分为`Bound/Unbound` Method，前者实例化绑定，后者没有。
* `self & cls`：`self`表示实例本身、`cls`表示类本身；类的实例化，先调用`__new__`方法，返回该类的实例对象；这个实例对象即为`__init__`的第一个参数`self`，所以`self`是`__new__`的返回值。
* 注意`obj.sort()`和`sorted(obj)`的区别：前者原地修改内存，后者返回一个新的对象
* `open()`和`codecs.open()`的区别：后者可以指定编码，e.g. `with codecs.open(FILE, 'r', encoding='utf-8')`。针对复杂文本的读取，使用`open`会造成读取后的编码不统一
* `is`判断是否是同一个引用对象，`==`判断值是否相等

## d. 特殊语法

* `,`特殊用法
  + `t=np.zeros([2,3]),`函数/值后面加`,`结果为元组，不可修改；
  + `print("Hello","World")`：输出字符串连接，但默认添加分隔符空格，即`Hello World`。
* 变量初始化：`x = 16_00_0 = 16000`、`x = 16_0-00 = 160`
* `val.reshape(-1)`中参数`-1`代表按行优先展开为行向量

# 2. 奇技淫巧
* 高阶函数
  + `map(fn, Iterable)`，`reduce(fn, Iterable)`
  + `filter(fn, Iterable)`
  + `sort/sorted(Iterable, key=fn)`
* `items.sort(key=lambda x:x[1], reverse=True)`：列表排序，如由元组构成的列表，`x[1]`表示取第2个字段排序，`reverse`表示降序。
* numpy列表索引中，`None, ...`[作用](https://blog.csdn.net/Soly_semon/article/details/104691986)：`None`在所在位置增加维度，`...`省略前面所有`:`操作。
* `list_[:]`表示拷贝原数据，用`id(list_)`查看内存地址不一样。
* `assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)`，代码简写
* `a op1 b op2 c ... opn z`等价与`(a op1 b) and (b op2 c) and ... and (* opn z)`
* `argparse`传入`bool`型参数，均解析为`True`，如何解决：

```python
def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser.add_argument('--rand_init', type=str2bool, nargs='?', const=False, help='')
```

* 格式化输出字符串，建议使用`format`

```python
a = [1, 2]
out1 = '%.2f, %.2f'%(*a)			# 1. error
out2 = '{:.2f}, {:.2f}'.format(*a)	# 2. correctly unpacked
```

* 调试过程`is/is not 'str'`结果为真，运行过程中`is/is not 'str'`结果为假
* 反转列表或ndarray：`b = a[::-1]`或`lst.reverse()`，前者都有效，后者仅列表有效
* `vars(cls_name)`将类内变量转换为字典形式返回，e.g. `**vars(args)`，简化函数输入
* 装饰器（`decorator`）：

```python
import functools

# 1. parameter-free
def log(fn):
    @functools.warps(fn)
    def wrapper(*args, **kwargs):
        print('{:s} is called.'.format(fn.__name__))
        return fn(*args, **kwargs)
    return wrapper

@log    # equal to `self_fn = log(self_fn)`
def self_fn():
    pass

# 2. parameter-need
def log(params):
    def decorator(fn):
        @functools.warps(fn)
        def wrapper(*args, **kwargs):
            print('{:s} is called.'.format(fn.__name__))
            print('Params: {:s}'.format(str(params)))
            return fn(*args, **kwargs)
        return wrapper
    return decorator

@log('params')  # equal to `self_fn = log('params)(self_fn)`
def self_fn():
    pass
```

* 用`functools.partial`来封装函数，减少输入参数
* 用comperator进行排序，[参考](https://www.delftstack.com/howto/python/python-comparator/)

```python
from functools import cmp_to_key

def compare(x: Tuple[int, int], y: Tuple[int, int]) -> bool:
    # 首先根据elem[0]升序，如果相等则根据elem[1]降序
    if x[0] == y[0]:
        return y[1] - x[1]
    return x[0] - y[0]

arr.sort(key=cmp_to_key(compare))
```

* [位运算，转换英文字符大小写](https://labuladong.github.io/algo/di-san-zha-24031/shu-xue-yu-659f1/chang-yong-13a76/)

```python
('A' | ' ') = 'a'   # 大写转小写
('a' & '_') = 'A'   # 小写转大写
('b' ^ ' ') = 'B'   # 互转
('B' ^ ' ') = 'b'   # 互转
```

# 3. 设计模式
* [课程内容笔记-Github整理](https://github.com/ThanlonSmith/design-pattern)
* 接口：对用户隐藏底层细节
  + `raise NotImplementedError`实现：不调用不会报错
  + `ABCMeta, abstractmethod`实现：继承的子类必须实现

```Python
from abc import ABCMeta, abstractmethod

class Payment(metaclass=ABCMeta):
    @abstractmethod
    def pay(self, money):
        pass
class WeChatPay(Payment):
    def pay(self, money):
        print(f'{money}')
```

## 创建型模式，类的创建

* 简单工厂：只实现一个`PaymentFactory`，统一处理；[参考代码，用函数实现](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/speaker_model.py#L21)
* 工厂方法：每个`Payment`单独实现一个`PaymentFactory`，代码量大
* 抽象工厂：不同工厂有不同的产品组合，e.g. 手机包含手机壳、CPU、操作系统等元素，不同手机实例化自己需要的元素
* 建造者：在抽象工厂的基础上，着重于控制组装顺序，e.g. 具体产品、抽象Builder、具体Builder、指挥者
* 单例模式：父类实现`__new__`方法，判断`hasattr(cls, '_val')`，初始化`cls._val = super().__new__(cls)`或直接返回

## 结构型模式，多个类如何组织

* 适配器：类适配器使用多继承实现，对象适配器通过组合实现；[参考代码，功能有点像](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/runner.py#L75)
* 桥模式：不同维度的类任意组合，e.g. `Shape`和`Color`两个维度，组成`BlueCircle/GreenRectangle`

```Python
from abc import ABCMeta, abstractmethod
# 抽象接口
class Shape(metaclass=ABCMeta):
    def __init__(self, color):
        self.color = color
    @abstractmethod
    def draw(self):
        pass
class Color(metaclass=ABCMeta):
    def __init__(self, shape):
        self.shape = shape
    @abstractmethod
    def paint(self, shape):
        pass
```

* 组合模式：用户对单个对象和组合对象的使用具有一致性，e.g. PPT绘图中单个形状，任意嵌套组合的形状（多叉树）操作相同
* 外观模式：对外封装，提供统一接口，提高灵活性、安全性，e.g. 实现的电脑类含`run, stop`方法，内部包含`CPU+Disk+Memory`的启动和停止
* 代理模式：远程代理、虚代理、保护代理（e.g. 提供读权限）

## 行为型模式，类的方法实现

* 责任链模式：依次请求多个对象，使之组成类似“链表”形式，e.g. 请假：项目主管->部门主管->总经理
* 观察者模式：“发布-订阅”，一对多的依赖关系，e.g. 公众号发布新内容，所有订阅者都收到更新

```Python
# 抽象的发布者
class Notice:
    def __init__(self):
        self.observers = []
    def attach(self, obs):
        self.observers.append(obs)
    def detach(self, obs):
        self.observers.remove(obs)
    def notify(self):
        for obs in self.observers:
            obs.update(self)
# 具体发布者
class StaffNotice(Notice):
    def __init__(self, company_info):
        super().__init__()  # 调用父类对象声明observers属性
        self.__company_info = company_info
    @property
    def company_info(self):
        return self.__company_info
    @company_info.setter
    def company_info(self, info):
        self.__company_info = info
        self.notify()

staff_notice = StaffNotice('初始化公司信息')
staff_notice.company_info = '广播信息到订阅者'
```

* 策略模式：一系列算法封装起来，使之可以互相替换，e.g. 打车匹配司机：高峰时期用A算法，平时用B算法
* 模板方法模式：定义某操作的算法框架，以及一些钩子操作供子类实现，e.g. 窗口启动、绘制、结束的框架确定，但每个窗口实例的细节不一样

```Python
from abc import ABCMeta, abstractmethod
from time import sleep
# 抽象类
class Window(metaclass=ABCMeta):
    # start, repaint, stop -> 原子操作/钩子操作
    @abstractmethod
    def start(self):
        pass
    @abstractmethod
    def repaint(self):
        pass
    @abstractmethod
    def stop(self):
        pass
    def run(self):
        self.start()
        while True:
            try:
                self.repaint()
                sleep(1)
            except KeyboardInterrupt:
                break
        self.stop()
# 具体类
class MyWindow(Window):
    def __init__(self, msg):
        self.msg = msg
    def start(self):
        print('窗口开始运行！')
    def stop(self):
        print('窗口停止运行！')
    def repaint(self):
        print(self.msg)

MyWindow("Hello...").run()
```
