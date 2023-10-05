实验过程

没有GPU: 可以申请, 用colab. 让学长帮我申请了GPU.

尝试DQN 问题1 : colab没有显示器.

尝试一个DRL 可以, 它也是用策略梯度

服务器上装个x,   用ssh转发一下

ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory

方法1:  我把代码抄一遍用pytorch复现一遍.


### 尝试DL2

1. 尝试 train.py, 失败了. 无法连接 tensorboard, localhost6006 refuse to connect . 在hku服务器上的问题是, python2 有很多不兼容, python3 没有numpy
2. 然后卸载了安装 tf1.13.1 , 显示**This page isn’t working right now** **e0844decf715** didn’t send any data.
3. There are not any runs in the log folder.

4. pip21.1.2 找不到tensorflow. pip10.0.1 可以找到但是显示没有.   

完成了优化器, checkpoint. 企图分离agent.

为什么训练过程这么长, over 150 LOC.

8.13

他每个文件都有单元测试的. 我测试了一下, 他没有ali trace , 我改成了随机.

https://adaptdl.readthedocs.io/en/latest/

[petuum/adaptdl: Resource-adaptive cluster scheduler for deep learning training. (github.com)](https://github.com/petuum/adaptdl)

adaptdl 是用两层, 可以试试能不能用DRL来做.

复现一下pollux,

table2 ,  99% tile是什么意思? 就是超过99%的

8.14

**怎么做实验? DRL没有env怎么办?** 

但是不是让你用k8s实际部署一个job, 到machine跑完, 测时间, 而是根据trace 模拟出来大概要多久,这样你就可以在本地训练大量数据

最好能做个模型在这些环境跑一跑,咱们就知道你的方案可不可行

simulation就是说你给一个job的placement和他的资源数量 告诉你这个job跑了多久,不是实际在机器跑 而是直接给你一个时间. 就是RL的env

**trace读了什么?**

我下午做一个45分钟就可以知道. 

可以参考它的simulator , 它fitting了现有的model, 这个已经搞懂了. 就是一个函数, 两个自变量 numps和num worker, 一个因变量speed. 用scipy.interpolate.Rbf 拟合.  

8月14日 , 目标搞懂它输入资源放置和输出速度的关系. 给一个job 的资源, 告诉你reward. reward是运行的epoch数/总epoch. 在哪里告诉你的?

get trace  做了什么?

就是生成一些job,给job赋值, job有很多属性

1. batch size 
2. 带宽( 为什么每类job的带宽是确定的呢? )
3. speed func(在job.py中用来得到epoch) 
4. 考虑了误差的 总epoch

trace是有100个job list的字典.  一个job list 10个job

为什么要是字典? TS是什么意思? 就是timeslot

好像看完了。 感觉不对， device placement的 模拟做不来？ 因为没有device 的trace ， co-located ， 这个trace不能用？  device placement的 trace 需要哪些参数？ 

然后明天要干嘛

 8.19日

debug一晚上, 

1.  一个一模一样的代码, 可以执行, 但是comparsion不能执行.   解决方法: 注释掉一些
2. 发现是trace有问题,   可能是pycharm跑, 输出到stdout之外? 终端试一试 ,还是不行 
3. possess =1 可以, 是电脑卡了吗?  也不至于, possess =2也会卡,  而且一模一样的40也没问题. 
4. imoprt trace 就爆炸了.  把trace 类注释,  发现import 就会爆炸 ,其实可以用相对路径来标记文件是不是自己写的. 
5. 递归寻找,  二分寻找.  二分注释,  找到出问题的那一行.  40个子进程要40个构造函数, 在fit中加入print, 确认是否每个进程执行一次,  是执行了40次, 而且会出错,  看看内存, 没有问题
6. 因为没有封装到函数, 所以执行了很多次, pycharm可以试试 wsl. 封装起来就可以正常运行了.

8.20

1. dict() python2 有问题. python3 需要  list( values()) , 因为 values()返回不是pickleable可序列化对象, 是一个view视图对象, 不能修改.
2. 做 , 验证集是怎么用的? 
3. rl_env 很难调试, def test():我随便写的, 不知道怎么写不会出错. 

8.25 

写了5天ppt和 research statement. 老师还是问我要 result,

写了一晚上, 写到神经网络有点难顶, 看不懂tflearn的连接, 还有一堆条件连接. 

8.27

感觉这两天要是搞出来的话,还是得把源代码跑通比较现实, 配环境头痛.

ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory

py3.7降低到3.5 . 空间不足不能安装, 问学长. 

colab试试. 拿不出东西来, 用它的源代码头痛是配置tensorflow的环境, 自己pytorch是神经网络不会.

卸载tesnorflow, 安装tensorflow 1.13.1 

安装python2.7 好像colab也自带了.

```text
!apt-get install python2.7
!python2 -m pip uninstall tensorflow
!python2 -m pip install tensorflow-gpu==1.13.1
!python2 -m pip list
!python2 train.py
```

安装了显示没有tensorflow, 一种解决方法是需要配置tensorflow的路径到sys.path中。

那要不还是用python3跑, 我之前也改过代码了.

```python
!python -m pip uninstall tensorflow
!python -m pip install tensorflow-gpu==1.13.1
!pip install tflearn
!python train.py
```

不知道它到底train了个啥. 

改了一下好像可以sl了. 但是tensorboard看不到,我得看看有哪些指标. 

1. 训练速度, sl 弄不出来, 因为是根据标签一样不一样来判断的. 

2. 应该可以打印loss的.监督学习.   感觉也挺简单的, 就是很多很多问题, 要做的快很难. 

   比如这里step好像卡住了. 不知道为啥 train.py:456 INFO: len(self.completed_jobs)25 train.py:461 INFO: len(self.memory)8192 然后就卡住了. 应该是 \# pull latest weights before training 出问题了.

   没有get到weight

3. comparsion的线程池不对, 我忘了我之前pytorch写的好像也出错了忘记哪里出错了, 哦,好像是因为那个. import.  但是代码改了还是不对, 封装起来了还是跑不出来. 不能finished all task

4. copy_reg.pickle(types.MethodType, _pickle_method) 不知道是不是这个没有封装好, 这又是个啥啊.

   ```
   try:
       import copy_reg
   except:
       import copyreg as copy_reg
   ```

  9.1  完成了val_loss

我不知道为啥sl 还要 val_jmr? 算一个loss不就行了吗? 为啥要知道reward? 

好像是因为要训练一会儿,  然后validation做一下DRL决策看看对不对. val_jmr 需要rl_step, 我不明白 env.step(output)的 masked_output = np.reshape(output[0] * mask, (1, len(mask)))  应该怎么reshape.

9.2

我先跳过test, 先看看下面能不能不计算这个jmr. 

jmr , np.reshape(output[0] * mask, (1, len(mask))) 

想加一个tensorboard.

`tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, *scope*=*self*.scope)`

可以获得各个weight和bisas. 返回一个list. pytorch没有总的,我就改成list装各个变量把.

https://blog.csdn.net/qq_43088815/article/details/89926074

```python
获得参数的方法
list(self.net.parameters())
print(type(model.state_dict()))  # 查看state_dict所返回的类型，是一个“顺序字典OrderedDict”
model.state_dict()、model.parameters()、model.named_parameters()# 这三个方法都可以查看Module的参数信息，用于更新参数，或者用于模型的保存。
```

No dashboards are active for the current data set.

以下都不行

```
tensorboard --logdir D:/DL2-pytorch/TensorBoard --port 8123
 tensorboard --logdir=D:/DL2-pytorch/TensorBoard --port 8123
 tensorboard --logdir=D:\DL2-pytorch/TensorBoard --port 8123
 tensorboard --logdir=./TensorBoard --port 8123
```

头痛,可能是没写进去不知道为啥. 

因为这里是空的,所以没写进去. 因为没有sl agent

```python
while not stats_qs[agent].empty():
    stats = stats_qs[agent].get()
```

头大, 为什么一定要sl agent啊?  要发送梯度给central_agent. 为了之后的分布式学习.

```python
    os.system("rm -f *.log")
    os.system("sudo pkill -9 tensorboard; sleep 3")
    os.system("mkdir -p " + pm.MODEL_DIR + "; mkdir -p " + pm.SUMMARY_DIR)
```

file.setting -> file encoding-> project encoding 为GBK就不会输出乱码, 否则中文error会变成乱码. 

pull weights,在tensorflow里可以直接用model.get_weights()和model.set_weights()来做，比较直观和方便。pytorch怎么做?https://blog.csdn.net/qq_36810398/article/details/107048855

感觉太多了写不完,  他这个是分多agent 训练, 发送信号很麻烦, 我想先一个agent 监督学习.

新开了一个分支 singleSL.

episode 每个episode 会验证一次然后保存性能好的模型. epoch就是所有数据examples都完成了一次.

[Digital-image-process/cifar10.py at main · LukeLIN-web/Digital-image-process (github.com)](https://github.com/LukeLIN-web/Digital-image-process/blob/main/Final Project/code and data/cifar10.py)

怎么证明 SL收敛到DRF了? NN的决策 reward = DRF的决策 reward?还是 JCT相同. validate 就是simulate了一个环境, 然后step一步步直到job结束.

为什么simulate 需要drf_env, 那drf和net什么关系? 怎么对比DRF和我们net的makespan? 是否需要RL_env? RL_env 没有

目前`num_jobs, jct, makespan, reward = env.get_results()` 

网络为啥是torch.Size([256, 5, 61])? 20个job, 因为 input 256个traj.state.

```
ACTION_DIM = 3 * SCHED_WINDOW_SIZE + SKIP_TS  
3*20 +1 61个
```
