Deep Learning-based Job Placement in Distributed Machine Learning Clusters

> 作者的话: We read the source code of Pensive (https://github.com/hongzimao/pensieve) and DeepRM (https://github.com/hongzimao/deeprm) as reference during our implementation. The integration with Kubernetes API is similar to Optimus (https://github.com/pengyanghua/optimus). If you have any concrete questions about the implementation details, feel free to ask us.
>
> harmony 为什么用DRL 不用LSTM? 有什么区别? 
>
>  You can try LSTM. I tried it but the performance is not good enough. For Google's paper, I tried that before but did not get how they built a dynamic NN. 

A 尝试解决什么问题

1. co located ML job干扰导致性能下降,  Mesos 集群调度没有考虑干扰.  详细的工作负载分析and 干扰建模处理.  harmony,  设计DRL 来place job, 最小化干扰和训练时间. 

2. 增强了奖励模型.  采用了 actor-critic , job-aware action space exploration and experience replay.  

3. **如何才能高效并行的产生海量的数据供神经网络进行学习是大规模DRL的关键。**缺乏不同place 决策的奖励样本, 构建了一个辅助奖励预测模型,  历史样本训练value NN. 然后valueNN 产生reward 让policyNN 可以不需要online也可以不断强化.

B方法有哪些关键元素

其实就看related work就讲了它的特殊性. 

1. 对于加速, 之前研究设备数量调整. 我们fix 资源, 研究了 DRL 优化job placement.  
2. 干扰他们都建模然后分析. 我们利用历史data trace 来训练 NN without loss of generality.
3. **DRL**  他们都假设有足够数据, 用simulation model 产生训练样本或者online 测量.  我们 不建模, 用reward prediciton NN来产生 训练样本.  

C 有什么我可以用的

policy NN应该可以用.

## Abstract

 介绍了harmony , 用最小化干扰和最大化性能方式place job. 用了DRL 框架, 增强了reward 模型,  用了各种方法 稳定训练提高收敛性.    缺乏不同place 的reward sample, 构建了reward预测模型,  

​	Production machine learning (ML) clusters commonly host a variety of distributed ML workloads, e.g., speech recognition, machine translation. While server sharing among jobs improves resource utilization, interference among co-located ML jobs can lead to significant performance downgrade. Existing cluster schedulers (e.g., Mesos) are interference-oblivious in their job placement, causing suboptimal resource efficiency. Interference-aware job placement has been studied in the literature, but was treated using detailed workload profiling and interference modeling, which is not a general solution. This paper presents Harmony, a deep learning-driven ML cluster scheduler that places training jobs in a manner that minimizes interference and maximizes performance (i.e., training completion time). Harmony is based on a carefully designed deep reinforcement learning (DRL) framework augmented with reward modeling. The DRL employs state-of-the-art techniques to stabilize training and improve convergence, including actor-critic algorithm, jobaware action space exploration and experience replay. In view of a common lack of reward samples corresponding to different placement decisions, we build an auxiliary reward prediction model, which is trained using historical samples and used for producing reward for unseen placement. Experiments using real ML workloads in a Kubernetes cluster of 6 GPU servers show that Harmony outperforms representative schedulers by 25% in terms of average job completion time.

## 一introduction

  worker和PS, 怎么高效place到服务器上?

许多调度器如 borg , mesos 给job分配更多CPU和内存, server少一些.    但是因为jobs 共享底层资源,  CPU caches, disk  IO ,  network IO and buses QPI,PCIe.  比如   GPU 分配给不同的ML  , job shuffle data between CPU和GPU时就share PCIe bus .    在NUMA 中, 两个分配的GPU没连到同一个CPU就要共享QPI（CPU到内存更快，）.  

一些ML 模型  CTC 读取图像预处理, 就是CPU 密集型,  一些AlexNET是磁盘IO 密集型,  一些网络带宽消耗大因为模型尺寸大(参数数量多) minibatch小(worker之间参数交换多) 如VGG 16

我们需要把低干扰的job放一起, 但是yarn , mesos是不考虑的.  一些文章 建立了干扰模型, 手动启发式把干扰纳入调度, 但是几十个干扰源, 需要仔细优化参数或者阈值, 而且不通用, workload type 或者硬件配置变了就不好用了. 

所以这里黑盒,  在NN中encode workload 干扰, 把raw cluster和 job 状态 map 到 job placement 选择.

贡献如下: 

1. 提出通用设计, 适应未知的干扰. 
2. 用各种训练技术比如 加快收敛, 构建辅助reward 预测模型,  用有限的历史samples 来训练, 为unseen placement 产生reward. 

## 二 background and motivation

### a. 分布式学习

SGD到底是PS 选一些 还是 worker选一些? 是worker选一些, PS是把所有worker发来的梯度都用上. 

整个 dataset处理过一次叫one epoch.

### b colocated 的干扰

#### case1 bin packing vs 独立standalone执行 

一个job 有多个PS和worker  由于全局同步,一个worker /PS的错误放置  就会让 性能下降更加严重.  

#### case2  pair -wise 成对干扰水平

 不同工作干扰不同.  所以有优化的机会.

#### case3 placement under representative policy

比较三种策略, 负载平衡: mesos,k8s,  多资源打包: google borg和tetris,  独立执行: 我们自己写. 

独立执行最快, 但是资源没有充分利用. 

ideally,  bin应该优于负载balance, 因为PS和worker在一起不用跨机器通信.  但是由于干扰就不一定.  放同一个服务器的越多干扰越严重. 

 Standalone execution leads to the best performance, but also resource underutilization. Ideally, bin packing should outperform load balancing, in terms of both resource utilization and training performance, since it avoids cross-machine communication by placing PS and worker together. But this is not true when the second job trains  , which performs better under the load balancing scheme. This is due to the more severe interference between training ResNeXt and CTC together. Fig. 5 further shows the training speed of ResNeXt (or VGG) when the number of co-located CTC training jobs increases. The more jobs are co-located, the worse the interference is.

更多job colocated, 组合数量huge, 难识别和建模interference. 

All 3 schemes are not good enough to achieve high resource utilization and training speed at the same time. When more jobs are colocated (the number of different combinations of jobs would be huge), performance interference is even harder to identify and model. We resort to a black-box policy learned through DRL for job placement.

### c 应用DRL的挑战

1. action and state space 指数增长,   insufficient or ineffective exploration 导致**收敛到良好的决策策略 ** 困难

2. 没有这么多traces 覆盖所有possible placement.  样本不够就不能训练DRL NN收敛到一个好的policy.  

 解决方法:  为了训练我们的 DRL 模型，我们需要一种方法来为 DRL 生成的placement decisions  提供 synthetic(合成)reward samples . 这个placement decisions是历史ML 集群trace中不存在的。 我们不依赖分析干扰模型 [11] [12]，而是采用更通用的方法，使用**另一个 NN 进行奖励建模，使用可用trace通过监督学习进行训练。**

 奖励建模 是监督学习.  强化学习就是自己训练自己, 自己生成奖励. 因为之前的placement和reward是远远不够的. 

We use Historical Trace to get reward function, use reward function to produce synthetic traces. synthetic traces 

输入: historical trace ,是有标签的. 

输出: reward prediction 

## 三.系统overview

DRL训练什么？  训练NN参数。 选择通过reward model 获得虚拟trace。 重复了怎么办？ 重复了没关系， 继续强化这个action。

online 训练什么 ？线上实际推断， 做出的placement和reward作为历史trace，给reward model 监督学习。 

SL 训练什么？ 历史trace输入， NN调整参数， 拟合 历史输入和输出的关系。

我们专注于使用参数服务器PS架构的分布式 ML 作业； 我们的设计可以很容易地扩展到使用 all-reduce 类型的算法在worker之间进行直接参数交换来处理工作 。

 job owner 要提交什么? 

1. 每个worker和PS需要的资源. 
2. PS 和worker总数
3. 总epoch数. 

harmony batches(批处理) interval中**新到达的**job, 然后决定他们的placement. 然后deploy.  不调度正在运行的job placement.(因为问了微软和阿里都没这么做过)

**离线学习**,   DRL大量实验和错误才能收敛, 所以不能一开始就 online study.

离线分为两步,

 1  监督学习训练 奖励预测NN,为place 决定了的数据提供奖励评估. label是每个job的reward也就是训练速度. 输入是job 信息和placement.

2 DRL 模型训练,  为新job 生成place决策

他是收集了一连串的 状态, 还是自动得到后面的状态? 

## IV. DEEP REINFORCEMENT LEARNING BASED PLACEMENT POLICY

### A DRL framework

**state space** 

这个job输入是缺点, 因为需要指定几个类型, 但是也是优点, 因为很多placment 其实只能训练一种计算图(不过如果训练很快可能也可以处理多种? 应该是要看你到底侧重那个问题, 这个论文是侧重 colocated 的, 有的论文不看重 colocated 而是就是优化一个job的放置). 

输入: job type,resource demand, # of workers/PS ,available resources on the servers, placement of workers and PSs 

x  每个job的类型.  N是job的数量,  L是job的类型数量也就是model的数量.  这个和DL2输入一样

r  jobs中worker/ PS的 resource demands.   

worker数量, K个资源类型.  比如K =2 , worker3个, PS 2个[  3,1,2,2,0,1 ]  3表示worker的数量,   (  DL2只有worker的输出, 没有resource demand比如CPU,GPU的输出, 在哪里有相关文献? ) 一般 CPU是固定的, 不需要输出. 是job owner提交的

w 和p分配给 job的worker/PS数量.  这个DL2输出可以获得 . 这个 •For r and w(p), I tried several combinations and chose the one with the best performance.

v 每个服务器上 每种资源的数量. 这个可以直接获得available  resource 

d M x2N ,服务器m上 运行了哪些job的哪些PS. 也就是当前的job   placement情况. 这个是existing. 比如sever m hosts 1 PS and 1 worker of job 2 and 1 PS and 2 workers of job 5 among 6 jobs; we have dm = [0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0]. (•是不是就是目前存在的job的运行情况, 不包括新提交的job? 如果是, 那就可以直接获得, 感觉应该是, 要不然还placement干嘛都place好了) 学姐说是

•d includes all concurrent jobs.

**action space**  接收到s, 根据策略 pi 选择动作a, 策略是概率分布, 由NN生成,  

输出 placing one worker/PS of job n on server m, 这个是新到达的job

action space 不能过大, 否则训练时间长 而且效果差. 

我们action space  2MN'  个actions, N' 表示新到达的jobs数. (n, 0, m )  放一个  n 的worker 到server m上 , (n, 1, m )  放一个  n 的PS 到server m上.

  一个interval 中可以多次inferences推理, 一次选一个action, until 提出 完整的placement. 或者没有resource时停止. 这可以reduce action space. 

**reward** 

我们设计了每个间隔的reward,  这一段和 DL2好像是完全一样的.

**NN** 

主要看看图7 ,  要mask invalid actions 也就是资源不够的action, 然后rescale调整概率和为1.  



### B DRL Model Training

和DL2也差不多. 

增加选择Q>0 的action的概率

1）Actor-critic：REINFORCE 算法可能会受到推导出的 Q 值的高方差（用于计算梯度），从而阻止策略模型的快速收敛 [25]。 为了减少方差，我们使用 actor-critic 算法改进了基本的策略梯度训练。 基本思想是引入依赖于状态的函数，以改进 SGD 中用于更新策略神经网络的梯度。 如果该动作的“质量”Qπθ(si,ai) 优于 si 中所有可能动作的“平均质量”，则我们reinforce an action . 我们evaluate how good an action is by its advantage  Q-V.   这样可以减少方差让policy learning more stable.

2) 探索action space,  加入交叉熵正则化 an entropy regularization term 到 gradient calculation.  第二个是就是用omega-greedy.  omega的概率随机选择bin packing and load balancing . 1-omega 的概率用NN输出策略.  bin packing 是剩余容量最少,  load balancing是 负载最少. 让NN有效探索 resource utilization 和workload interference. 

3) experience replay

uses  FIFO buffer to store samples from multiple past interval instead of last interval, it could reduce the correlation among the samples so that accelerate converge. 

## V reward 预测model

NN不需要手动指定特征,  更通用.

输入:

x  每个job的类型. 

不需要r  jobs中worker/ PS的 resource demands. 

w 分配给 job的worker数量

p 每个服务器上 每种资源的数量

d 每个服务器上 运行了哪些job的哪些PS

输出: 预计的训练速度.

中间经过一系列 hidden 全连接层 

## VI performance evaluation

### a implementation

**scheduling  on k8s** interval的开始,  harmony   发送http requests 到k8s的API 服务器 来queries 没调度的job和当前cluster状态. 然后用训练好的policy NN makes placement decisions.  k8s 就启动各个设备,  harmony  用在线收集的data 更新 DRL 和reward 模型.  

**DRL 训练** 离线训练 用 tf ,  这一段有详细介绍每一层. 

**Reward Model 训练**  和DRL差不多. 

### B evaluation methodology

讲了具体的机器配置. 

baseline  把harmony和 负载均衡LB,  tetris,  LIF-Line 最小干扰优先 这三种比较. 

### C performance 

讲了具体性能比较. 

### D Deep Dive

讲了harmony 神经元数量, 隐藏层数量,  测试了value network, exploration ,experience replay这些设计的提升效果. 

## VII related work

**ML job scheduling**  dorm 公平资源分配, oasis 在线分配, slaq 和optimus 建立了性能模型动态资源分配. 他们研究设备数量调整. 我们fix 资源, 研究了 DRL 优化job placement 

**interference-aware task placement** 

1. 之前有研究VM placement , 把各种负载分类,
2. bu 等  研究mapreduce的网络干扰和局部性干扰
3. xu等 建立了mapreduce的分析模型,研究VM 干扰. 
4. paragon等 用collaborative filtering 预测应用性能

他们都建模然后分析. 我们利用历史data trace 来训练 NN without loss of generality.

**DRL**   

1. mao 等用DRL 设置task 并行和执行顺序 
2. liu等 用DRL 设计动态能源管理policy 
3. mao等 用它调整视频流速率. 
4. mirhoseini等用来 优化operator placemetn 
5. xu 等用来选择交通path 

他们都假设有足够数据, 用simulation model 产生训练样本或者online 测量,.  我们 不建模, 用reward prediciton NN来产生 训练样本.  

## VIII 结论

harmony 考虑干扰, 优化placement, 加速训练. 

不分析设计, 我们先用历史traces 训练 reward NN , 然后训练 DRL , DRL做出placement 决策.  我们认为我们 reward NN是通用的可以用在其他历史trace不足的DRL问题上.

