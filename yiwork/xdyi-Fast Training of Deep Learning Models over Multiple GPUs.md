Fast Training of Deep Learning Models over Multiple GPUs

fastT的脉络

## abstract

## 问题

之前很多是黑盒训练, 要训练很久而且耗费很多资源.有的只能拆分几个层而不是拆分所有.

但是整个模型在单个GPU真的最好吗？我们觉得不太好.

## 系统的优点:

1.白盒算法, 快速找到device placement和执行顺序,有的时候即使计算速度长, 但是每次迭代的速度还是有进步.(为什么?) 因为通过分解操作和 execution ordering执行定序拓展解空间

2.提出了新的启发函数,  新的启发函数是什么?  新的启发算法是那一整个算法

1. 我们可以很好兼容 TensorFlow, 各种模型都可以用,不用修改任何一行代码

## 3问题定义

问题输入, 1. 一个 dag计算图,  2.gpu的集合和他们的内存,3. 计算和沟通的开销模型(哪来的?) 计算cost模型就是 一个设备跑一个operation 需要的计算时间.

输出: 1. 操作应该怎么划分的一个list 2. 设备怎么placement 3. 执行顺序.

然后讲了特例 如果不是DAG怎么办.

## 4 系统设计

python client 加了 device placer, strategy calculator, cost models,

data-flow executor 加了  order enforcement

### device placer:   分配给不同device 不同操作

根据策略计算器计算出的策略将每个操作分配到设备上。

### order enforcement:组织操作的执行顺序.

从策略计算器获取操作的执行顺序后，将顺序列表中操作的索引设置为它们的优先级，并在 TensorFlow 的执行器中确保执行顺序。

### cost models成本模型: 记录设备上操作的执行时间和传输时间.

第一步cost models 预热, 先保存参数checkpoints, 然后根据operation partition lists创建新图, 启用了新策略后, 分配设备和顺序. 记录每次时间, 时间更长就回滚上一个策略.  直到变化不大的时候就算完成pre训练, pre训练好了之后, 就placement and execution order strategies ,placement and execution order strategies 用完之后，如果时间更长，也会回滚. 之后只有执行时间异动大的时候再更新cost model.

pre 训练的时候, 用DPOS来计算placement and execution order strategies 下面简称PEOS, 用这些策略训练DNN模型,  分析训练过程来修改成本模型, 如果成本模型中有的时间还没有, 我们的算法会设置为0 . 通常几次迭代就可以获得完整的成本模型

一开始为了减少分析开销和获得每个device准确的操作时间,我们用数据并行(塞得下吗?), 然后 模型并行. 用线性回归来获得通信开销模型.把同 source destination设备的tensor 收集到一组中, 每一组有一个 张量size为x , 传输时间为y的线性模型, 用线性回归来得到.

### strategy calculator:算device placement,执行顺序 以及 operation partition lists

预训练的时候计算PEOS 以及 operation partition lists

正式训练, 定期激活 profiler,更新 cost model, 重计算,如果算出来时间小, 那就activate 新策略.

## 怎么训练的?

一个listing schedule , 一个 关键路径中拆分操作.(为什么可以支持细粒度和模型并行性?)

## listing scheduling

解决方案两个阶段, 一 operation prioritization 选择device placement sequence of all operations(问题:  什么是device placement sequence? 其实就是 优先级队列 L)

二 device selection 选择操作 然后分配给best device

### operation prioritization

首先, 每个操作有一个rank, 可以递归计算出来.

我们使用 rank_u (oi) 作为 oi 的优先级，使得下一个要放置的操作始终是当前子图新关键路径中的入口操作，排除已经考虑过的操作。

### device selection

EST(oi,dj)要等好几个, 要等设备j available, 还要等前继所有节点执行完并且传输过来了，才能开始执行.

怎么选设备?

如果oi是关键路径上的操作, 就选择执行关键路径oi最快的设备 ,否则, 把每个device试试执行oi, 选eft最小的. 然后把每个操作分配好设备了.

```python
ranku值降序创建优先级队列 L。
while L is not empty do
	oi <- L.dequeue() # 根据优先级逐一分配.
    if oi in SETcp:
        Snew[oi] = dcp(oi)# oi的策略就是用关键路径cp device
    else:
        for d in D: #D是device的set
            if oi 进来后d的memory超出:
                EFT(oi,d) =  +∞  
            else:
                compute EFT(oi,d)
        Snew[oi] = min(EFT(oi,d)) #不同d中最小的eft
        FT(oi) = EFT(oi,Snew[oi])
计算Compute Execution list A by sorting operations in ascending order of ST(oi)
Compute FT(oexit)=EFT(oexit,Snew[oexit])
```



摘要

This paper proposes FastT, a transparent module to work with the TensorFlow framework for automatically identifying a satisfying deployment and execution order of operations in DNN models over multiple GPUs, for expedited加速 model training. We propose white-box algorithms to compute the strategies with small computing resource consumption in a short time. Recently, similar studies have been done to optimize device placement using reinforcement learning. Compared to those works which learn to optimize device placement of operations in several hours using large amounts of computing resources, our approach can find excellent device placement and execution order within minutes using the same computing node as for training. We design **a list of scheduling algorithms** to compute the device placement and execution order for each operation and also design an algorithm to **split operations in the critical path** to support fine-grained (mixed) data and model parallelism to further improve the training speed in each iteration. We compare FastT with representative strategies and obtain insights on the best strategies for training different types of DNN models based on extensive testbed experiments.

他和TensorFlow框架一起工作. 自动部署, 多个GPU可以顺利训练DNN。 开销小。 也可以用在reinforcement 学习。几分钟就可以安排好。 设计一系列调度算法（为什么？ 为了计算device placement和执行顺序）还有一个算法 切分关键路径的操作。支持细粒度数据和模型并行。

他是怎么想出来的？

## introduction

Deep Learning (DL) has become increasingly popular over the past years in various application domains such as computer vision, speech recognition and robotics. With increasingly complicated models and larger datasets, training of a deep neural network (DNN) model becomes an extremely time consuming job. Parallelizing the training using workers equipped with multiple GPUs or in a distributed environment is popular with current machine learning (ML) frameworks [2, 6, 9, 13]. 

Three of the most common parallelization strategies are data parallelism, model parallelism and pipeline parallelism. 

**Data parallelism** places a replica of the entire neural network (NN) on each device ( e.g. , a GPU card) so that each device processes **a subset of the training data** and synchronizes model parameters in different replicas副本 at the end of each training iteration. 每个设备有个训练集的子集。

 **Model parallelism** handles models with a large number of parameters, which cannot fit into the device memory, by assigning **disjoint partitions** of an NN each to a different device.将NN的不相交的分区分别分配给不同的设备 

**Pipeline parallelism** divides a DNN model into **stages** and places stages on multiple devices; it further divides each mini-batch批次 into micro batches, so different devices can process different micro batches simultaneously. 

To date, it is still not clear that given multiple GPUs, what the best strategy is to deploy a specific model onto the devices. Commonly, a practitioner may use data parallelism and replicate the model onto each GPU, but is this really always the best strategy even when a single GPU can hold the entire model?但是整个模型在单个GPU真的最好吗? And how about large models that cannot be entirely replicated to a single GPU? 

In this paper, we show that a mixture of fine-grained data and **model parallelism** combined with some heuristics is able to find a satisfying device placement in a fast manner with little resource consumption. We also seek to develop a software module to enable **automatic model deployment** without requiring model developers’ code modification, which can seamlessly无缝地 work with existing frameworks such as TensorFlow. For a small model which can be deployed in a single GPU, it should find a strategy achieving faster training than normal data parallelism, if there is one; for a large model which cannot be deployed entirely in a single GPU, it provides a good deployment across multiple GPUs.

 We propose FastT, a transparent module that automatically finds and activates a satisfying deployment and execution order of operations for different kinds of models in a multiGPU environment. Our contributions are summarized as follows:

▷ We propose new heuristics启发函数 that find deployment and execution orders in a few minutes, which are better than or as good as previous strategies that require hours to be computed. 更快 The main reasons lie in that we **extend the solution space** by considering operation split and execution ordering, and we use efficient **white-box heuristics** rather than search or learning-based methods to reduce strategy calculating time. FastT is efficient enough to be executed on one node (the same as a single worker used for model training), removing the need for an additional cluster for strategy search.

▷ We build adaptive cost models to facilitate our algorithms. To minimize profiling分析 overhead while obtaining accurate operation execution time on each device and inter-device communication time, we use **data parallelism** as the starting strategy (as long as it is feasible), try out different placements and apply linear regression回归 to obtain the communication cost model.

▷ We consider a larger solution space than previous approaches by considering both execution order and fine-grained parallelism within operations. We observed significant performance variation under the same device assignment with different operation execution orders.不同顺序性能变化很大 FastT decides execution order and achieves fine-grained parallelism by **splitting some operations on the critical path** to further improve the processing speed. Experiments show that FastT achieves up to 59.4% speedup compared with pure data parallelism with the larger solution space.

▷ We provide an open-source implementation of our method that transparently works with TensorFlow: developers do not have to modify their ML model to leverage our solution. FastT is useful for various models, and able to automatically calculate and activate placement and execution without involving the ML developer. We have built FastT based on TensorFlow: once the FastT module is turned on, developers can transparently use it with their existing models implemented with all kinds of TensorFlow’s Python APIs without modifying a single line of their code. 

## 2 Background and Motivation

##  2.1 DNN Training and Parallelism

 Training a DNN is an iterative process which uses a large amount of data to tune model parameters for minimizing a loss function. In current training frameworks [2, 6, 9, 13], different kinds of computation are implemented by different operations (such as Conv2D, MatMul), and input and output of these operations are called tensors张量. The computing process can typically be represented by a DAG (Directed Acyclic Graph), whose nodes are operations and edges are tensors. 

**Data Parallelism.** The input data are partitioned to different devices. Each device shares the same model parameters. Gradients梯度 from all devices are applied to update the global model. Data parallelism can be applied in a single worker/machine with multiple GPUs [19, 26, 27], and among multiple machines [37]. 每个数据算梯度然后更新整个模型

**Model Parallelism.** The input data are sent to all devices without partition; each device is responsible for tuning a different part of the model parameters. Model parallelism is typically used for models with a large parameter size [26, 34, 45]. 每个设备调一部分参数. 适用于 参数size很大的models.

**Pipeline Parallelism.** Pipelining has been proposed to accelerate DNN training with multiple accelerators [12, 14, 22]. Many DNN models stack layers sequentially; naïve model parallelism may result in only one active accelerator anytime during training. With pipeline parallelism, similar to model parallelism, different layers are deployed on different accelerators; a mini-batch is further divided into several micro-batches and these micro-batches can be processed at different layers at the same time to fully utilize all accelerators.

## 2.2 Fine-grained device placement

 **Operation-level** device placement. With model parallelism, a model is typically partitioned in the **layer level**. A layer consists of multiple operations. To expand the solution space, some operation-level approaches are proposed [19] to decide device placement of each operation separately.   更小粒度, operation level和layer level. 都属于模型并行

Parallelism within operations. To further extend the solution space, some studies [11, 16] investigate potential parallelism within individual operations. For example, for a Conv2D operation, it can be further parallelized by being partitioned on the batch size dimension or the channel dimension [27]. Such an approach can be regraded as fine-grain mixture of data parallelism and model parallelism according to different parallelizable dimensions of different operations. 

## 2.3 Limitations and Challenges 

Previous research [28, 46] has proposed strategies to manually optimize parallelism based on human experts’ domain knowledge and intuitions. For example, Krizhevsky [28] uses data parallelism for convolutional and pooling layers and switches to model parallelism for fully-connected layers to accelerate the training of convolutional NNs (CNNs). In addition, some automated frameworks [19, 23, 27, 32] are proposed for finding efficient parallelism strategies in a limited search space. REINFORCE [32] uses a reinforcement learning method to learn efficient operation assignments on multiple GPUs. TicTac [23] explores the impact of the order of send/recv operations in distributed training. Allowing fine-grained parallelism within a single operation, FlexFlow [27] builds a new training architecture to explore the SOAP (Sample-Operation-Attribute-Parameter) search space considering parallelism within and across operations. Some other frameworks focus on specific types of networks such as CNN and RNN (Recurrent循环 Neural Network), and provide APIs for developers to split operations by themselves such as TensorFlow mesh [7] and tofu [44].

The existing proposals have the following limitations: 

First, the purpose to find optimal device placement is to save time and computing resource for a training job, and the finding process itself should not be time and resource consuming. Some existing approaches require a large amount of resources and spend a long time to obtain the strategy. For example, REINFORCE [32] and GDP [48] use another big cluster consisting of tens of workers and spend hours on learning the device placement policy 找方法很费资源很花时间

Second, existing approaches may not be generic通用的 enough for different kinds of models or not compatible with popular training frameworks. For example, OptCNN [26] is designed for CNNs. FlexFlow implements the training framework itself, and does not support representative frameworks such as Tensorflow or MXNet.

Third, the solution space can be particularly large. Most existing studies only consider device assignment of operations, but not the execution order of operations.除了分配外, 顺序也很重要 For example, FlexFlow defines a fine-grained search space beyond data and model parallelism, and schedules operations in the ready queue with a simple FIFO strategy.之前的虽然有细粒度但是没啥调度顺序优化

We seek to address the following challenges in this paper: 

▷ We consider not only device placement of operations, but also partitions of operations and their execution orders; the solution space becomes much larger than what the existing studies tackle. Finding a satisfying solution in a timely manner with small computation resource consumption is critical for solution adoption in a production environment. 研究怎么切分operations和 安排顺序. 解空间特别大.

▷ It is easier to design strategies for a specific type of models. However, a generic approach needs to analyse the structure (DAGs) of different models and builds the respective cost models  模型怎么通用

▷ Since ML developers may use various APIs to implement their models (even when they are using the same ML framework such as TensorFlow), it is hard to design a unified software module to transparently support their existing models without modification. 怎么实现transparently支持

A practical model deployment and execution module must be fast, light-weight, generic and compatible with existing training architectures at the same time.

## 3 Problem Definition 

We first formally define the problem we intend to solve. The objective is to find a good device placement and execution order to achieve parallelism across operations in a DNN model, and also identify potential operations which can be partitioned into several sub-operations to achieve fine-grained parallelism within individual operations. The input of the problem includes: (a) the DAG computing graph, (b) the set of devices (GPUs) and memory limitation of each device, and (c) the cost models for computation and communication. The computation cost model provides computing time of a given operation on a specific device, and the communication cost model gives inter-device tensor communication time of adjacent operations running on different devices 一个计算开销模型, 一个沟通开销模型.

The output solution consists of three parts: (i) a partition list of operations which should be partitioned (each item in the list has three elements, the operation’s name, partition dimension, and the number of partitions); (ii) device placement of each non-partitioned operation and each sub-operation (due to splitting operations in the partition list); and (iii) execution order of operations and sub-operations.  一个list写怎么分, 然后怎么分配device, 第三个是执行顺序.

It should be noted that we focus on NN whose computing graph is a DAG. Some networks can be implemented as a graph with cycles in TensorFlow, e.g. a dynamic RNN which includes a while loop, and whether to exit from the loop is decided during runtime. For such a model, we optimize execution执行模式 of the DAG within each of its loops.  如果是环, 就每次loop优化execution of DAG

The problem of deciding execution order and placement of a DAG with unit operation execution time is known as **the single execution time scheduling problem**, which is NP-complete [42]. Our problem poses even greater challenge as we assume heterogeneous operation execution time每一次操作时间还不一样. Therefore, we propose efficient heuristic algorithms in Sec. 5 to find a good solution of our problem.

## 4 System Design

FastT is built based on TensorFlow, addressing parallelism both within and across operations in a DNN model. It calculates both device placement and execution order for each operation in the computation graph, with operations potentially further partitioned.

Fig. 1 illustrates how FastT fits into the architecture of TensorFlow, and coloured blocks represent components we implement for FastT. **The strategy calculator** (算device placement和执行顺序)computes device placement and execution order for the current model using the algorithm to be introduced in 5.2.  **The device placer** (分配任务)assigns different devices to run different (sub-)operations according to the strategy computed by the strategy calculator.  In the dataflow executor, **order enforcement**执行器 is responsible for organizing the execution order of (sub-)operations. **The cost models** component记录执行时间和传输时间 records the execution time of (sub-)operations executed on different devices and the data transmission time when adjacent operations are handled on different devices.

The workflow of FastT is as follows: Initially, FastT requires several pre-training steps to bootstrap引导,自引 the cost models, which uses different placement strategies to run the DNN model to update its cost models. 新策略要先检查点保存, To activate a new strategy computed with the updated cost models, the training session does checkpoints of current model parameters, and restarts to create a new graph based on the operation partition lists; then the device placer activates the device placement and the order enforcement module enforces the execution order of (sub-)operations. The training session then restarts with restored parameters from the checkpoints. 

After a new strategy is activated, FastT records the per-iteration model training time; if it finds that the per-iteration execution time with the new strategy is even longer than the previous one, it rolls back the strategy to the previous one.(策略保存在哪里?) When the cost models become stable (the average time of the same (sub- )operation(s) on the same device(s) does not vary much), we finish the pre-training stage.预训练完成

 Afterwards, the model is trained normally using placement and execution order strategies computed by the strategy calculator, and the cost models are updated only when the execution times have changed significantly based on our periodical profiling. 第一步cost models 预热, 先保存参数checkpoints, 然后根据operation partition lists创建新图, 启用了新策略后, 分配设备和顺序. 记录每次时间, 时间更长就回滚上一个策略.  直到变化不大的时候就算完成pre训练, 之后只有执行时间异动大的时候再更新cost model.

Currently, we use checkpointing and restart the model for activating a new strategy, since commonly adopted TensorFlow versions do not allow modification of a graph structure when the session has already run the graph. 运行的时候不能改所以我们 用checkpoint 然后restart改. After the normal training stage starts, the cost models are not updated often. 

**Cost Models.** The computation cost model provides the execution time of a (sub-)operation on a device, using the **operation’s name and device** as the key. The communication cost model provides the tensor transfer time between adjacent operations assigned to two different devices, according to tensor size and device pair. Different from simulation-based measurement of such time [27], we profile the training process and record real execution/transmission time, based on the RunMetadata [5] generated by the TensorFlow profiler. 不模拟,利用runmetadata记录真实的时间

​	During the pre-training stage, FastT first uses its algorithm (DPOS in Sec. 5) to compute device placement and execution order strategies (a default data or model parallel strategy is used when cost models are empty), trains the DNN model with these strategies for several iterations (aka steps), and profiles the training process to update the cost models. Especially, when our algorithm finds a cost that it needs (e.g. , execution time of an operation on a device) is not in the cost model, it sets the cost to 0, so that the algorithm prefers to explore the placement, and then the profiler can obtain the real cost of this placement in the following training steps. It typically only takes several iterations to obtain the complete computation cost model, considering that we use data parallelism as the starting strategy (as long as the model can be fit into a GPU) by which each operation is replicated to different GPUs and their execution time on different devices is learned. For a large model that cannot be fit into a GPU, we use model parallelism, try different placements on multiple GPUs, and obtain the cost models. 就是试试把这些操作放这个GPU,然后试试放另一个GPU.  model小就 data 并行, model太大了就model 并行

​	To build the communication cost model, we gather tensors across the same source-destination device pairs into one group. For each group, we use linear regression to obtain a linear model: tensor size vs. transfer time. In each update of the cost model, newly collected data are fed and parameters of the linear model are re-computed. The models capture available bandwidth and potential congestion along each device-device path. 线性回归获得线性模型

**Strategy Calculator.** It is the key component to carry out the algorithms that we will discuss in Sec. 5. During the pre-training stage, it calculates device placement, execution order and operation partition lists, and obtains the cost models. During the normal training stage, it periodically activates the profiler, updates the cost models, and recalculates new strategies. If the estimated per-iteration training time with the new strategies (among output of our DPOS algorithm) is smaller than that of previous strategies, the new strategies are activated.

**Device placer.** Device placer is responsible for assigning each operation onto a device (GPU) according to the strategy computed by the strategy calculator. **Order Enforcement.** After obtaining execution order of (sub-)operations from the strategy calculator, the enforcement module sets the indices of (sub-)operations in the order list as their priorities, and enforces the execution order in TensorFlow’s executors.

## 5 Operation Placement and Ordering Heuristics

##  5.1 Listing Scheduling

 We design a listing scheduling method to compute the device placement and execution order of operations, inspired by algorithms handling DAG task scheduling over multiprocessors [20, 41]. With listing scheduling, the whole solution space is reduced in two phases: (i) operation prioritization for deciding device placement sequence of all operations, and (ii) device selection which chooses the operation in the order of their priorities and assigns the best device to each selected operation, to minimize the operation’s finish time. (i)决定所有操作的设备放置顺序, (ii) 选择操作 然后分配给best device

**Operation Prioritization**. The priority decides the device placement sequence, which is slightly different from the execution order of operations. We exploit a critical-path [41] based heuristic for computing a rank for each operation oi in the DAG:
$$
rank_u(o_i) = w_i + max(c_i,j +rank_u (o_j)), oj是oi的后继
$$
where w_i is the maximal execution time of operation oi (over different devices that it could be assigned to run), succ(oi) is the set of immediate successor operations of oi , and ci,j is the maximal transmission time of the tensor from operation oi to operation o_j (over different device pairs that they can be located on). rank_u (oi) represents **the length of the critical path from operation oi to the exit operation**, and can be computed recursively by traversing the computation graph, starting from the exit operation. The rank of the exit operation is: rank_u (o_exit ) = w_exit. 好像是从后往前从exit开始计算 rank,上面那个rank的计算方式也是一个从sink节点开始，recursive算出来的.

​	we use rank_u (oi) as oi ’s priority, such that the next operation to be placed is always the entry operation in the new critical path of the current sub-graph, excluding 排除the operations that has already been considered. 

**Device Selection & Execution Order**. We use EST (oi ,dj) and EFT (oi ,dj) to represent the earliest **execution start time** and the earliest **execution finish time** of operation oi on device dj , respectively. For the entry operation o_entry of the DAG, we have     入口开始时间为0

EST(oentry,dj) = 0 for dj ∈ set of devices

For other operations, EFT and EST can be computed starting from the entry operation as follows: dj就是device j,  oi就是operation i

EST (oi ,dj)  = max{avail[j] 最早可用时间, max( EFT(om) +the actual tensor transmission time between oi ’s immediate predecessor om and oi ) }om 从oi的直接前驱操作中选一个出来.

EFT  =w_ij +EST (oi ,dj)  wij是device j上的操作i执行时间

Here, avail[j] is the earliest available time of device dj ;pred(oi) is the set of immediate predecessor operations of oi ; cˆ dj m,i is the actual tensor transmission time between oi ’s immediate predecessor om and oi , if oi is assigned to device dj . Note that avail[j] is not the time when dj completes the execution of its last assigned operation不一定是完成最后一个任务: it is possible for our algorithm to **insert an operation** into an earliest idle空闲 time slot between two already-scheduled operations on a device; the length of the idle time slot should be sufficient to execute this operation, and inserting the operation into this idle time slot should preserve precedence优先 constraints(我也不知道优先约束是啥); avail[j] is the start time of such a timeslot. We use ST (oi) and FT (oi) to represent the actual execution start time and execution finish time of operation oi .

问题1:算法5.1是否是我理解的这样? est的max意思是不是就是选最早的一个?

不是, 是最late的那个.他得等到前继所有节点执行完，才能开始执行. 应该是因为要等所有参数到达了之后再执行. 

问题2: 什么是优先约束? 是不是就是操作有的要先有的后.

Our algorithm aims to minimize the overall actual execution time of operations on the computation graph’s critical path (based on their placement), which is the lower bound of the end-to-end execution time of the DAG (there could be gap time between operation executions). To compute the critical path, the entry operation is selected, and then we recursively select the operation with the largest rank among the successors. 就是递归选择后继.是不是时间长的先执行? 时间短的后执行? 不太对, rank大不代表时间长.rank代表的是从他这里开始，执行，一直执行到最后的sink节点，所需要花费的时间.它本身这个节点时间很短，他的rank也有可能很大.

起始节点是这个图中唯一一个入度为零的节点. 然后找这个起始节点相邻的所有的节点中，rank最大的.如果有多个入度为0的节点，你可以创建一个虚拟的节点，然后把这个虚拟节点指向所有其他入度为零的节点，然后这个起始节点就是entry

​	We consider operations in the DAG according to the order computed. If the operation is on the critical path, assign it to a **critical-path device**.这里讲了怎么选择关键路径设备,就是模拟然后计算一下 We choose a critical-path device as follows: for each available device, we simulate placing as many remaining operations on the critical path as possible onto the device (within its memory capacity), and compute the average execution time of the operations on the device using values from the computation cost model; we choose the device with the smallest average time as **a critical-path device**. If an operation is not on the critical path, we assign it to another device with sufficient memory which minimizes the EFT of the operation.因为关键路径是最费时的, 所以要用计算关键路径操作最快的设备,叫做critical-path device. During operation-device assignment, when a critical-path device’s memory **is full**, we find another critical-path device and assign as many critical-path operations to it as possible.直到这个设备满了的话找另一个critical-path device,然后assign 尽可能多的关键路径操作给它.  这一段很重要, 就是讲怎么计算device placement和执行顺序的.

​	Our Device Placement and Operation Sequencing (DPOS) algorithm is given in Alg. 1. We identify the following properties of DPOS. We use ωDPOS to represent the end-to-end processing time of the DAG. The time intervals in [0,ωDPOS ] can be categorized into two exclusive sets A and B:A includes all time intervals when all the devices are busy, and B includes intervals when at least one device is idle. B是有停的device的时刻.If B = ∅, DPOS is obviously optimal. Thus, we focus on the case where B , ∅. We assume B is the union of N intervals: We use O to represent the set of all operations in the DAG.

```python
# Algorithm 1 Device Placement and Operation Sequencing (DPOS) 
1: Input: Graph G(O,E); Device Set D; Computation Cost Model Ccomp; Communication Cost Model Ccmmu;
2: Output: New Device Placement Strategy Snew; Execution Order List A[]; Finish Time of Exit Operation FT(oexit).
3: Set wi to be the max execution time of operation i and ci,j to be the max communication time between operations i and j .
4: Compute ranku, critical path SETCP. 
5: Select a device set dCP to place operations in critical path based on average computation time and memory capacity.
6: Create priority queue L for operations by decreasing order ofranku values.
7: while L is not empty do    
8:	oi ← L.dequeue()
9: if oi ∈ SETCP then
10: Snew[oi ] = dCP(oi )# 
11: else
12:  for d in D do
13: 	if memory need of oi exceeds capacity of d then
14:  EFT(oi ,d) ← +∞  
15: else  
16: Compute EFT(oi ,d) 
17:  end if   
18:  end for 
19:Snew[oi] = arg minEFT(oi ,d) d∈D
20:FT(oi) = EST(oi,Snew[oi])
21: end if 
22: end while 
23: Compute Execution list A by sorting operations in ascending order of ST(oi )
24: Compute FT(oexit)=EFT(oexit,Snew[oexit]) 
25: Return: Snew, A, FT(oexit)
```

```python
 ranku值降序创建优先级队列 L。
while L is not empty do
	oi <- L.dequeue() # 根据优先级逐一分配.
    if oi in SETcp:
        Snew[oi] = dcp(oi)# oi的策略就是用关键路径cp device
    else:
        for d in D: #D是device的set
            if oi 进来后d的memory超出:
                EFT(oi,d) =  +∞  
            else:
                compute EFT(oi,d)
        Snew[oi] = min(EFT(oi,d)) #不同d中最小的eft
        FT(oi) = EFT(oi,Snew[oi])
计算Compute Execution list A by sorting operations in ascending order of ST(oi)
Compute FT(oexit)=EFT(oexit,Snew[oexit]) 
```

Lemma 引理1 There exists a chain X : oi_m → oi_m-1 → . . . → oi1 in O that covers B, if the memory capacity of devices is sufficient to host operations assigned. 如果有容量, 那么一定存在一个chainX可以coverB. That is, the total execution time plus maximal overall data transmission time along chain X is no less than the total duration of B:

Theorem 1 The end-to-end processing time of the DAG, ωDPOS , satisfies: ωDPOS ≤ 2ωopt + Cmax , where ωopt is the optimal DAG execution time in an ideal system without tensor transmission time, and Cmax is the maximal overall data transmission time along any chain in O. The detailed proofs are given in the Appendix. DAG 的端到端处理时间 ωDPOS 满足： ωDPOS ≤ 2ωopt + Cmax ，其中 ωopt  optimal是没有张量传输时间的理想系统中的最佳 DAG 执行时间，Cmax 是沿 O 中的任意链的最大总数据传输时间。详细的证明在附录中给出.

## 5.2 Operation Splitting操作分解

​	The DAG execution time may be further reduced by **splitting operations on the critical path into sub-operations**, for further parallelism to reduce the overall execution time of the critical path. Different types of operations have different dimensions to be split. For example, Conv2D can be partitioned on the batch size dimension for fine-grained data parallelism within the operation, and also on the channel dimension to achieve fine-grained model parallelism. Splitting operations does not change training semantics through graph modification, hence resulting in no model accuracy loss. We propose our second heuristic OS-DPOS (Operation Splitting Device Placement and Operation Sequence) to perform operation splitting based on DPOS. 

​	The input graph to Alg. 2 is decided as follows: if the model is too large to be fit into a single device, we input the DAG of the model; otherwise, we construct a data parallel graph based on the model DAG as the input, where the model is replicated as many times as the number of devices 一开始数据并行但不是纯数据并行(i.e., we adopt data parallelism as our start deployment strategy in order for the algorithm to identify a better strategy beyond pure data parallelism). In the algorithm, a function SplitOperation is invoked to **generate the updated graph** when an operation is split on a specific dimension with a certain split number. As an example, here we only show one split method which is suitable for some types of operations (e.g. , suitable for Conv2D and not for BatchNorm). Different split methods are available for splitting other types of operations [7, 26, 27].

The algorithm first invokes Alg. 1 to compute **an initial device placement and execution order.**也就是代码里第三行 Then it calculates the new critical path based on the placement strategy and splits the operations along the critical path in descending order of their computing time. 按降序拆分,就是把最长的时间给拆分. For a specific operation, Alg. 1 is called to compute the corresponding critical path, device placement and execution order after splitting it on each dimension and with each split number, and the best split of the operation which achieves the smallest FT of the exit operation in the DAG is identified. Only if this time with the best split is smaller than before splitting, the algorithm records the corresponding best split dimension and split number, and adds the decisions to the split list; otherwise, the algorithm stops the loop and no longer explores the remaining operations on the critical path. 先计算初始device placement 和执行顺序, 然后计算 新关键路径.

```python
# Algorithm 2 OS-DPOS 
1: Input: Graph G(O, E); Device Set D; Computation Cost Model Ccomp; Communication Cost Model Ccmmu;
2: Output: Operation Split List SP[]; New Device Placement Strategy S; Execution Order List A;
3: Compute Snew; A[]; FTold(oexit) using DPOS(G,D, Ccomp,Ccmmu). # 就是init 
4: Compute Critical path (CP) based on Snew and G. 
5: sort CP by descending order of computation time. 
6: Initialize SP ← [];Ginit ←G(O, E);S ← Snew 
7: for operation op in CP do 
8: With different d ∈ parallelizable dimensions and n∈ # of GPUs(D), call DPOS(SplitOperation(Ginit, op,d, n), D,Ccomp,Ccmmu,S) and record the smallest FT(oexit) and corresponding dimension d, split num n, Snew and Anew.
9: if FT(oexit) < FTold(oexit) then
10:Update: FTold(oexit) ← FTnew(nexit), Ginit = Gnew, S ← Snew, SP ← SP ∪ (op,d, n), A ←Anew #把拆分操作 (op,d, n) 放在SP中
11:else 
12:	break # stops the loop and no longer explores the remaining operations on the critical path.
13: end if 
14: end for 
15: Return: SP, S, A.
16: function SplitOperation (Graph:G(O,E), Operation:op, Dimension:d, Split num:n)
17: for i ← 1, 2, ..., n do
18: Create new sub-operation si
19: end for 
20: for operation pre ∈ predecessors(op) do
21:add a split node sp and connect it to the n partitions split from edge (pre, op) on dimension d: p1,p2, ...,pn.
22: connect pi to si .
23: end for 
24: for operation suc ∈ successors(op) do
25:add a concatenate(= connect) node con that concatenates s1, s2, ..., sn.
26: connect con to suc.
27: end for 28: Remove operation op and edges connecting to it. 29:
Return: Updated graph Gnew 30: end function
```

```python
Input: Graph G(O, E); Device Set D; Computation Cost Model Ccomp; Communication Cost Model Ccmmu;
Output: Operation Split List SP[]; New Device Placement Strategy S; Execution Order List A;
for op in CP:
    对于每个不同维度d和不同的GPUn, 调用DPOS(SplitOperation(Ginit,op,d,n),D,Ccomp,Ccmmu,S) 然后记录最短的FT(Oexit) and 对应的dimension d,  split n, Snew 和Anew(不知道是怎么split的)
    if FT(oexit) < FTold(oexit):
        Update: FTold(oexit) ← FTnew(nexit), Ginit = Gnew, S ← Snew, SP ← SP ∪ (op,d, n), A ←Anew 
    else:
        break
return SP,S,A      
function SplitOperation (Graph:G(O,E), Operation:op, Dimension:d, Split num:n)
	for i in range(n):
        创建子操作si
    for pre in op的前继结点:
        add a split node sp and connect it to the n partitions split from edge (pre, op) on dimension d:p1 p2 p3 (不懂 split node是什么)
        connect pi to si
    for operation suc in op的后继结点:
        add a connect node
    	connect con to suc
    remove operation op and edges connecting to it .
    return updated graph Gnew
```

It is noteworthy that FastT may not use all the input devices, and can choose a subset which achieves better performance than using all. Strategy calculator in FastT carries out Alg. 1 to derive device placement and execution order of all (sub-)operations.

## 6 Implementation and Evaluation 

### 6.1 System Implementation

We implement FastT over TensorFlow 1.14. 

**Strategy Calculator** is built in Python client of TensorFlow (1660 LoC in Python). We add the control logic inside **the initialize function and run function** of class BaseSession. It is the entry point to invoke TensorFlow C++ core runtime from Python, and most high-level Python APIs are based on the BaseSession class. Therefore, model developers can transparently use our module with their existing models. The strategy calculator activates TensorFlow profiler for updating the cost models and computes new strategies in the run function of BaseSession using a single CPU core.  修改基类中的函数实现transparent

**Device Placer** is simple module implemented with 20 LOC in Python. It first checks the co-location constraints of operations and then uses built-in functions of TensorFlow to implement the device placement. When the training is done over multiple machines, we use **in-graph** [4] implementation so that a single global computation graph can be placed on these machines.

**Cost Model.** We extend TensorFlow internal tracer to fetch the raw meta-data of each operation during the training process (198 LOC in Python), for building the cost models. **Order Enforcement** is implemented within the executors in TensorFlow C++ runtime (107 LOC). By default, the runtime scheduler executes the operations in the ready queue following FIFO (First-In-First-Out). We set each operation with a priority according to the execution order computed by the strategy calculator, and schedule operations according to their priorities. We used to directly add control dependency to enforce execution order, which adds strong constraints in the graph, loses the chance for further optimization (such as the graph pruning by TensorFlow), and sometimes leads to poor performance. We hence exploit the priority-based method to provide the scheduler more flexibility, while satisfying control dependencies.  给每个操作根据执行顺序设置优先级.

​	Since we directly modify the code in TensorFlow's `Session.run` function to take over the control of all following processing, the developers do not need to change a single line of their model code, when using the TensorFlow framework compiled with our modules.

### 6.2 evaluation methodology

**Testbed setup.** We deploy FastT-boosted TensorFlow framework in physical machines, each equipped with 8 NVIDIA Tesla V100 GPUs with NVLinks, where each GPU has 16GB.memory, and 2 Intel(R) Xeon(R) Platinum 8163 CPUs, where each CPU has 24 cores. 

**Benchmark models.** We experiment with 5 CNN models (VGG19 [39], ResNet200 [24], AlexNet [29], LeNet [1] and Inception-v3 [40]) and 4 NMT models (Transformer [43], Bert-large [17], GNMT [46] and RNNLM [47]). 

**Baseline strategies.** We use data parallel (DP) strategies and results of REINFORCE [32], GDP [48], FlexFlow [27] and Post [18] as baselines. For data parallelism, we adopt default data parallel implementation in TensorFlow slim [3], and compare the performance under both strong scaling (which retains the same global batch size when the number of GPUs varies) and weak scaling (which retains a fixed batch size at each GPU). For REINFORCE, GDP, FlexFlow and Post, we compare the strong scaling performance with results extracted from their papers (they all adopt strong scaling): REINFORCE, GDP and Post need tens of servers to compute their policies; the available source code of Flexflow only includes the part of applying a given strategy but not the code for running their search method to find the strategy, and is hence not directly usable for experimental comparison.

We use **training speed (samples/second)** rather than the per-iteration training time as the performance metric, because in weak scaling, the global batch size grows with the number of GPUs, and as a result per-iteration times cannot be directly compared.每次迭代时间不同,性能指标是样本/秒.  Since our method preserves the semantics of model training, and does not change the number of iterations to converge for each model, we do not show the total iteration number in our evaluation. In strong scaling, we choose the global batch size to fully utilize a single GPU to ensure no out-of-memory (OOM) when using only one GPU for training; in weak scaling, we choose the per-GPU batch size to fully utilize a single GPU without incurring OOM. All our results are averaged over 500 iterations after a warm-up of 10 iterations.

### 6.3 Performance of FastT 

**Per-iteration speed-up.** In Table 1, we see that with strong scaling, FastT outperforms default data parallelism in most cases, and achieves up to 59.4% speed-up when training VGG using 4 GPUs. With more GPUs (e.g., 8), the performance of both strategies may degrade due to more communication overhead among model replicas and smaller batch size per GPU which cannot achieve good GPU utilization, but FastT still does better. In the case of 8 GPUs (2 servers), we experiment in a distributed setting with 4 GPU cards each on two servers, and include inter-server communication time into our cost models. The improvement of FastT over the default strategy is in general better in this distributed setting, than with all 8 GPUs on the same server. This is because the default strategy performs worse in a multi-server setting than on the same server, while FastT can find better solutions by capturing the communication overhead across servers using the communication cost model. FastT可以capture通讯开销来改进.

With weak scaling, the performance of data parallelism in Table 2 is similar to the performance reported in DAWN-Bench [15] and NVIDIA Benchmark [8].We see that FastT still performs better than data parallelism, and a 19.2% speed-up when training VGG in the case of 16 GPUs (2 servers). As compared to the speed-up in Table 1, the improvement over data parallelism is smaller, which is because the utilization of each GPU with data parallelism is much higher than in the strong scaling setting, leaving us a much smaller optimization space by moving operations around across the devices.数据并行性的改进较小，这是因为具有数据并行性的每个 GPU 的利用率远高于强扩展设置下，通过在设备之间moving operation 的优化空间小.

We observe that the improvement with FastT is better with Bert-large than Transformer. The Transformer model can be fit into a single GPU with the standard batch size, so that data parallelism performs pretty well already. When training Bert-large, the batch size per GPU is much smaller, as otherwise out-of-memory (OOM) errors occur; hence training Bert-large with data parallelism may not do well due to the underutilization of GPU computation capacity with the small batch size. On the other hand, FastT can find better solutions of placing most operations in the model in one GPU, to better utilize GPU computation capacity while minimizing inter-GPU communication.

Unless otherwise stated, our following experiments are based on strong scaling, and the global batch size used to train a model is the same as indicated in Table 1.

Support larger batch size for very large models. For bert-large, its model cannot be fit in a single GPU when batch size is larger than 16. We set the maximal sequence lengths in bert models to be 64. Table 3 shows that with FastT,we can efficiently exploit 2 GPUs to train it with larger global batch sizes (e.g. , 48), while data parallelism can only support global batch size of 32. In addition, developers do not need to worry about manual placement of such a large model between devices. 

**Order Enforcement.** We evaluate the performance gain brought by operation execution ordering, and compare with TensorFlow’s default execution order. In TensorFlow, the executor chooses operations from a ready queue using FIFO. In Fig. 2, each model is run using 2 GPUs. We see that per iteration time is reduced by up to 26.9% when order enforcement is enabled. 订单执行或者顺序执行

**Time for strategy calculation.** Table 4 shows the time needed to compute placement and execution order with Alg. 2 in FastT, which is within **several minutes for most models**. It takes more than 1 hour to compute the strategies for deploying the Transformer model over 8 GPUs, due to the very large number of operations in the model. Besides, the strategies are computed through real model training, such that the strategy search time includes profiling time and system restart time (for activating changed strategies). Still FastT can compute the strategies using much less time and resources than existing approaches such as REINFORCE and GDP.

### 6.4 Comparison with other strategies

 We next compare the speed-up of FastT with REINFORCE, GDP, FlexFlow and Post. We use strong-scaling data parallelism as the baseline, and show each strategy’s processing speed divided by that of the data parallel strategy in Fig. 3. The models being evaluated are those with results in the respective papers as well. FastT outperforms REINFORCE, GDP and Post in all respective cases, as REINFORCE, GDP and Post do not consider data parallelism and operation split, and hence their solution spaces are limited. FlexFlow may find a better solution than FastT, due to its larger solution space and extensive search-based algorithm to find the strategy. However, FastT’s performance is close; being compatible with TensorFlow, it is more generally usable. Further, the time complexity of FastT is linear with the number of operations and devices, while the search space in FlexFlow increases exponentially with the increase of operations and devices. 好处: 兼容, 更普遍可用. 时间复杂度线性.

### 6.5 Analysis of result placements

**Operation placement.** Fig. 4 shows the number of operations assigned to each GPU with FastT. Different from pure data parallelism that assigns model replica to each GPU, FastT does not always allocate operations evenly among GPUs. In the case of 4 GPUs, one GPU has many more operations while the numbers on others are pretty even. Our investigation shows that replicas of operations with large parameters are placed in one GPU rather than 4 GPUs, to avoid inter-GPU aggregation(问题:  我不明白这个inter-GPU是啥意思,是不是因为传输这些梯度代价大,好像是的看下面说广播参数到所有副本的开销) of gradients of these parameters during training. For other computation-intensive operations, they are evenly placed onto 4 GPUs to reduce end-to-end processing time, which implies that the computation time saving due to data parallelism exceeds the cost for aggregating gradients across 4 GPUs for these operations.  我们的调查表明，具有大参数的操作的副本放置在一个 GPU 中而不是 4 个 GPU 中，以避免在训练期间这些参数的梯度在 GPU 间聚合. 对于其他计算密集型操作，它们被均匀地放置在 4 个 GPU 上以减少端到端处理时间，这意味着由于数据并行性而节省的计算时间超过了为这些操作在 4 个 GPU 上聚合梯度的成本。

**Operation split.** Table 5 shows the split decisions for some representative operations in Vgg-19, as made by FastT, together with their execution time (before splitting) and parameter sizes. We can see that in Vgg-19, some conv operations have longer execution time than others, so they are most likely to be split. Fc operations with **large parameter sizes are not split**, to avoid overhead of broadcasting parameters to all replicas. Operations being split usually have longer execution time and smaller parameter size, 拆分的时间长但是参数sizes小. to strike a good trade-off between computation performance gain and extra communication overhead incurred by the split.

Table 6 compares model training performance when we enable operation split in FastT and not. The experiments are done under the settings achieving the best speedup as in Table 1. We see that with CNN models such as Inception, Vgg and ResNet, Conv2D and Conv2Dbp are the key operations whose splits bring performance gain. However, for LeNet and AlexNet, these operations are not split due to small input tensor sizes to them (such that these operations’ computation time is small). Further, operations in LSTM-based NMT models (GNMT and RNNLM) are not split because no computation intensive operation is found. For attention-based models (Transformer and Bert-large), MatMal operations are split, which are the most computation-intensive operations in these models. 这段就讲了各种模型拆分哪些操作

**Time Breakdown.** We show the average computation time and memory copy time (i.e., tensor transfer time) when training the models using pure data parallelism and FastT on 2 GPUs in Fig. 5. Due to overlap of computation and memcpy (communication), the overall per-iteration training time is usually not equal to the sum of computation and memcpy time. We observe that even though **the computation time with FastT is increased, its memcpy time and per-iteration time are reduced**. Main reasons are as follows. With data parallelism, operation replicas require gradients from other replicas in each iteration, which involves memory copy since the replicas are assigned to different GPUs. FastT can reduce memcpy cost by assigning some replicas of an operation to the same GPU (as validated by its uneven operation assignment among all GPUs), which on the other hand may increase GPU time due to processing more operations on some GPU.对于数据并行性，操作副本在每次迭代中都需要来自其他副本的梯度，这涉及内存复制，因为副本被分配给不同的 GPU。 FastT 可以通过将一个操作的一些副本分配给同一个 GPU 来降低 memcpy 成本（正如其在所有 GPU 之间不均匀的操作分配所验证的那样），另一方面，由于在某些 GPU 上处理更多操作，这可能会增加 GPU 时间。

## 7 Related Work

之前的缺点:  只考虑了计算图的模型并行, 黑盒学习时间久消耗计算资源多.

 **Device Placement for Deep Learning Models**. Researchers have been seeking the best placement strategy to assign operations in a DNN to different devices, to minimize execution time of the computation graph.  The Google team used reinforcement learning to tune a placement strategy [32]. Some follow-upwork propose more advanced algorithms to reduce learning time for deriving the policy [19, 21, 30], enlarge the solution space for better strategies [12, 26, 27, 44], optimize the reward function and sampling methods [18, 19, 33], or learn a more general model applicable to different computation graphs [10, 35, 36, 48]. For example, Placeto [10], GDP [48] and REGAL [35] use GNNs to generalize their models so that they can handle unseen computation graphs, and REGAL further considers the execution order of operations; however, these proposals only consider **model parallelism of computation graphs**, so the performance is limited. All the above studies treat the placement problem as a black box, and usually require hours of learning to obtain a satisfying policy, using large amounts of computing resource for policy training. Stanza [45] separates CONV layers and fully-connected layers into different workers to reduce communication overhead; it only optimizes these two types of layers.我们可以优化所有层 DLPlacer [34] studies hybrid data and model parallelism, but its device placement is based on a subgraph of the model rather than the entire graph.没有考虑全图.

**Fine-grained parallelism within operations**. For neural networks such as a CNN, the fully connected layer is much larger than others; operations in that layer can be partitioned into several small sub-operations, and sub-operations can be assigned to different devices to reduce the execution time along the critical path. Alex [28] uses data parallelism for convolutional and pooling layers and switches to model parallelism for densely-connected layers to accelerate CNNs. TensorFlow mesh [38] provides high-level APIs for developers to specify parallelizable dimensions for different kinds of operations. They depend on developers to manually decide the parallelism strategy, which requires lots of experience. Tofu [44] utilizes a partition-n-reduce method to split a single operation into fine-grained operations, and a dynamic programming method to recursively optimize the partition. It does not consider device placement ofoperations. OptCNN [26] parallelizes CNN models by splitting operations along batch and channel dimensions; it does not consider parallelism across different operations.

**Pipeline parallelism for DNN training.** Chen et al. [14] use a pipelining strategy to update models with delayed data and allow to compute different layers concurrently. Wu et al. [46] accelerate computation ofRNN on GPUs in the pipeline manner. PipeDream [22] introduces a pipeline approach to reduce communication overhead for synchronized training with the parameter server architecture [31]. GPipe [25] uses pipelines to address the memory bottleneck for large NNs. However, pipeline parallel training usually does not retain the exact semantics of the original model: multiple versions of parameters exist during training (similar to asynchronous training), which may lead to prolonged model convergence, or convergence to a different accuracy. FastT does not have this problem when used for strong scaling. On the other hand, these pipeline strategies can be complementary to FastT. After FastT obtains operation placement and execution order, it can further split a mini-batch into micro-batches and allow pipelined training in the similar fashion as proposed in Gpipe. FastT 在用于强缩放时没有这个问题。 另一方面，这些流水线策略可以作为 FastT 的补充。 在 FastT 获得操作放置和执行顺序后，它可以进一步将 mini-batch 拆分为 micro-batch，并允许以类似于 Gpipe 中提出的方式进行流水线训练。

### 8 Concluding Discussions 

This paper presents FastT, a transparent module on TensorFlow to automatically find satisfying operation splitting device deployment and execution order for DNN models running over multiple GPUs. We carefully design the system architecture and propose efficient heuristics with theoretical performance bound. FastT achieves up to 63.6% speed-up as compared with pure data parallelism, and outperforms representative approaches as well in terms of per-iteration training time, strategy computation time and resource consumption, or generality. It is applicable to different types of DNN models and requires no modification of the origin ML code for developers using TensorFlow. Looking forward, we have noticed that some new features
have been published in TensorFlow which allow cycles in computation graphs, such as dynamic RNN layers. Currently, FastT does not handle graphs with cycles. 局限是不处理有环的图.A potential solution is to break the cycles and reorganize the graph to be a DAG. We leave this as future work. Further, we build most parts of the framework on TensorFlow’s Python Client API, so FastT currently supports developers who use Python to build their models. We will migrate our modules to TensorFlow kernel to support more APIs.

