# Fast Training of Deep Learning Models over Multiple GPUs

摘要

This paper proposes FastT, a transparent module to work with the TensorFlow framework for automatically identifying a satisfying deployment and execution order of operations in DNN models over multiple GPUs, for expedited加速 model training. We propose white-box algorithms to compute the strategies with small computing resource consumption in a short time. Recently, similar studies have been done to optimize device placement using reinforcement learning. Compared to those works which learn to optimize device placement of operations in several hours using large amounts of computing resources, our approach can find excellent device placement and execution order within minutes using the same computing node as for training. We design **a list of scheduling algorithms** to compute the device placement and execution order for each operation and also design an algorithm to **split operations in the critical path** to support fine-grained (mixed) data and model parallelism to further improve the training speed in each iteration. We compare FastT with representative strategies and obtain insights on the best strategies for training different types of DNN models based on extensive testbed experiments.

他和TensorFlow框架一起工作. 自动部署, 多个GPU可以顺利训练DNN。 开销小。 也可以用在reinforcement 学习。几分钟就可以安排好。 设计一系列调度算法（为什么？ 为了计算device placement和执行顺序）。 还有一个算法 切分关键路径的操作。支持细粒度数据和模型并行。

他是怎么想出来的？

# introduction

Deep Learning (DL) has become increasingly popular over the past years in various application domains such as computer vision, speech recognition and robotics. With increasingly complicated models and larger datasets, training of a deep neural network (DNN) model becomes an extremely time consuming job. Parallelizing the training using workers equipped with multiple GPUs or in a distributed environment is popular with current machine learning (ML) frameworks [2, 6, 9, 13]. 

Three of the most common parallelization strategies are data parallelism, model parallelism and pipeline parallelism. Data parallelism places a replica of the entire neural network (NN) on each device ( e.g. , a GPU card) so that each device processes **a subset of the training data** and synchronizes model parameters in different replicas副本 at the end of each training iteration. 每个设备有个训练集的子集。 Model parallelism handles models with a large number of parameters, which cannot fit into the device memory, by assigning **disjoint partitions** of an NN each to a different device.将NN的不相交的分区分别分配给不同的设备 Pipeline parallelism divides a DNN model into **stages** and places stages on multiple devices; it further divides each mini-batch批次 into microbatches, so different devices can process different microbatches simultaneously. To date, it is still not clear that given multiple GPUs, what the best strategy is to deploy a specific model onto the devices. Commonly, a practitioner may use data parallelism and replicate the model onto each GPU, but is this really always the best strategy even when a single GPU can hold the entire model?但是整个模型在单个GPU真的最好吗？ And how about large models that cannot be entirely replicated to a single GPU?

In this paper, we show that a mixture of fine-grained data and **model parallelism** combined with some heuristics is able to find a satisfying device placement in a fast manner with little resource consumption. We also seek to develop a software module to enable **automatic model deployment** without requiring model developers’ code modification, which can seamlessly无缝地 work with existing frameworks such as TensorFlow. For a small model which can be deployed in a single GPU, it should find a strategy achieving faster training than normal data parallelism, if there is one; for a large model which cannot be deployed entirely in a single GPU, it provides a good deployment across multiple GPUs.

 We propose FastT, a transparent module that automatically finds and activates a satisfying deployment and execution order of operations for different kinds of models in a multiGPU environment. Our contributions are summarized as follows:

▷ We propose new heuristics启发函数 that find deployment and execution orders in a few minutes, which are better than or as good as previous strategies that require hours to be computed. 更快 The main reasons lie in that we **extend the solution space** by considering operation split and execution ordering, and we use efficient **white-box heuristics** rather than search or learning-based methods to reduce strategy calculating time. FastT is efficient enough to be executed on one node (the same as a single worker used for model training), removing the need for an additional cluster for strategy search.

▷ We build adaptive cost models to facilitate our algorithms. To minimize profiling分析 overhead while obtaining accurate operation execution time on each device and interdevice communication time, we use **data parallelism** as the starting strategy (as long as it is feasible), try out different placements and apply linear regression回归 to obtain the communication cost model.

▷ We consider a larger solution space than previous approaches by considering both execution order and fine-grained parallelism within operations. We observed significant performance variation under the same device assignment with different operation execution orders.不同顺序性能变化很大 FastT decides execution order and achieves fine-grained parallelism by splitting some operations on the critical path to further improve the processing speed. Experiments show that FastT achieves up to 59.4% speedup compared with pure data parallelism with the larger solution space.

▷ We provide an open-source implementation of our method that transparently works with TensorFlow: developers do not have to modify their ML model to leverage our solution. FastT is useful for various models, and able to automatically calculate and activate placement and execution without involving the ML developer. We have built FastT based on TensorFlow: once the FastT module is turned on, developers can transparently use it with their existing models implemented with all kinds of TensorFlow’s Python APIs without modifying a single line of their code.

# 2 Background and Motivation

##  2.1 DNN Training and Parallelism

 Training a DNN is an iterative process which uses a large amount of data to tune model parameters for minimizing a loss function. In current training frameworks [2, 6, 9, 13], different kinds of computation are implemented by different operations (such as Conv2D, MatMul), and input and output of these operations are called tensors张量. The computing process can typically be represented by a DAG (Directed Acyclic Graph), whose nodes are operations and edges are tensors. 

**Data Parallelism.** The input data are partitioned to different devices. Each device shares the same model parameters. Gradients梯度 from all devices are applied to update the global model. Data parallelism can be applied in a single worker/machine with multiple GPUs [19, 26, 27], and among multiple machines [37]. 

**Model Parallelism.** The input data are sent to all devices without partition; each device is responsible for tuning a different part of the model parameters. Model parallelism is typically used for models with a large parameter size [26, 34, 45]. 每个设备调一部分参数. 适用于 参数size很大的models.

**Pipeline Parallelism.** Pipelining has been proposed to accelerate DNN training with multiple accelerators [12, 14, 22]. Many DNN models stack layers sequentially; naive model parallelism may result in only one active accelerator anytime during training. With pipeline parallelism, similar to model parallelism, different layers are deployed on different accelerators; a mini-batch is further divided into several micro-batches and these micro-batches can be processed at different layers at the same time to fully utilize all accelerators.

## 2.2 Fine-grained device placement

 **Operation-level** device placement. With model parallelism, a model is typically partitioned in the **layer level**. A layer consists of multiple operations. To expand the solution space, some operation-level approaches are proposed [19] to decide device placement of each operation separately.   更小粒度, operation level和layer level. 都属于模型并行

Parallelism within operations. To further extend the solution space, some studies [11, 16] investigate potential parallelism within individual operations. For example, for a Conv2D operation, it can be further parallelized by being partitioned on the batch size dimension or the channel dimension [27]. Such an approach can be regraded as fine-grain mixture of data parallelism and model parallelism according to different parallelizable dimensions of different operations. 

## 2.3 Limitations and Challenges 

Previous research [28, 46] has proposed strategies to manually optimize parallelism based on human experts’ domain knowledge and intuitions. For example, Krizhevsky [28] uses data parallelism for convolutional and pooling layers and switches to model parallelism for fully-connected layers to accelerate the training of convolutional NNs (CNNs). In addition, some automated frameworks [19, 23, 27, 32] are proposed for finding efficient parallelism strategies in a limited search space. REINFORCE [32] uses a reinforcement learning method to learn efficient operation assignments on multiple GPUs. TicTac [23] explores the impact of the order of send/recv operations in distributed training. Allowing fine-grained parallelism within a single operation, FlexFlow [27] builds a new training architecture to explore the SOAP (Sample-Operation-Attribute-Parameter) search space considering parallelism within and across operations. Some other frameworks focus on specific types of networks such as CNN and RNN (Recurrent循环 Neural Network), and provide APIs for developers to split operations by themselves such as TensorFlow mesh [7] and tofu [44].

The existing proposals have the following limitations: 

First, the purpose to find optimal device placement is to save time and computing resource for a training job, and the finding process itself should not be time and resource consuming. Some existing approaches require a large amount of resources and spend a long time to obtain the strategy. For example, REINFORCE [32] and GDP [48] use another big cluster consisting of tens of workers and spend hours on learning the device placement policy 找方法很费资源很花时间

Second, existing approaches may not be generic通用的 enough for different kinds of models or not compatible with popular training frameworks. For example, OptCNN [26] is designed for CNNs. FlexFlow implements the training framework itself, and does not support representative frameworks such as Tensorflow or MXNet.

Third, the solution space can be particularly large. Most existing studies only consider device assignment of operations, but not the execution order of operations.除了分配外, 顺序也很重要 For example, FlexFlow defines a fine-grained search space beyond data and model parallelism, and schedules operations in the ready queue with a simple FIFO strategy.

We seek to address the following challenges in this paper: 

▷ We consider not only device placement of operations, but also partitions of operations and their execution orders; the solution space becomes much larger than what the existing studies tackle. Finding a satisfying solution in a timely manner with small computation resource consumption is critical for solution adoption in a production environment. 研究怎么切分operations和 安排顺序. 解空间特别大.

▷ It is easier to design strategies for a specific type of models. However, a generic approach needs to analyse the structure (DAGs) of different models and builds the respective cost models  模型怎么通用

▷ Since ML developers may use various APIs to implement their models (even when they are using the same ML framework such as TensorFlow), it is hard to design a unified software module to transparently support their existing models without modification. 怎么实现transparently支持

A practical model deployment and execution module must be fast, light-weight, generic and compatible with existing training architectures at the same time.

# 3 Problem Definition 

We first formally define the problem we intend to solve. The objective is to find a good device placement and execution order to achieve parallelism across operations in a DNN model, and also identify potential operations which can be partitioned into several sub-operations to achieve fine-grained parallelism within individual operations. The input of the problem includes: (a) the DAG computing graph, (b) the set of devices (GPUs) and memory limitation of each device, and (c) the cost models for computation and communication. The computation cost model provides computing time of a given operation on a specific device, and the communication cost model gives inter-device tensor communication time of adjacent operations running on different devices 一个计算开销模型, 一个沟通开销模型.

The output solution consists of three parts: (i) a partition list of operations which should be partitioned (each item in the list has three elements, the operation’s name, partition dimension, and the number of partitions); (ii) device placement of each non-partitioned operation and each sub-operation (due to splitting operations in the partition list); and (iii) execution order of operations and sub-operations.  一个list写怎么分, 然后怎么分配device, 第三个是执行顺序.

It should be noted that we focus on NN whose computing graph is a DAG. Some networks can be implemented as a graph with cycles in TensorFlow, e.g. a dynamic RNN which includes a while loop, and whether to exit from the loop is decided during runtime. For such a model, we optimize exection执行模式 of the DAG within each of its loops.  如果是环, 就每次loop优化exection of DAG

The problem of deciding execution order and placement of a DAG with unit operation execution time is known as **the single execution time scheduling problem**, which is NP-complete [42]. Our problem poses even greater challenge as we assume heterogeneous operation execution time每一次操作时间还不一样. Therefore, we propose efficient heuristic algorithms in Sec. 5 to find a good solution of our problem.

# 4 System Design

FastT is built based on TensorFlow, addressing parallelism both within and across operations in a DNN model. It calculates both device placement and execution order for each operation in the computation graph, with operations potentially further partitioned.

Fig. 1 illustrates how FastT fits into the architecture of TensorFlow, and coloured blocks represent components we implement for FastT. The strategy calculator (算device placement和执行顺序)computes device placement and execution order for the current model using the algorithm to be introduced in 5.2.  The device placer (分配任务)assigns different devices to run different (sub-)operations according to the strategy computed by the strategy calculator.  In the dataflow executor, order enforcement执行器 is responsible for organizing the execution order of (sub-)operations. The cost models component记录执行时间和传输时间 records the execution time of (sub-)operations executed on different devices and the data transmission time when adjacent operations are handled on different devices.

The workflow of FastT is as follows: Initially, FastT requires several pre-training steps to bootstrap引导,自引 the cost models, which uses different placement strategies to run the DNN model to update its cost models. To activate a new strategy computed with the updated cost models, the training session does checkpoints of current model parameters, and restarts to create a new graph based on the operation partition lists; then the device placer activates the device placement and the order enforcement module enforces the execution order of (sub-)operations. The training session then restarts with restored parameters from the checkpoints. After a new strategy is activated, FastT records the per-iteration model training time; if it finds that the per-iteration execution time with the new strategy is even longer than the previous one, it rolls back the strategy to the previous one. When the cost models become stable (the average time of the same (sub- )operation(s) on the same device(s) does not vary much), we finish the pre-training stage. Afterwards, the model is trained normally using placement and execution order strategies computed by the strategy calculator, and the cost models are updated only when the execution times have changed significantly based on our periodical profiling. 第一步cost models 预热,  第二步根据 操作分割list 创建新图, 然后分配设备和顺序. 记录每次时间, 时间更长就回滚上一个策略.









