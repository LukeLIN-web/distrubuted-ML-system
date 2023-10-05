# HeteroG

This paper proposes HeteroG, an automatic module to accelerate deep neural network training in heterogeneous GPU clusters. To train a deep learning model with large amounts of data, distributed training using data or model parallelism has been widely adopted, mostly over homogeneous devices (GPUs, network bandwidth). Heterogeneous training environments may often exist in shared clusters with GPUs of different models purchased in different batches and network connections of different bandwidth availability (e.g., due to contention). Classic data parallelism does not work well in a heterogeneous cluster, while model-parallel training is hard to plan. HeteroG enables highly-efficient distributed training over heterogeneous devices, by automatically converting **a single-GPU training model to a distributed one according to the deep learning graph and available resources**. HeteroG embraces operation-level hybrid parallelism, communication architecture selection and execution scheduling, based on a carefully designed strategy framework exploiting both GNN-based learning and combinatorial optimization. We compare HeteroG with existing parallelism schemes and show that it achieves up-to 222% training speed-up. HeteroG also enables efficient training of large models over a set of heterogeneous devices where simple parallelism is infeasible.

### 1 introduction

Deep Learning (DL) models have become increasingly complicated and large over the past years. Training of a deep neural network (DNN) is extremely time consuming. Parallelizing training using multiple workers in a distributed environment is adopted with current machine learning (ML) frameworks [1, 5, 44]. Two most common parallelization strategies are data parallelism
(DP) and model parallelism (MP) [9, 40, 59]. With data parallelism, a replica of the entire neural network is placed on each device (e.g., a GPU card); each device processes a subset of the training data and synchronizes model parameter updates among different replicas. For models with large parameter sizes that cannot be fit entirely into a single device’s memory, model-parallel training is adopted by assigning disjoint partitions of the DNN to different devices; no gradient aggregation is needed, but intermediate activations should be transferred across devices. Performance of model parallelism highly depends on the model-to-device assignment decisions made by ML developers.

​	To optimize distributed training, Krizhevsky et al. [30] and Wu
et al. [60] manually optimize parallelism based on human experts’ domain knowledge. Some automated frameworks [13, 39] were proposed for finding efficient model parallelism strategies. GDP [64] and Placeto [2] use Graph Neural networks (GNN) to learn operation-to-device assignment strategies. Parallax [28] utilizes hybrid PS and AllReduce communication methods in data-parallel training. All of them focus on training over homogeneous devices. 他们是homo我们是hetero

​	Instead, we focus on DNN training expedition **in heterogeneous environments.** In shared ML clusters containing GPUs of different models and many DL jobs, a new-arrival training job often faces the following situation: GPUs of its desired type are not available at its required number, while there are available GPUs of other models. With standard data parallelism, the job may have to wait for its required number of GPUs of the same model become available, or make do with the fewer number of GPUs available. The job cannot exploit available GPUs of different models due to the poor performance of data-parallel training over heterogeneous devices: the processing speed is imbalanced over different devices, and the devices and communication channels (network links across servers and internal links among multiple GPUs within a single server) are less efficiently used with synchronous training (due to waiting).gpu数量不够, 类型不一致, 

​	In data-parallel training, parameter server (PS) architecture [31]and AllReduce methods [36, 45] are widely used for parameter synchronization. In homogeneous environments, AllReduce usually performs better than PS [28, 31] by fully utilizing the links among all devices; in a PS architecture, the links to parameter servers may become the bottlenecks. In a heterogeneous environment, a single PS or AllReduce operation at the end of each training iteration for aggregating all parameter updates may be less efficient: parameter synchronization (aka communication) now takes longer time due to imbalanced computation speeds among devices, and low utilization of the communication channel results. 参数通信时间长, 

​	We advocate fine-grained, hybrid parallelism and parameter
synchronization methods among operations in a DNN model, for training acceleration in heterogeneous ML clusters. We propose HeteroG, an automated module that converts a single-GPU training model to a distributed one, achieving optimized training speeds. HeteroG generates detailed parallelism strategy, device placement, gradient communication method, and execution order for each operation in a DNN model. A novel strategy framework is designed incorporating a GNN for deciding operation parallelism, placements and communication schemes, and a combinatorial optimization problem to generate execution order of operations. Main contributions of this paper are summarized as follows:



▷ We propose an automated module to generate hybrid, operation level parallelism schemes for expedited distributed DNN training in heterogeneous environments. 为什么可以自动? hybrid是哪里hybrid?

▷ We design a novel strategy framework including a GNN-based policy network and a combinatorial optimization problem组合优化问题, with synergy协同效应 to comprehensively produce the large set of strategies enabling highly-efficient distributed training.

▷ To provide generality, a carefully designed GNN is used to
learn structural information from different DNN graphs, and produce good deployment strategies for a broad range ofDNN models. It decides the replication number of each operation and the device placement of these replicas to fully utilize different devices and communication channels. For replicas where gradient aggregation is needed, it also decides the communication methods (PS or AllReduce), enabling different communication modes for different gradient aggregation operations in a DNN model. 

▷ An efficient heuristic is designed to solve the combinatorial optimization problem on operation execution ordering. It ensures high resource utilization (GPU devices, network links) and maximal computation-communication overlap, with proven performance bound to optimal schedule.

▷ HeteroG is implemented as a python module in Tensorflow.
Developers only need to implement single-GPU models and invoke HeteroG’s simple API. HeteroG automatically generates a distributed training model with the strategies it finds, and deploys it in the heterogeneous cluster. 

▷ We carry out extensive experiments in a heterogeneous cluster. HeteroG is carefully compared with existing parallelism schemes: it achieves up-to 222% training speed-up as compared to data parallelism and existing hybrid parallelism designs; it can enable efficient training of large models over heterogeneous devices where simple parallelism is infeasible. We observe that a fine-grained hybrid of parallelism strategies and gradient aggregation methods for different operations, as well as variable replica numbers across heterogeneous devices, contribute to the good performance of HeteroG, based on very efficient utilization of available computation and communication resources.

### 2 BACKGROUND AND MOTIVATION 

2.1 DNN Training and Parallelism 

Training a DNN is an iterative process that uses a large number of
samples to tune model parameters for minimizing a loss function. In current training frameworks [1, 5, 44], different kinds of computation are implemented by different operations (such as Conv2D,MatMul), and input and output ofthese operations are called tensors (e.g., gradients, activations). The computing process can typically be represented by a DAG (Directed Acyclic Graph), whose nodes are operations and edges represent tensors.

**Forward and Backward Computation.** In each training iteration, one batch of samples is fed into the DNN model. Operations in forward propagation (FP) takes output of precedent operations as input and generates output based on parameters. A loss is produced based on outputs at the end of FP. After FP, gradients of model parameters are computed from back to front, i.e., backward propagation (BP). The gradients are then applied to the parameters using some optimization algorithm, e.g., Stochastic Gradient Descent (SGD).

**Model Parallelism (MP).** Operations in the DNNmodel are placed on different devices [9, 40, 59]. Each device maintains part of the parameters of the model. **Data Parallelism (DP).** The dataset is partitioned into mini-batches for training at each device. Each device maintains a replica ofDNN model and carries out FP and BP, and gradients from different devices need to be aggregated before applied to update parameters. 

**PS and AllReduce.** They are two popular architectures for parameter synchronization in data-parallel training [31, 36, 45]. In a PS architecture, parameters are stored in centralized parameter servers; each worker computes its gradients based on its local dataset and parameters, pushes the gradients to PSs and pulls updated global parameters from PSs. In an AllReduce architecture, each worker computes gradients and aggregates gradients from other workers for parameter updates using an AllReduce algorithm [36, 45]. 

**Communication in MP and DP.** With model parallelism, when two adjacent operations are placed on different devices, the output of precedent operation needs to be transferred to successor operation; if the two devices are in different physical servers, network communication is involved. With data parallelism, communication occurs during gradient aggregation/parameter synchronization.

2.2 Potential Training Expedition Methods in Heterogeneous Clusters

**PS could be better than AllReduce.** 异构的时候参数服务器更好Fig. 1 shows that a single AllReduce architecture for data-parallel training may perform well in a homogeneous cluster (GPU0, GPU1, GPU2 have the same computation power), but not in a heterogeneous environment (GPU0 is slower than GPU1 and GPU2 with computation power ratio of 1:2:2). Three adjacent operations in BP are considered, where GA represents gradient aggregation (following each BP operation). In case of imbalanced computation power of GPUs, the communication channel is not fully utilized, gradient synchronization takes longer time, and the training time is prolonged. 

​	In the heterogeneous setting, we can use the PS architecture for parameter synchronization and let the slowest GPU run both a worker and the PS functionalities. In this way, as shown in Fig. 2(a), communication for synchronizing parameters with the slowest worker is eliminated, and training is expedited. Note that in case of PS-based parameter synchronization, GA operation includes parameter push and pull to/from the PS; in this example, the GA operation at GPU0 (serving as the PS) starts when gradients are received from other devices. In a PS architecture, each worker can independently send their gradients to the PS. GA1 happens at GPU1 and GPU2 once they have finished their respective BP1, when their gradients are sent to GPU0; GA1 at GPU0 indicates receipt of these gradients from GPU1 and GPU2 (since GPU0 serves as the PS), which can start when the gradients are received and does not need to wait for the completion of BP1 in GPU0 itself. PS表现更好, 最慢的gpu作为PS接收

**Placing more replicas to faster devices.** Balancing workload among different devices can potentially lead to better utilization of computation power and communication channel. We can place more operation replicas in faster devices to achieve the balance. An operation can be replicated by dividing the input along the batch size dimension, i.e., each replica processes an even partition of the origin operation’s input and its execution time is shorter than the original operation’s. In Fig. 2(a), 3 replicas of each BP operation are processed on 3 GPUs; in Fig. 2(b), we make 5 replicas of each BP operation, and place a number ofreplicas in 3 GPUs in proportion to their computation power. In this way, we can still use AllReduce for gradient aggregation, as the GAoperations are largely synchronized without long waiting time, like in a homogeneous environment.给快的GPU放更多副本, 这样GA可以更加同步.

**Using MP to eliminate gradient communication.** With DP, communication occurs due to gradient aggregation among multiple replicas. We can place some operations on a single device without replication (model parallelism), to reduce some communication overhead. In Fig. 2(c), BP2 and BP3 are only placed on GPU1, such that parameters in these operations are only maintained on GPU1 and no gradient aggregation (of these parameters) is needed from other devices. The small yellow rectangle denotes activation transfer time to send/receive output of BP1 from other devices to GPU1. AllReduce is used for gradient synchronization among replicas of BP1 in this example. MP可以减少GA gradient 

2.3 Challenges

 Exploring the above opportunities comes with challenges.
PS may not be the one-for-all communication architecture in a heterogeneous cluster. In PS architecture for gradient aggregation, the links to parameter servers may become bottlenecks. PS中到参数服务器的通讯有瓶颈

Hybrid communication methods could provide a satisfying solution: use the PS architecture for aggregating gradients of operations where link bandwidth is not the bottleneck, while exploiting AllReduce for operations whose replicas’ computation is relatively balanced. However, it is difficult to judge which conditions the operations satisfy, which is closely related to placement of their replicas. 混合通信方法可以提供一个令人满意的解决方案：在链路带宽不是瓶颈的情况下，使用 PS 架构聚合操作的梯度，同时利用 AllReduce 进行副本计算相对平衡的操作。 但是，很难判断操作满足哪些条件，这与其副本的放置密切相关。

**Proportional distribution of whole-model replicas may not be sufficient.** We train VGG19 [49], ResNet [19], Inception-v3 [50], MobileNet_v2 [46] and Transformer [52] models respectively using DP on 4 GPUs (two Tesla V100 GPUs and two GTX 1080Ti GPUs), and compare result per-iteration training time ofplacing one model replica on each GPU vs. placing two model replicas on each Tesla V100 GPU and one replica on each GTX 1080Ti GPU (computation power ofthe two types ofGPU is roughly at the ratio of2:1). Fig. 3(a) shows that the speed-up with proportional workload allocation is small, about 9 ∼ 27%. We further investigate execution time ofsome representative operations in VGG19 and Transformer, when each is run on a Tesla V100 GPU and a GTX 1080Ti GPU, respectively. Fig. 3(b) shows normalized operation execution time by dividing the real time by that of running on the V100 GPU. The average speedup when using the V100 GPU varies significantly from 1.1 to 1.9; even for the same type of operations, the speed-up variance is also quite high, due to different input sizes. The large variation across operations implies that uniform proportional model replication among devices may not be efficient for training expedition; finegrained replica allocation at individual operation level could bring more efficient computation power usage for most expedited end to-end model training.

**Tradeoff in communication and computation overhead between DP and MP.** Though using model parallelism for some operations eliminates communication of their gradients, there exists data transfer for sending input into operations and dispatching output to other operations. Besides, completion time of the operations is longer, as compared to their parallel execution over multiple devices. It is difficult to decide whether to use DP or MP for an operation, which depends on the amount of data or gradient for transfer, computation power of devices to place the operations, etc.

Tackling these challenges, we carefully design a strategy framework to produce operation-level parallelism, placement, communication and scheduling strategies.

### 3 SYSTEM DESIGN 

3.1 HeteroG Overview

​	HeteroG is designed as a middleware between the client API and core processing engine in a state-of-the-art training framework (e.g., TensorFlow [1], MXNet [5]), to produce the best distributed training scheme for a given DNN model over a set of heterogeneous devices. HeteroG takes as input the DAG of the DNN and the device set, and produces a distributed execution graph with operations’ device placements, gradient aggregation methods and execution order. 给一个DNN , 可以优化不同类型GPU的训练,产生一个执行图, 执行顺序.

​	Fig. 4 shows the overall architecture of HeteroG. The Graph Analyzer analyzes the DNN’s computation graph. The Strategy Maker runs our strategy framework to generate optimized strategies for operation placement, tensor communication and execution schedule. Then, the Graph Compiler applies the strategies to produce the distributed training DAG and enforces execution orders with the execution engine.  图分析器,  分析DNN的计算图.    strategy maker调度 操作, 执行顺序. 然后 图编译器 产生图然后执行.

​	To facilitate strategy making with Agent and Scheduler, the profiler runs different models in the given environment to profile execution time of each operation and transfer time of tensors across different devices; the Simulator exploits profiled information to estimate per-iteration training time under different strategies, for Agent’s policy learning.

3.2 Graph Analyzer
Graph Analyzer analyzes the original computation DAG, i.e., obtains the `graphdef` of the DNN model, which is a low-level representation of the computation graph regardless of which API is used to build the DAG (e.g., Estimator, Keras, etc.), in case the TensorFlow framework is used.

3.3 Strategy Maker

 Our complete set of distributed training strategies includes the following. 

(i) Parallelism (DP or MP) and placement for each operation:
for DP, an operation is replicated into multiple replicas which are deployed onto multiple devices, with input data evenly divided among the replicas; with MP, the operation is not replicated and deployed onto one single device.

 (ii) Gradient communication methods (PS or AllReduce) for gradient aggregation operations. 

(iii) Execution order of operations based on placements.

​	The goal is to minimize per-iteration training time of the DNN, i.e., end-to-end execution time ofthe respective DAG. The complete problem is very hard in nature: even the subproblem ofdeciding the execution order ofoperations within a restricted solution space (i.e., not considering operation replication and communication methods) is already NP-hard, which can be reduced to the DAG task scheduling problem [32] or the job-shop problem [3]. Given the significant hardness of solving the complete problem using a combinatorial optimization approach, we design a novel strategy framework joining both combinatorial optimization and graph neural network (GNN)-based learning to tackle the large strategy space. 基于组合优化和GNN的学习

​	We divide the strategy set into two parts and tackle each using a
different methodology based on output from each other:

​	 Part-I includes decisions (i) and (ii) which modify the single-GPU computation DAG into a distributed graph; 

​	Part-II includes decisions (iii) for setting the execution order of
operations in the distributed training graph. 

​	We adopt a GNN to produce Part-I strategies; we design an efficient heuristic to solve the remaining Part-II problem (which is still NP hard though with a smaller decision space), given Part-I decisions; we compute DAG execution time based on all the decisions and use it as the reward for GNN policy learning. The rationale原理 behind is to pursue an optimization problem (for Part-II decisions) that is close to a known one, with efficient approximation algorithms in place, while using the GNN to produce decisions for harder components of the complete problem. 第一部分让GNN, 然后第二部分我们有heuristic. GNN的reward是DAG执行时间.

The Strategy Maker consists of the following components for
strategy making: 

**Agent.** The agent runs the GNN, using input feature vector created base on profiling data, and generates Part-I decisions. Details of the GNN design will be introduced in Sec. 4.1. 

**Scheduler.** The scheduler runs the heuristic (Sec. 4.2) to compute execution order of all operations, based on decisions made by the Agent.

Two auxiliary辅助 modules are used for building the Agent:
**Profiler.** It profiles the given DNN model to obtain execution time of each operation on different devices under different batch sizes, the size of the tensor transferred between operations, and the link bandwidth between each pair of devices. We run the given DNN model on each device with different representative batch sizes, if the model can be fit into the device memory. For a large model that cannot be fit into a GPU, we use model parallelism, and try different placements on multiple devices. These allow us to measure computation time of each operation on different devices with different input sizes, so that we can build a linear regression model to predict computation time of a specific operation at other batch sizes, according to the type of operation, the shape of its input, the device that runs the operation, and other attributes of the operation such as the dilation ofa Conv2D node. For models that can be fit into a single device’s memory, it takes less than 10 minutes to complete the profiling; for larger models that cannot be fit into a GPU, the profiling typically takes less than half an hour. We transfer data with different sizes between each pair of devices, record the transfer time and build a linear regression model for transfer time prediction over each link based on the size of tensor for transfer. 

**Simulator.** The simulator is used for training the GNN in the Agent. It simulates training according to the strategies produced by the Agent and the Scheduler, using profiled data from the Profiler. It estimates the per-iteration training time for setting rewards for GNN training, and also tracks memory usage on each device, to set bad rewards for strategies leading to memory overflow.

3.4 Graph Compiler

 The Graph Compiler receives strategies produced by the Strategy Maker and generates a distributed training model which can be directly run in the heterogeneous environment. 

Operation replication. For operations that use DP, Graph Compiler creates replicas of the operation and places them onto the devices (i.e., by setting the ‘device’ attribute of the node as in TensorFlow). The number of replicas placed on each device is decided by the Agent. 

Gradient Aggregation. When the PS architecture is chosen for parameter synchronization among replicas of an operation, one device (where a replica of the operation is deployed) performs as the PS as well (to reduce some gradient communication overhead), storing the parameters; gradients from other replicas are sent to the PS. The PS device is chosen as one that minimizes completion time of gradient aggregation. 让梯度聚合完成时间最短

When AllReduce is selected, gradients are synchronized among
all replicas of the operation using an Allreduce algorithm: ring-based AllReduce [36, 45], or a hierarchical AllReduce structure that aggregates gradients among GPUs on the same physical server first and then across servers. We always use the better structure among the two 两种reduce方法选一个by estimating the communication time of the two based on the given network topology. 

We adopt synchronous同步 SGD for DNN training in HeteroG: after gradient aggregation, updated parameters are applied to all replicas. Consequently, parameters are consistent among all model replicas, and the accuracy ofthe trainedDNNmodel is not affected regardless of the model transformation.
**Order Enforcement.** Each operation in the distributed training graph is assigned with a priority according to the execution order computed by the Scheduler, for the execution engine to schedule the operations accordingly.

3.5 Client API
HeteroG provides a simple programming interface `get_runner` for developers to call after they build the single-GPU graph. As shown in Fig. 5, `get_runner` accepts as arguments a single-GPU graph (generated by `model_func`), input dataset (`input_func`), device information (`device_info`) including IP addresses (or hostnames)

```python
import heterog 
def model_func():
	#create single GPU model 
    loss = ...
    train_op = ... 
    return train_op
def input_func():
    #create input dataset 
    dataset = ...
    return dataset
dist_runner = heterog.get_runner(
	model_func, 
    input_func, 
    device_info,
	heterog_config)
dist_runner.run(steps) # 开发人员指定steps
```

of machines and GPU IDs, and an optional HeteroG configuration object (heterog_config) containing extra arguments if needed (e.g., a file path to save trained variables, whether to use default execution order or our order scheduling algorithm). A developer can first define a single-GPU computation model (lines 3-7) and input dataset (lines 9-12), and then invoke get_runner (lines 1418). The API computes deployment strategies and produces the distributed training model; the returned dist_runner object contains the modified graph and its run function executes the modified training model according to the execution order, with a maximum number of training steps specified by the developer (line 20).

### 4 STRATEGY FRAMEWORK

 An illustration of our strategy framework is given in Fig. 6.
4.1 GNN-based Policy Learning

 We adopt a GNN for Part-I strategy making due to close relation of our decisions with the structure of the DNN graph: embeddings是一个向量 produced by a GNN encode features of the DAG and have been shown effective in facilitating graph-related decision making [2, 64]. We do not use a GNN to produce all strategies as execution order decisions are hard to be described as GNN output (because execution order decisions are for operations on distributed graph rather than original single-GPU graph) and the action space would be too large to learn.

4.1.1 Model Feature Encoding 

 Different DNN models have different numbers of operations. A GNN is used for creating a flat feature vector for each DNN model, by encoding the graph information into a set of embeddings.  Per-node embeddings. We employ a graph attention neural network (GAT) [53], which achieves better performance than GCN [11, 55, 57] when handling graph-based problems, by aggregating features among neighbors based on correlation coefficient between each pair of feature vectors and using multi-head mechanism to enhance aggregation performance. The GAT takes as input the DAG of DNN model, in the form of: (1) a node feature matrix, where each row contains the operation’s attributes (e.g., execution time when running on different devices, the input and output sizes, the average tensor transfer time between each pair of devices);1 (2) an adjacency matrix describing data dependencies. It generates a per-node embedding vector eo, by encoding attributes ofimmediate neighbors ofo using multi-head attention layers:

Here K is the number of heads of multi-head attention layer, ∥ denotes concatenation of the output of each head, σ is non-linear transformation, No is the set of neighbors of o including o itself, αoj is the correlation coefficient between feature vectors of node o and node j,W is the weight vector to be learned, and e output embedding of node j from the previous attention layer.

*Per-group embeddings.*多个结点聚集成组, 一组学习a set of策略 A DNN model typically contains thousands of operations. Making decisions for each of them results in a very large action space, and hence significant challenge in finding good strategies. We therefore further gather multiple nodes into groups, and learn a set of strategies for nodes in the same group, significantly reducing the action space. We design a nearest-neighbor method to decide the groups: If the number of operations exceeds the maximal group number N, we choose the top-N operations with longest average execution time (these operations contribute more to the per-iteration execution time). We group each of the other operations with one of the N operations with the least number of hops in-between (we want nearby operations to have similar strategies to reduce communication overhead and extra split/concat operations). A per-group embedding gi is computed by encoding information from all nodes in this group:

where Gn contains all the nodes in group n.

4.1.2 Strategy Network

 Embeddings of node groups are concatenated into a feature vector连接成特征向量, further fed into a strategy network for making Part-I decisions on operation replication/device placement and gradient aggregation method. We employ a Transformer-XL network [8], which has been shown excellent in handling long embeddings (e.g., for language translation). We encode Part-I decisions as output of the strategy network. An N × (M + 4)-dimensional action space is designed, where M is the number of GPUs. In the (M+ 4)-dimensional vector for each group, each of the first M elements represents placing operations in this group to the corresponding device using model parallelism (i.e., no replication on the other devices). The last 4 elements correspond to different data parallelism schemes: the four combinations between two replication decisions (replicating the group onto each of the M devices with one replica per device and proportionally placing a number of replicas of the group onto each device according to computation power) and two communication methods ( PS or AllReduce for gradient aggregation in the group). A softmax function is used to produce an action for each group, out of the M+4 strategies.

4.1.3 GNN Training

 The graph embedding GAT(图注意力网络graph attention networks) and strategy network are trained end-to-end together through reinforcement learning (RL) [27]. In each round, a set of DNN graphs, G, are sampled as input to the GAT. For each graph, deployment strategies are produced from the strategy network and a reward is computed by the simulator based on simulated training of the respective DNN using the deployment strategies and execution order (produced by the heurisitc algorithm in Sec. 4.2). The reward is the additive inverse of the square root of the per-iteration execution time of the DNN graph . if there is no out of memory (OOM) error; otherwise, we multiply the computed reward by 10, to lower the chance of producing the respective strategy.

The objective of RL is to maximize the overall reward over the |G| input graphs: J(θ) =  累加ED∼πθ (G)[RG,D]+λH(πθ), where θ is the set of weights in the GAT and strategy network to learn, and πθ is the policy distribution to produce actions. The regularization termH(πθ) [17] allows πθ to have a high entropy, i.e., high diversity in the decisions, for sufficient exploration of the action space. λ balances exploration and exploitation. With each reward, weights are updated by policy gradients [58]:

4.2 Execution Order Scheduling

Even though the computation operations are already partially ordered based on the data-flow dependency of DAG, there still exist situations that multiple operations placed on the same device are ready to run at the same time, and different orders to execute them may lead to different training time. The scheduler decides the global execution order of all operations (including concat and split) based on the modified training graph after applying the Part-I decisions.

​	Here, we further treat a link between two GPUs as a device. We regard parameter synchronization among a operation’s replicas as a communication operation, and deem that it is placed on a link if the respective PS or AllReduce-based parameter synchronization makes use of the link. Our order scheduling algorithm ensures that every GPU processes at most one computation operation at a time, and every link sends tensor for at most one communication operation at a time.
​	Our execution order scheduling to minimize per-iteration training time is a combinatorial optimization problem, similar to but simpler than classical task scheduling problems with inter-task dependencies [3] (as the placement of each operation is given). Nonetheless, our problem is still NP-hard, as it is a generalization of the job-shop problem [14]: the job-shop problem schedules tasks with chain-like precedence constraints given their machine placement, while our problem allows arbitrary precedence relations among operations. List scheduling algorithms are commonly used for solving dependency-based task scheduling problems approximately [32]. The core idea of list scheduling algorithms, e.g., HEFT [22], is to assign priorities to tasks, and then assign tasks to the best devices and schedule them on the respective devices in order of their priorities.

​	We adapt the idea for our execution scheduling. We compute a rank for each operation:

where pi is the computation or communication time of operation oi , and succ(oi ) is the set of all successors of oi . Given device placement of the operations, on each device, we order operation execution according to their ranks, and run an operation with a higher rank when it is ready (i.e., its dependencies have all been done), before moving on to the next operation. Multiple devices can execute their respective ready operations concurrently; since we consider inter-GPU links as devices, this maximally allows computation and communication overlap.

​	We can prove a (tight) performance bound of our order schedule heuristic. Let TLS andT∗ be the per-iteration execution time using our heuristic and the ideal optimal schedule, respectively. Recall M is the number of GPUs, and M^2 is the maximal number of links. Detailed proof is in the Appendix.
Theorem 1. TLS is no larger than (M +M^2)T∗. 

Theorem 2. There exists an instance of our execution order scheduling problem where TLS/T∗ ≈ M +M^2.

### 5 IMPLEMENTATION

HeteroG is implemented on TensorFlow 1.14 as a python module that developers can readily import into their TensorFlow code. Core design of HeteroG is generally applicable and can be implemented in other ML frameworks as well.

**Graph Analyzer** is built in Python with 480 LoC.
**Strategy Maker.** The Agent is implemented in Python with 2156 LoC. We use 12 multi-head attention layers in the GAT( GRAPH ATTENTION NETWORKS), with 8 heads in each layer. The maximum number of groups, N, is 2000. There are 8 layers in the Transformer-XL strategy network.

​	The Simulator and the Scheduler are built in Rust with 1862 LoC. The simulator simulates training process of the converted DAG. It maintains a ready queue for each device, consisting of operations assigned to the device in computed execution order, whose dependencies have been cleared. It keeps removing an available operation from the head of each ready queue, calculating completion time of the operation according to completion time of its dependencies and the device it is placed on, and adding its child nodes into the ready queue if their dependencies are all cleared. The simulator also simulates memory allocation and releasing when executing an operation (using reference counting), and records the peak memory usage on each of the device. The simulator records the link bandwidth utilization between each pair of devices. When more data are transferred using a specific link, the estimated communication time becomes longer accordingly.

