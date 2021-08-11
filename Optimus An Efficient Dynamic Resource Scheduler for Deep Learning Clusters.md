Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters

解决了什么问题

基于online resource performance model, minimize  job training time. 

怎么解决的?

1.建立精确性能模型, 可以track 训练精度,  可以online fitting 预测收敛需要的epochs. 

2.根据性能模型设计了动态分配资源的方法. 还提出了task placement scheme 方案

3.目前PS框架 负载不均衡, 提出 reduce 通信开销, 平均分配model slices给PS.  

C我可以怎么自己用这方法



D 哪些参考文献是值得follow 的

50 Device Placement Optimization with Reinforcement Learning

E存在的缺陷:

假设 job有smooth平滑的 loss曲线而且running to completion, 这两个假设在生产系统中不一定是true. 



## 摘要

面向深度学习cluster的scheduler 算法. optimus, 基于online 资源性能模型加速. 用online fitting,  Optimus 在训练过程中使用在线fitting拟合来预测模型收敛性，并建立性能模型以准确估计作为每个作业中分配资源的函数的训练速度. 基于这些模型，设计并使用了方法来动态分配资源和放置深度学习任务，以最大限度地减少作业完成时间。

## 1 introduction

first 谷歌的borg, 腾讯微软百度的yarn-like 都是固定资源, 比如晚上worklaod低有额外资源也不会去利用.

second , yarn mesos, borg 用于通用cluster 资源管理, corral 用于周期性数据并行job, tetrisched 处理基于reservation-based , 不是为ML设计的.  

optimus 应用场景:PS框架，　并行DL jobs.

Optimus为每个正在运行的作业建立资源性能模型，并基于作业进度和集群负载动态地调度资源到作业，以最小化平均作业完成时间和最大完成时间makespan。

贡献

1. 建立精确性能模型, 可以track 训练精度,  可以online fitting 预测收敛需要的epochs. 通过利用通信patterns 和训练过程的iterativeness , 我们进一步建立了资源性能模型.  和之前不同, 我们不需要了解ML的内部结构和 cluster的硬件配置, 我们方法是用不同资源配置运行一个job, 收集训练速度data, 然后调整模型. 
2.  动态分配资源和task placement.  根据性能模型设计了动态分配资源的方法, 加速job. 还提出了task placement scheme 方案, 减少了通信开销,加速训练.
3.  均衡负载.   目前MXNet 这些PS框架 负载不均衡, 我们提出 reduce 通信开销, 平均分配model slices给PS.  

## 2 background and motivation

### 2.1 DLmodel training

与实验模型不同，生产模型是成熟的，通常可以很好地收敛到全局/局部最优，因为所有超参数（例如，学习率，小批量大小）都在实验阶段得到了很好的调整。在这项工作中，我们专注于这样的生产模型，并利用它们的收敛性来估计培训工作的收敛进度。

特别地，我们利用训练损失的收敛性来决定DL任务的完成。如果连续两个时间段之间的loss改变在几个时间段内一直低于job owner指定的阈值，则DL模型收敛。基于训练损失的训练收敛在实践中很常见[48，71]，训练损失的收敛通常意味着生产模型的其他指标（如精度）的收敛（即无过度拟合）[5]。在一些没有“正确答案”的场景中，很难定义训练/验证的准确性，例如语言建模[6]。validation loss通常用于防止模型过拟合，只有在必要时（例如，在每个历元结束时）才对验证数据集进行evaluate评估，而我们可以在每一步之后获得训练损失，以获得更精确的曲线拟合（§3.1）。

### 2.2 The Parameter Server Architecture

### 2.3 Existing Cluster Schedulers

和harmony 的介绍也差不多

 **Static resource allocation.**  静态制定了就不改了. 给一个job增加资源不一定会变快,可能还会变慢, 训练速度还受可用带宽影响.   改进 :optimus 调整了资源的数量和位置

工作规模不敏感:   spark 用fifo  , mesos和yarn 用 DRF或变体. 这些策略不知道job size.  短作业可能 starvation  改进 :optimus 考虑了job的大小

## 3 PERFORMANCE MODELING OF DL JOBS 

怎么获得资源配置和 收敛时间关系?   在线估计job的epoch数, 和 一个epoch需要的时间.

### 3.1 学习收敛曲线

我们根据每个DL工作的训练进度绘制训练损失曲线，并进行在线模型拟合，以预测模型距离收敛的距离。

 **数据预处理。** 为了更好的模型拟合，我们执行如下 outlier异常值去除：如果损失数据点不在其邻居的某个范围内（例如，在随后的 5 个时期的最小损失和前 5 个时期的最大损失之间），我们 将数据点视为异常值，在进行模型拟合时使用其邻居的平均值来替换该点。 我们还将损失值归一化，方法是将每个原始值除以目前收集的最大损失值（通常是第一个损失值）。 这样，不同 DL 作业中的损失值都在 0 和 1 之间。

**online fitting**   用一个模型fitting loss 曲线, 每一step 收集一个数据点, 预处理然后non-negative least quares 最小二乘法 solver.  如果要几十万个step 那就可以隔几步然后采样, 

### 3.2 资源速度建模

我们接下来基于参数服务器架构中的计算和通信模式构建resource-speed模型。

 **系统模型** 在一个典型的 DL 工作中，在一个 worker 上完成一个训练步骤所花费的时间包括在 worker 上进行前向传播（即损失计算）和反向传播（即梯度计算）的时间，worker 将梯度推送到参数服务器 ，参数服务器更新参数，worker 从参数服务器拉取更新的参数，加上额外的通信开销。  带宽瓶颈通常在PS端. 

一个worker 上一个step持续时间可以表示出来.

根据等式直到 , 为了每一个step 时间最短,  worker 的处理速度应该差不多, PS应该负载平衡.

然后可以导出一个job的训练速度. 也就是单位时间内完成的训练step数. 

异步的话:  

同步训练 : 每一step 所有worker 的minibatch 总大小应该不变, 这样改变worker数量 训练结果也不变.

**model fitting** 先小样本处理几十秒, 构造初始speed函数,  然后开始job, 用初始函数资源调度, 收集数据点, 校准系数.

## 4动态调度

In our DL cluster, jobs arrive in an online manner. Optimus periodically allocates resources to the active jobs (new jobs submitted in the previous scheduling interval and unfinished jobs submitted earlier), by adjusting the numbers and placement of parameter servers/workers in each job in the shared DL cluster. Its scheduling algorithm consists of two parts: resource allocation and task placement.  作业以 online 到达,  定期分配资源. 

### 4.1  分配资源

 解决一个优化问题,  optimization problem. 但是他非线性甚至非凸函数, NP hard,  用启发式算法来求解. 

先给每个job分配一个PS, 然后根据 边际效益排序,  选择边际效益最大的来添加资源.  直到资源用完或者边际收益<=0

为了mitigate减轻预测错误导致的性能下降degration,  边际效益 ×一个系数, job一开始优先级低一些, 减轻一开始训练预测误差大的影响. 

### 4.2task 放置

为了reduce 交换参数的开销. 

证明了最佳放置原则:   用最少的 server to host the job, 每个server上deploy 相同数量的 PS和worker. 

证明背后的原则是：（a）将工作人员和参数服务器合并可以减少跨服务器的数据传输，（b）在每个服务器上打包相同数量的工作人员/参数服务器可以使同步训练的每个步骤中的最大数据传输时间最小化。我们也可以将这些原则应用于异步培训工作，以平衡多个工人的培训速度。

基于这些原则，我们设计了一个布局方案，以最小化训练期间的数据传输时间，如下所示。我们将集群中的所有服务器按其当前资源可用性的降序排列（在我们的实验中使用了可用的CPU容量）。我们将工作按其资源需求的递增顺序排列（即，最小的工作优先），以避免工作匮乏（即，小工作得不到任何资源）。对于每个作业，我们检查前k个服务器上的资源是否足以承载作业（从k=1开始）。如果是这样，我们将参数服务器和作业中的工人均匀地放置在k个服务器上；否则，我们将检查第一个k+1、k+2、··服务器，直到找到足够的服务器来放置作业。然后，我们更新k服务器上的可用资源，并再次对服务器列表进行排序。重复上述过程，直到放置了所有作业或服务器上没有足够的资源来承载更多作业。请注意，服务器可以容纳的作业数可能小于我们通过资源分配算法（考虑整个集群中的总体资源容量）分配资源的作业数。未放置的作业将在下一个调度间隔中临时暂停和重新调度。

## 5 SYSTEM IMPLEMENTATION

 We next present some implementation details of Optimus.

### 5.1 Data Serving 

存在HDFS  默认块大小128MB, 复制因子为2.一开始给每个worker 差不多chunk, worker 数量变了, 就重新分配chunk 让负载平衡. 

### 5.2 Straggler Handling

如何处理slow worker?

异步 可以监视训练速度 , 太慢了, 比如中位数的一半速度.

同步  在PS上监视梯度到达时间, 训练速度为 两个到达时间的间隔. 

launching 一个新worker replace a straggler.

### 5.3 Load Balancing on Parameter Servers 

怎么切分参数块? 如果小于阈值就随机分给一个PS, 否则切分.  不同模型有不同阈值.  阈值设计很难.   tf也有这种问题. 

最小化三点,  1  两个PS参数大小的最大差异 ; 2 在一个训练步骤中，参数服务器和工人之间的参数更新请求总数（工人的每个请求请求一个更新的参数块   ;3.两个参数服务器之间的参数更新请求数的最大差值。我们设计了一个参数分配算法（PAA），如下所示。

| 块大小            | PS选哪个?        |
| ----------------- | ---------------- |
| <平均参数大小的1% | 最少更新请求数的 |
| 1% - 100%         | 剩余容量最小的   |
| >平均size         | 分割它           |

一旦一个参数块（或分区）被分配给一个参数服务器，我们就把服务器上的参数更新请求数加1。

### 5.4 elastic  弹性训练

分配资源改变时怎么动态分配?  检查模型参数并保存到HDFS, 然后从检查点 restart job, redeploy PS. 

### 5.5 k8s

把optimus 作为一个normal pod, 也就是 a unit of deployment 和一个或多个容器紧密couple. 它轮询k8s master 主机 来获得cluster information and job states. 容错用 etcd 一种分布式reliable kv storage 作为 job state的容错存储器,  k8s will automatically restart the scheduler if it fails.

## 6 evaluation

### 6.1 Methodology

1. Hadoop[11]、Yarn[61]和Mesos[40]等许多资源管理器中采用的一种基于公平性的调度器，它使用支配资源公平性（DRF）[34]将资源分配给作业并动态地重新调度

2. tetris[37]，它优先将资源分配给持续时间较短或资源消耗较小的作业，并将作业打包到服务器以最小化资源碎片。

 metrics指标:我们使用平均作业完成时间（JCT）作为系统性能的指标。此外，我们评估makespan作为资源效率的一个指标，它是从第一个作业到达所有作业完成所经过的总时间。最小化makespan相当于最大化资源效率

### 6.2 Performance

 和DRF比,  optimus 不会运行大量task,  因为DRF 是 work-conserving and 给job分配尽可能多的资源, 但是资源多不一定快.  optimus 有更高的CPU利用率. 

**Resource adjustment overhead** 
**Scalability.**  模拟了在几千个结点的集群中,  submit 和调度大量job .  性能和默认调度器差不多 , 5秒 1万六结点 调度4000个task,  十万个job.  10分钟调度一次, 调度开销非常小. 

### 6.3 Sensitivity analysis 

 预测误差的影响,  训练模式的影响, job到达过程的影响. 

**Prediction error.** 研究了收敛时间和训练速度的预测误差对最优解的影响程度, 进行了仿真.  速度估计误差对性能影响更大,

怎么研究训练模式对性能影响? 异步训练所有, 同步训练所有.  同步时性能增益更大,这是因为所有的工人都有最新的参数同步训练，这样模型收敛更稳定，收敛估计误差更小，所有工人在同步培训中的培训速度相同，速度估计误差较小。

还研究了两种job arrival processes. 一个 poisson 泊松过程,  一个interval 3个arrivals . 第二个是 从7小时内 google 集群中提取的traces.

### 6.4  检查设计细节

**Resource allocation.** 控制变量法To see how effective our marginal gainbased resource allocation algorithm is, we replace it with the **resource allocation** schemes in the fairness scheduler or Tetris, while still adopting the same **task placement** algorithm in Optimus. 

 资源分配非常重要. 60%,  placement 10%-15% , 效果还可以. 

Parameter server load balancing. 参数服务器之间的参数size差异、参数服务器之间的参数更新请求数差异以及参数服务器和工作服务器之间的更新请求数是表示参数服务器上的负载不平衡或开销的三个主要因素。

summary:

（1） 实验表明，与公平调度相比，Optimus更快。此外，Optimus可以在5秒内扩展到在16000个节点上调度100000个task，并且其资源调整开销很小，即2.54%。
（2） 进一步提高估计精度不会使Optimus的性能提高太多（15%），并且Optimus在各种工作负载下的性能都优于DRF和Tetris。
（3） 资源分配算法、任务分配方案和参数服务器负载均衡算法分别使Optimus的性能提高了62%、17%和20%。

## 7 DISCUSSIONS 

We now discuss extensions and future work on Optimus.

1. 不同的负载k 8s 可以多种调度器,  每个调度器负责一种workerload, 
2. 收敛估计, 
3. 减少检查点开销. 设置检查点的阈值来限制重新启动的频率。对于长作业或大作业，阈值可以更小，以避免频繁的资源重新分配。

## 8 RELATEDWORK

**Performance modeling.** Jockey [32] and Morpheus [44] use 周期性job的trace 动态调整, while Optimus does not depend on the previous run of the same job since production training data often change (e.g., daily). PerfOrator [53] builds a resource-to-performance model of big data queries by 估计查询大小和分析硬件estimating query size and profiling hardware, while we use high-level system modeling approach不用了解硬件或job内部细节. 

|                      | 其他方案                                                     | optimus                                                      |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| performance modeling | 利用job 的trace ,                                            | 不依靠之前的trace,                                           |
|                      | 估计查询大小和分析硬件                                       | 高层建模不了解硬件或job内部细节.建模拟合了基于sample runs的参数化性能模型 |
|                      | 非常细粒度建模                                               | 捕获高级计算和通信模式                                       |
| job scheduling       |                                                              | 侧重深度学习workload                                         |
|                      |                                                              |                                                              |
|                      | eagle 把资源划分为两个区分别用于长job和短job                 | 重点放在job的动态资源配置上, 运行在PS架构上的ML job          |
|                      |                                                              | 利用job的特性设计了资源分配和任务分配算法,                   |
|                      |                                                              | 没有修改底层ML 框架                                          |
|                      | azalia 用model-free  DRL 实现模型并行, 对于资源分配不是通用和有效的 | 我们发现负载不平衡问题在这些分布式框架中很常见，我们在其中一个框架MXNet中提出并实现了PAA算法。 |

 [后面的工作](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9355203)    认为Optimus最大限度地减少了在 PS 架构中运行的多个作业的完工时间和平均作业完成时间。 它设计了一种新颖的资源性能模型，并提出了一种调度方案来动态调整分配的工作人员和服务器的数量。 

缺点:  However, the adjustment does not support “reshaping down” and is implemented based on job termination, which results in significant overhead

## 9 conclusion

Optimus是一个定制的集群调度器，目标是在深度学习集群中实现高训练性能和资源效率。它的核心是一个精确的深度学习工作负载性能模型，该模型是通过利用数据模型训练的特性（如收敛性、迭代性）和参数服务器体系结构的通信模式构建的。**建模了训练速度的模型,   构建了  speed和资源的关系.**  在性能模型的基础上，设计了基于边际增益的资源分配算法和训练速度最大化的任务分配方案。在Kubernetes集群上的实验表明，Optimus的性能明显优于典型的集群调度算法。
