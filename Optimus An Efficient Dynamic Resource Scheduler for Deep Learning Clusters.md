Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters

解决了什么问题

基于online resource performance model, minimize  job training time. 

 深度学习训练工作是资源密集型且耗时的。 高效的资源调度是深度学习集群性能最大化的关键。 现有的集群调度器很大程度上不是为深度学习job量身定制的，并且通常为每个作业指定固定数量的资源，从而阻碍了高资源效率和作业性能。 本文提出了 Optimus，一种用于深度学习集群的定制作业调度器，它基于在线资源性能模型最大限度地减少作业训练时间。  Optimus 在训练过程中使用在线fitting拟合来预测模型收敛性，并建立性能模型以准确估计作为每个作业中分配资源的函数的训练速度. 基于这些模型，设计并使用了一种简单而有效的方法来动态分配资源和放置深度学习任务，以最大限度地减少作业完成时间。 我们在 Kubernetes 之上实施 Optimus，这是一个用于容器编排的集群管理器，并在具有 7 个 CPU 服务器和 6 个 GPU 服务器的深度学习集群上进行实验，使用 MXNet 框架运行 9 个训练作业。 结果表明，Optimus 在作业完成时间和完工时间方面分别优于代表性集群调度程序约 139% 和 63%。

怎么解决?



3.1 学习收敛曲线

我们根据每个DL工作的训练进度绘制训练损失曲线，并进行在线模型拟合，以预测模型距离收敛的距离。 数据预处理。 为了更好的模型拟合，我们执行如下异常值去除：如果损失数据点不在其邻居的某个范围内（例如，在随后的 5 个时期的最小损失和前 5 个时期的最大损失之间），我们 将数据点视为异常值，在进行模型拟合时使用其邻居的平均值来替换该点。 我们还将损失值归一化，方法是将每个原始值除以目前收集的最大损失值（通常是第一个损失值）。 这样，不同 DL 作业中的损失值都在 0 和 1 之间。图 5 显示了通过运行表 1 中的示例 DL 作业收集的示例损失曲线（它们是来自官方 MXNet 教程 [15, 16] 的 DL 示例），使用 MXNet 框架在带有 1 个 E5-1650 v4 CPU 和 2 个 NVIDIA TITAN X GPU 的服务器上。 每个作业的学习率设置为固定。 训练进度是模型已训练的 epoch 数与收敛所需的 epoch 总数之比。

3.2 资源速度建模

我们接下来基于参数服务器架构中的计算和通信模式构建资源到速度模型。 系统模型。 在一个典型的 DL 工作中，在一个 worker 上完成一个训练步骤所花费的时间包括在 worker 上进行前向传播（即损失计算）和反向传播（即梯度计算）的时间，worker 将梯度推送到参数服务器 ，参数服务器更新参数，worker 从参数服务器拉取更新的参数，加上额外的通信开销。 假设作业中有 p 个参数服务器和 w 个工人。 每个参数服务器的带宽容量为B，模型大小（即参数的总字节数）为S。 一个worker训练一个minibatch时的前向传播时间ism·Tforward（一个mini-batch的大小乘以平均处理时间 一个例子）。 反向传播时间 Tback 与 tm 无关，通常是固定的。 梯度的大小

### 4动态调度

In our DL cluster, jobs arrive in an online manner. Optimus periodically allocates resources to the active jobs (new jobs submitted in the previous scheduling interval and unfinished jobs submitted earlier), by adjusting the numbers and placement of parameter servers/workers in each job in the shared DL cluster. Its scheduling algorithm consists of two parts: resource allocation and task placement.  作业以 online 到达,  定期分配资源. 

4.1 

在每个调度间隔中，让 Qj 表示作业 j 需要运行以实现模型收敛的剩余步数/时期数（§3.1），而 f(pj ,wj ) 是作业 j 的当前训练速度函数（§3.2  ）。 我们可以将作业 j 的剩余运行时间 tj 估计为 f(pj,wj ) 。 让 Or Qj j (Nr j ) 表示作业 j 中每个工人（参数服务器）占用的类型 r 资源的数量。  Cr 是 DL 集群中 type-r 资源的总容量，R 是资源类型的数量。  J 是当前活动作业的集合。 我们的调度程序旨在最小化这些作业的平均完成时间。 我们可以解决以下优化问题来决定每个作业 j ∈ J 的工人/参数服务器的数量，其中（7）是容量约束：

#### 4.2task 放置

### 5 SYSTEM IMPLEMENTATION

 We next present some implementation details of Optimus.

#### 5.1 Data Serving 

We store training data in Hadoop Distributed File System (HDFS) [3] with a default chunk size of 128MB and a replication factor of 2. At the beginning ofa job, we assign a roughly equal number ofchunks to each worker in a round-robin manner, so that each worker has a similar workload. When the number ofworkers changes due to our dynamic scaling, we reassign the data chunks so that the workload on each worker is still balanced.

#### 5.2 Straggler Handling

Stragglers, i.e., slow workers (we will discuss the case of slow parameter servers in §5.3), influences a synchronous training job significantly, due to the need of synchronizing all workers in each training step. For asynchronous training, it is also important to

#### 5.3 Load Balancing on Parameter Servers 

Our DL jobs are running on the MXNet framework. We identify possible significant load imbalance among parameter servers in MXNet, due to its way of dividing model parameters among parameter servers: for each block of parameters (i.e., parameters of one layer in an NN), if its size (i.e., the number of parameters) is smaller than a threshold (106 by default), then it is assigned to one parameter server randomly; otherwise it is sliced evenly among all parameter servers. Setting the threshold is difficult since different models may have different appropriate thresholds, and different threshold values often lead to a big difference in computation workload among parameter servers. Such a load imbalance problem also exists in other distributed ML frameworks such as TensorFlow.

