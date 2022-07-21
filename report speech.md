### report speech 

我科研项目的主题是利用hierarchical reinforcement learning to jointly allocate GPU resources and place jobs. 

首先我们来介绍一下我们的背景. 

如今，领先的IT公司运营ML集群，深度学习模型在这里接受培训，以提供各种AI驱动的服务。高效的资源调度对于最大限度地利用昂贵的DL集群至关重要。 现在很多调度器可决定如何将资源分配给作业，以最大限度地减少训练时间并最大限度地提高集群利用率.

例如, 我们分配给一个job三个worker和一个参数服务器来训练他的参数.... 介绍一下PS 架构.

 但是，同一服务器上的协同定位 ML 作业可能会相互产生负面干扰并体验性能不可预测性。 例如，当服务器上的 GPU 卡分配给不同的 ML 作业时，当作业在其分配的 CPU 和 GPU 之间混洗数据时，将共享 PCIe 总线。 在非统一内存访问架构中，当两个分配的 GPU 未连接到同一 CPU 时，QPI 总线是共享的。 我们无法最大化服务器上的资源碎片利用率。

目前, Tiresias 处理将 GPU 分配给单个作业（即作业放置）和随时间调度多个作业。 但它是非规模自适应调度程序。 它要求用户在提交作业时指定 GPU 数量，该数量将在作业的生命周期内固定。 另外没有考虑共同的干扰.

我们希望能联合,Co-adaptive Cluster Scheduler for DL.

### 陶瓷介绍中文

   最近的弹性调度器可以为每个作业自动选择适当数量的资源，但问题是工作线程和 PS 很可能分布在不同的物理服务器上. 然而，即使没有过度订阅资源，同一服务器上的协同定位 ML 作业也可能会相互干扰,性能不可预测。
   谷歌大脑使用RNN来预测运算在计算图中的位置。 它基于数据流而不是PS架构。 但是，规划器只能优化目标模型（例如，TensorFlow 图）的训练时间，如果遇到另一个模型，则需要重新训练此规划器。 此外，建模通常不考虑多类型工作集群中的干扰。 相反，我们希望找到不同类型工作的有效安置。 它更通用，更高效。
   Pollux 使用自适应协同优化相互依赖的因素来提高调度性能。 但它没有考虑同地干扰。 我们希望共同优化资源分配和设备放置。
   使用一个 NN 来产生资源分配和放置决策具有挑战性，主要是因为动作空间要大得多。 这就是我们要使用分层 RL 进行调度的原因。

完成了从 trace 中提取数据,拟合运行速度与worker和PS关系函数,

因为没有device placement的trace, 我手动estimate the influence of different placements. 

​	Even though recent elastic schedulers can automatically select an appropriate amount of resources for each job,  the problem is that the workers and PSs may well be distributed onto different physical servers. However, even without over-subscription of resources, co-located ML jobs on the same server may interfere with each other negatively.
​	Google brain uses a recurrent NN to predict the placement of operations in a computational graph. It is based on data flow instead of PS architecture. However, the planner can only optimize the training time for a target model (e.g., a TensorFlow graph) If we meet another model, we need to retrain this planner. Further, the modeling typically does not consider interference in a multiple-type job cluster. On the contrary, we want to find the efficient placement of different types of jobs. It is more generic and efficient. 
​	Pollux uses adaptively co-optimizing inter-dependent factors to improve scheduling performance. But it doesn't take co-located interference into account. We want to co-optimize resource allocation and device placements.

​	Using one NN to produce both resource allocation and placement decisions is challenging, mainly because of the significantly larger action space. That is why we want to use hierarchical RL to schedule.

   We could input the current information of the job and cluster, the first level allocate of resources, the second level of additional input cluster available resources and  the current placement. The final output of the allocated resources and placement. Then,  we calculate reward for the first and second levels.
   The figure is shown. Some of the inputs are from the cluster API and some are from the job owner. The input from the job owner can be fixed in the experiment. At first we need train two networks separately. It helps convergence.

​	Now I have read previous work and find potential limitations, have written the motivation and background of my paper.  

I design the hierarchical RL and make a theoretical comparison. 

I  implement our design and test it in experiment.

That's all. Thank you for listening!



