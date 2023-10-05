A Deep Learning-driven Scheduler for Deep Learning Cluster  2019 sept13

为什么分配worker多了不行?

A 尝试解决什么问题

问题是

1. ML负载特征不可知. 
2.  要基于对特定的ML 框架理解 来use 调度启发式函数.不高效不通用.

用DL更高效通用,动态分配资源, 离线预热. 

B方法有哪些关键元素

1. 用黑盒 without explicitly modeling ML framework, workload, interference.
2.  很多人用来 DRL , 有的静态资源分配 ,    有的根据任务执行顺利来调度基于图的并行job. 但是他们没有研究 job执行同时调整资源. 
3. Mirh他们也是优化计算图的placement ,   Xu他们用DRL选择网络结点的路由,  mao 用DRL动态决定视频流速率. 但是他们没有用在线RL来改进NN ,都是离线用自己建模的分析模型或模拟器生成的数据来训练. 
4. SLAQ 用在线fitting 调度经典ML , 但是没有人用 使用PS架构的分布式ML job.  optimus提出了 基于在线fitting 资源性能模型的动态资源scheduler.   但是这些都依赖详细建模和简化假设.   gandiva 把job 迁移到更合适的gpu , gpu共享的资源分配也是一个探索方向.  

C我可以怎么自己用这方法

D 哪些参考文献是值得follow的

### 摘要

   有效的资源调度对于最大限度地利用昂贵的 DL 集群至关重要。 现有的集群调度器要么对 ML 工作负载特征不可知，要么使用基于操作员对特定 ML 框架和工作负载的理解的调度启发式，这些方法效率较低或不够通用。 在本文中，我们展示了可以采用 DL 技术来设计通用且高效的调度程序。

​	 DL2 是 DL 集群的 DL 驱动调度程序，通过动态调整分配allocate给job的资源大小来针对global 训练作业 expedition。 DL2 提倡联合监督学习(应该就是线上和线下)和强化学习方法：基于现有集群调度器产生的作业trace，通过离线监督学习对NN进行预热； 然后将NN插入(代码上是怎么plug 的?  )实时 DL 集群，通过在 DL 作业的整个训练过程中进行的强化学习进行微调(我们的改进就是能不能placement 也用强化学习)，并用于以在线方式决定作业资源分配。 通过在准备监督学习阶段应用现有集群调度器做出的过去决策，我们的方法可以实现从现有调度器的平滑过渡，并在最小化平均训练完成时间方面提供高质量的调度器。 我们在 Kubernetes 上实施 DL2，并在 MXNet 上的 DL 作业中启用动态资源扩展。 广泛的评估表明，DL2 在平均作业完成时间方面优于公平调度程序（即 DRF）44.1% 和专家启发式调度程序（即 Optimus）17.5%

## 1 介绍

分布式 ML 框架（例如 TensorFlow [15]、MXNet [20]、Petuum [65] 和 PaddlePaddle [11]） 训练 DL 模型非常耗费时间和资源。 有效的资源调度对于运行具有多个训练作业的共享 DL 集群至关重要，以便最好地利用昂贵的资源并加快完成训练。 当今的 ML 集群中存在两大阵营调度器。在第一个阵营中，通用云/集群调度器被应用，并且可能被定制，用于分布式 ML 作业调度。 例如，Google 使用 Borg [59] 作为其 DL 集群调度器； 微软、腾讯和百度使用自定义版本的类似 YARN 的调度程序 [58] 来管理 DL 作业。 使用的代表性调度策略包括先进先出 (FIFO) 和显性资源公平 (DRF) [24]。 根据用户规范分配资源，并且不会在训练期间调整资源分配。 正如我们将在 §2.2 中看到的，为作业设置合适的资源量是困难的，静态资源分配会导致集群中的资源利用不足。

​	另一种是白盒,  第一步workload建模, 第二步  提出调度启发式函数进行 resource allocation.  缺点是三点: 

1. 需要深入了解 ML 框架和workload
2.  模型和框架coupled
3.  通常不考虑多租户集群的干扰 (怎么干扰的?  This is because the jobs share underlying resources such as CPU caches, disk I/O, network I/O and buses. Some ML jobs are CPU intensive )

​	 In the second camp, recent studies have proposed white-box heuristics for resource allocation in ML clusters. Typically they tackle the problem in two steps: set up analytical models for DL/ML workloads, and propose scheduling heuristics accordingly for online resource allocation and adjustment. Designing heuristics requires a deep understanding of ML frameworks and workloads, and the analytical model is tightly coupled with the ML framework implementation (e.g., a new feature or optimization in evolving ML frameworks may invalidate the analytical model) [49]. Further, the modeling typically does not consider interference in a multi-tenant cluster (§2.2)

这篇文章不是上面两种, 是黑盒调度.  分配.

​	In this paper, we pursue a DL cluster scheduler that
does not depend on expert heuristics and explicit modeling, resorting to a black-box end-to-end approach enabled by modern learning techniques. We propose DL2, a deep learning-driven scheduler for deep learning clusters, that dynamically adjusts resource allocation to training jobs on the go. DL2 learns resource allocation policies through experience using deep reinforcement learning (DRL): the <u>policy neural network</u> takes the current system state as input, produces resource allocation decisions for all the current training jobs and gradually improves the decisions based on feedback. However, merely applying off-the-shelf现成的 RL algorithms to scheduling does not produce high-quality decisions, and careful design according to the problem nature is in need.

策略NN: 

输入:  当前系统的状态, 这个状态有哪些? 

输出: 所有训练job的resource allocation 

在本文中，我们寻求一种不依赖于专家启发式和显式建模的 DL 集群调度程序，采用现代学习技术支持的黑盒端到端方法。 我们提出了 DL，一种用于深度学习集群的深度学习驱动的调度程序，它动态调整资源分配以训练

​	Existing DRL applications in resource scheduling scenarios  (§8) use simulators to generate training data for offline training, and apply trained models for resource scheduling in a live system. The core of such a simulator is typically an explicit performance model as mentioned above, and hence the inaccuracy of the simulator may lead to low-quality trained model. Instead of extensive offline training over large simulation, DL2 takes a different approach: we bootstrap the model using minimal offline supervised learning with any available historical job traces and decisions of any existing scheduling strategy employed in the cluster; then we use online training with feedback from ongoing decision making in a live system, with carefully designed techniques to guide model convergence to high-quality decisions, which minimize average job completion time in the cluster.

​	现有资源调度场景中的现有 DRL 应用程序, 用模拟器生成用于离线训练的训练数据，并在实时系统中应用训练模型进行资源调度。 此类模拟器的核心通常是如上所述的显式性能模型，因此模拟器的不准确性可能会导致低质量的训练模型。 DL2 没有对大型模拟进行广泛的离线训练，而是采用了一种不同的方法：我们使用最小的离线监督学习引导模型，其中包含任何可用的历史作业trace和集群中采用的任何现有调度策略的决策； 然后我们使用在线训练和实时系统中正在进行的决策制定的反馈，使用精心设计的技术来引导模型收敛到高质量的决策，从而最大限度地减少集群中的平均工作完成时间。

​	In summary, we make the following contributions in DL2:  

​	In contrast to previous DL cluster scheduling approaches that require analytical performance model and job profiling, DL2 adopts a more generic design, i.e., using DRL to schedule DL workloads. Instead of simulation-driven RL model training, we adopt online training with real feedback from online resource allocation (§2).我们在 DL 中做出了以下贡献： 与之前需要分析性能模型和作业分析的 DL 集群调度方法相比，DL 采用了更通用的设计，即使用 DRL 来调度 DL 工作负载。 我们采用online 资源分配的真实反馈的在线培训，而不是模拟驱动的 RL 模型培训（第 2 节）。

​	我们发现直接将简单的 RL 方法应用于我们的在线调度程序培训通常会导致错误的决策。 为了避免在线 RL 开始时的错误决策，我们在准备离线监督学习阶段应用 DL 集群中现有调度程序所做的过去决策。 我们的方法可以实现从现有调度程序的平滑过渡，并自动学习超越现有调度程序性能水平的更好调度程序（§3）。(这些是一些具体的方法,  不算key idea. 可以记下这个论文的主干, 然后再来论文中找他的具体实现. ) 为了优化在线 RL，特别是针对 DL 作业调度，我们提出了作业感知探索以在动作空间中进行有效探索，并采用额外的训练技术（例如，演员评论算法、经验回放）进行样本高效学习（第 4 节）。我们在流行的分布式 ML 框架 MXNet [20] 中设计和实现弹性扩展，以实现动态worker/PS调整（§5) 

## 2背景

### 2.1 PS框架

​	我们专注于PS  架构 ，该架构在分布式 ML 学习框架中广泛用于并行训练，例如 MXNet [20]、TensorFlow [15]、PaddlePaddle [11] 和 Angel [32]。 请注意，DL2 也可以扩展到 all-reduce，如第 7 节所述。 在 PS 架构中，模型例如深度NN (DNN)，在多个PS (PS) 之间进行分区，训练数据集在worker之间进行分割（在数据并行训练模型中）。 每个worker的数据分区又被分成minibatch； 每个worker在本地处理一个小批量并计算模型参数变化，通常表示为梯度。 梯度被推送到保持全局模型参数的 PS。 

我们专注于同步训练，其中worker的训练进度是同步的，并且 PS 在每次迭代中收到所有worker的梯度后更新全局模型。 更新的参数被发送回worker。 worker通过使用更新后的参数处理下一个小批量来开始下一个训练迭代/步骤。 在整个数据集中的所有小批量处理完一次后，就完成了一个 training epoch。 输入数据集通常训练多个epoch，直到模型收敛。

### 2.2motivation

1. 设备多的时候,  训练时间不是线性增长
2. 不同模型 设备数是不同的,不能默认配置.
3. 静态分配导致 资源空闲, 需要动态调整.
4. 白盒和ML 框架couple, 改了就要remodeling,另外也考虑多租户干扰很复杂.

   The typical workflow for a user to train a model in a DL cluster is as follows: The user specifies how many PSs and workers she/he wishes to use and the  each PS/worker needs, and then submits the job to the scheduler (e.g., Borg [59], YARN [58], Mesos [31]). The scheduler allocates PSs and workers to the job according to both user demand and its scheduling strategy, and the allocated resources then remain fixed over the entire training course of the job. This workflow has two limitations, as illustrated below. 

**Difficulty in setting the right worker/PS numbers.**  有更多的PS和work的时候, 训练速度如何提高?  How does a job’s training speed improve when more PSs and workers are added to the job? We train 3 classical models, i.e., ResNet-50 [30], VGG-16 [53] and Seq2Seq [23], in our testbed of 6 machines (see §6 for hardware details), and measure their training speeds (in terms of the number of samples trained per unit time), when increasing the number of workers and keeping the number of PSs equal to the worker number. Each worker uses 1 GPU, 4 CPU cores, 10GB memory and each PS has 4 CPU cores, 10GB memory. In Fig. 1, the speedup is calculated by dividing the training speed achieved using multiple workers/PSs (they are deployed onto physical machines in a load-balanced fashion) by the training speed obtained using one worker and one PS colocated on a single machine. We observe a trend of decreasing return, i.e., adding PSs/workers does not improve the training speed linearly. This is because more communication overhead is incurred when there are more PSs or workers. 因为有更多的通信开销.

​	On the other hand, is an equal number of PSs and workers (as a general rule of thumb) always the best? We fix the total number of PSs and workers to be 12 and measure the training speed of two models under different combinations of PS/worker numbers (i.e., 4:8, 6:6, 8:4) [49]. Fig. 2 shows that Seq2Seq achieves highest training speed when there are 4 PSs and 8 workers, while VGG-16 is trained fastest with 6 PSs and 6 workers.

​	From the above, we see that it is challenging to reason about which job will have the largest marginal gain from extra resources and what the best PS-to-worker ratio is, as they are affected by many factors, e.g., allocated resources, models. Existing schedulers largely side-step this problem and leave it to the user to decide how many PSs/workers to use.

静态资源分配。  GPU 集群资源通常没有被充分利用：当一个训练作业完成时，它释放的资源（例如，昂贵的 GPU）可能会空闲，而不是被仍在运行的剩余作业利用.  我们看到 GPU 利用率水平随时间变化，当集群负载低/高时，为训练作业中的动态资源扩展/缩减提供了机会。 我们提倡在训练作业中随着时间的推移动态调整 worker/PS 数量，以最大限度地利用 DL 集群中的可用资源来加快作业完成。 有了这个，我们进一步不要求用户提交 他们的worker和PS数量,

the number of workers/PSs they want to use in their jobs (who nonetheless may not be at the best position to decide that), but will decide the best worker/PS numbers for each user at each time based on both global resource availability and individual jobs’ performance.

​	**White-box heuristics.** There have been existing studies which explicitly model detailed relationship between the training speed and resources within jobs, and design scheduling heuristics based on the resource-speed model, e.g., SLAQ [67], Optimus [49] and OASiS [18]. They have two limitations. First, in order to derive an accurate performance model, the modeling process is coupled tightly with ML framework implementation, and remodeling is often needed when the framework changes (e.g., adding new features or adopting optimization). For example, Optimus models computation and communication as two separate procedures during one training step; its model needs to be rebuilt when new features are incorporated into ML frameworks, e.g., overlapping backward computation with communication, gradient compression [20].其次，**在不考虑多租户 GPU 集群中的干扰的情况下构建显式性能模型。** 例如，SLAQ和 Optimus [49] 假设 PS 上没有网络拥塞，而 OASiS [18] 和 Optimus [49] 假设可用带宽是一个常数。 但是，我们观察到训练相同模型的速度可能会有很大差异。 Second, explicit performance models are built without considering interference in multi-tenant GPU clusters. 白盒启发式。 已有研究明确地对作业内的训练速度和资源之间的详细关系进行建模，并基于资源速度模型设计调度启发式. 它们有两个限制。 首先，**为了推导出准确的性能模型，建模过程与 ML 框架实现紧密耦合，当框架发生变化（例如，添加新功能或采用优化）时，通常需要重构。** 例如，Optimus 在一个训练步骤中将计算和通信建模为两个独立的过程； 当新功能被纳入 ML 框架时，它的模型需要重建，例如，将反向计算与通信重叠，梯度压缩 [20]。  此外，对 ML 作业之间的干扰进行显式建模也非常困难 [17]，因为每个额外的维度（NN结构、并行架构、运行时隔离等）都会增加复杂性。

In contrast to white-box model-based schedulers, we
resort to a black-box approach and design an RL-based resource scheduler: it automatically learns end-to-end resource allocation policy without requiring expert heuristics and without explicitly modeling the ML framework, the workload, and the interference.与基于白盒模型的调度器相比，我们采用黑盒方法并设计了基于 RL 的资源调度器：它自动学习端到端的资源分配策略，无需专家启发式方法，也无需对 ML 框架,工作量和干扰进行显式建模 、

### 2.3 Deep Reinforcement Learning

​	观察状态,  根据策略选择动作,  move到新的状态, 显示reward, 根据reward 更新策略.  DRL has been widely used for sequential decision making in an unknown environment, where the agent learns a policy to optimize a cumulative reward by trial-and-error interactions with the environment. In each iteration, the agent observes the current state of the environment and then chooses an action based on the current policy. The environment moves to a new state and reveals the reward, and the policy is updated based on the received reward.

​	   现在offline显性建模调度器生成trace 来训练DRL模型,  但是1 因为 ML 框架改了就要remodeling,  2 因为 性能模型没有考虑干扰就不准确,   用历史轨迹 缺点是: 没有所有可能的决策因为决策空间很大.    

​	Therefore, instead of offline training in a simulated environment, we advocate online RL in the live cluster, exploiting true feedback for resource allocation decisions produced by the DRL agent, to learn a good policy over time. Pure online learning of the policy network model from scratch can result in poor policies at the beginning of learning (see Fig. 10). To avoid poor initial decisions and for the smooth transition from an existing scheduler, we adopt offline supervised learning to bootstrap the DRL policy with the existing scheduling strategy.

​	我们提出  一开始离线监督学习因为 一开始纯online RL 收敛很慢,  而且现在的调度器可以继续用在offline.然后  在online, 给DRL agent 产生的 resource allocation policy true feedback , 让他学习策略.

## 3DL overview

​	DL2 的最终目标是在实时DL 集群中找到最佳资源分配策略，并最小化所有并发作业的平均作业完成时间。

### 3.1 DL 集群

​	在具有多个 GPU 服务器的 DL 集群中，随着时间的推移提交 job。 每个job都运行一个分布式 ML 框架（例如我们的实验中的 MXNet），重复训练数据集来learning特定的 DL 模型。 提交作业后，job owner，提供她/他分别运行每个worker和每个 PS 的资源需求，以及要运行的训练时期总数。 例如，一个worker至少 1 个 GPU 和一个 PS 需要许多 CPU 内核。 可以基于专家知识或工作经历来估计实现模型收敛的总训练epoch数（例如，模型的损失或准确性的收敛)

​	根据资源可用性和训练速度，每个作业可能会从一个时间段到另一个时间段运行不同数量的worker和 PS（由调度程序决定）对于同步训练，为了保证相同的训练结果（模型）同时改变worker的数量，我们调整每个worker的mini batch大小，以便用户指定的作业中的总batch大小仍然保持不变  对于异步训练，每个worker的小批量大小保持不变，而worker数量不同（因为全局批量大小等于每个worker的批量大小）

### 3.2 DL Scheduler

​	Our DL-based scheduler, DL2, adopts joint offline and online learning of a policy neural network (NN) for making resource allocation decisions to concurrent jobs in the cluster. An overview of DL2 is given in Fig. 5. 

#### Offline supervised learning. 

For warm-up, we use supervised learning to train the policy NN, to initialize a policy whose performance is as good as the existing scheduler in the DL cluster. A small set of historical job runtime traces collected from the cluster are used for supervised learning, to allow the NN to produce similar decisions as made by the existing scheduler. This step is a must due to the poor performance of applying online RL directly (see Fig. 10)   DL2 采用策略NN (NN) 的联合离线和在线学习，为集群中的并发作业做出资源分配决策。 图 5 给出了 DL2 的概述。

离线监督学习。  warm up 使用监督学习来训练策略NN，以初始化一个策略，其性能与 DL 集群中现有的调度程序一样好。 从集群中收集的一小组 runtime trace用于监督学习，以允许 NN 生成与现有调度程序所做的类似决策。 由于直接应用在线 RL 的性能不佳， this step is a must.

#### Online reinforcement learning. 

Online RL works in a time-slotted fashion; each time slot is a scheduling interval, e.g., 1 hour. At the beginning of a scheduling interval, the policy NN takes the information of all the concurrent jobs as input state, and produces the numbers of workers and PSs for each job. The concurrent jobs include new jobs arrived in the previous time slot (after previous scheduling) and jobs which were submitted earlier and whose training has not been completed yet. Workers and PSs are placed on physical machines following the placement policy in the cluster, such as load balancing . Jobs’ training progress is observed at the end of each time slot, and used as the reward to improve the policy network.

比如一个小时调度一次 

调度的对象: 之前没完成的和这个小时新来的job

输入: 所有并发的job信息

输出:  每个job的 PS和worker数.

reward:  job的训练进度. training progress.                       

## 4 detailed design

### 4.1 Policy Neural Network

为什么要用这些参数? 不告诉你d行不行? 这些信息我认为是容易获得的, 而且有助于做出优秀决策的输入.

State. The input state to the policy NN is a matrix 

输入包括 job type,  job has run的时间,  remaining epochs , allocated resources, allocated workers, allocated PSs.https://q.uiver.app/?q=WzAsOCxbMywxLCJwb2xpY3lOTiJdLFs1LDEsImFsbG9jYXRlXFxcXHdvcmtlci9QUyJdLFswLDAsImpvYlxcXFx0eXBlIl0sWzAsMSwicnVudGltZSJdLFswLDIsInJlbWFpbmluZ1xcXFxlcG9jaCJdLFswLDMsImRvbWFpbnQgcmVzb3VyY2UiXSxbMCw0LCJqb2Inc1xcXFx3b3JrZXIiXSxbMCw1LCJqb2Inc1xcXFxQUyJdLFswLDFdLFsyLDBdLFs3LDBdXQ==

- x 表示在作业中训练的 DL 模型JxL的矩阵，其中 J 是我们正在调度的时间段中并发作业的最大数量的上限，L 是集群中任何时候训练作业类型的最大数量。 我们将 DL 作业训练类似的 DNN 架构视为我们输入中的相同类型。 例如，基于相同的预训练模型的微调作业很常见，它们可以被视为同一类型。(这应该是one hot 编码的)

- d 表示  一个 J 维向量，编码每个作业在集群中运行的time slot数，用于所有作业。 例如，di 是作业 i 运行的time slot数。也就是输入运行了多久. 

- e 一个 J 维向量，编码为每个作业训练的剩余 epoch 数。ei 是用户指定的总训练时期数之间的差值. 也就是输入还需要多久. 这些是怎么调整的? yixin bao说是尝试了各种输入找到效果最好的. 

- r一个 J 维向量, 表示分配给每个job的主导资源, 例如，ri 是分配给作业 i 的主导资源（与集群中资源的总容量相比，作业占用的资源最多） 这个就是类型. 比如CPU , GPU .  每个worker的单位资源是固定的, 不是job owner提交的,  在论文中 一开始确定下来简化问题. 

- w和u 它们中的每一个都是一个 J 维向量，其中第 i 项是当前时隙中分配给作业 i 的worker (PS) 的数量。 

根据作业的到达时间进行排序。 输入状态不直接包括集群中可用的资源容量； 我们的调度程序可以处理集群中随时间变化的整体资源容量。(因为是一个个放置, 直到用完为止.)

  这个state 一次是放多少? 是放这么大一个矩阵, 所以行数 = job的+5,列数= job的类型  

**action** 输出包括  3 × J + 1 个动作

 NN 产生策略 π : π(a | s; θ) → [0, 1]，这是动作空间上的概率分布。  a 代表一个动作，θ 是NN中的当前参数集。 一个简单的设计是允许每个动作指定分配给所有并发作业的worker/PS 的数量； 这导致了一个指数级大的动作空间，包含所有可能的worker/PS 编号组合。 大的action space会导致大量的训练成本和缓慢的收敛

  为了加快NN的学习，我们简化了动作定义，并允许NN通过每个推理从以下 3 × J + 1 个动作中输出一个动作：(i) (i, 0)，意味着分配一个worker到工作 i, (ii) (i, 1), 为作业 i 分配一个 PS, (iii) (i, 2), 为作业 i 分配一个worker和一个 PS, (iv) 一个无效动作，表示当前时隙停止分配资源（因为分配更多资源不一定会导致更高的训练速度)由于每个推理只输出要分配给 J 个作业之一的增量资源，因此我们允许对 NN 进行多次推理，以在每个时隙中生成完整的资源分配决策集( 这个是说需要一个timeslot的时间生成吗?)：在生成一个动作后，我们更新状态 s， 然后使用NN产生另一个动作； 重复推理直到资源被使用完或产生无效动作。  

  虽然我们在每个时间段内为每个作业重新生成worker/PS ，但对于在前一个时间段内运行的作业，我们比较新的和以前的数量并执行动态缩放以仅调整部署数量（§5  ）

  **NN架构**。 输入状态矩阵 s 连接到一个全连接层，使用 ReLU [48] 函数进行激活。 这一层的神经元数量与状态矩阵的大小成正比。 该层的输出聚合在一个隐藏的全连接层中，然后连接到最终的输出层。 最后的输出层使用 softmax 函数 [25] 作为激活函数。  NN 架构是基于经验训练试验empirical training trials设计的(作者也不知道为啥反正可以work)

### 4.2 Offline Supervised Learning

​	In offline supervised learning, we use stochastic gradient descent (SGD) [56] to update parameters θ of the policy NN to minimize a loss function, which is the cross entropy of the resource allocation decisions made by the NN and decisions of the existing scheduler in the traces [38]. The NN is repeatedly trained using the trace data, e.g., hundreds of times as in our experiments, such that the policy produced by the NN converges to the policy of the existing scheduler.

​	用随机梯度下降 SGD 来更新参数,  就是NN资源分配决策和目前scheduler trace决策的交叉熵. 训练几百次.收敛到现有调度器的决策.

### 4.3 在线强化学习奖励。

​    Reward. DL2 targets average job completion time minimization in the entire cluster. Job completion time would be a natural reward to observe, but it is only known when a job is finished, which may well be hundreds of time slots later. The significant feedback delay of the reward is unacceptable for online RL, since the delayed reward provides little guidance to improve the early decisions. We design a per-timeslot reward to collect more reward samples through the job processes, for more frequent RL model updates to expedite convergence. The per-timeslot reward is the sum of normalized number of epochs that the concurrent jobs have trained in this time slot, where the number of epochs trained in job i (ti) is normalized over the overall number of epochs to train for the job (Ei)

  reward ,  每个job的rt加起来. rt = 这个timeslot训练的epoch/总共要训练的epoch 

​	基本原理是job一个时间段内运行的 epoch 越多，完成所需的时间段就越少，因此最大化累积奖励相当于最小化平均作业完成时间。 规范化是为了防止偏向大型工作。

怎么更新NN ? 

用 累计reward求导.

​	**基于策略梯度的学习。** 在在线强化学习中，通过离线监督学习获得的策略NN使用 REINFORCE 算法 [62] 进一步训练，以最大化预期累积折扣reward  我们将问题建模为具有长期影响的非线性问题，而不是具有一轮独立反馈的传统线性模型，例如上下文老虎机 [36]，因为不同时隙中的动作是相关的。  REINFORCE 算法通过对 E[ 求和∞ t=0 −γtrt] 执行 SGD 来更新策略网络的参数 θ。

​    梯度可以先计算 Q,  Q= 选择action a后的预期 cumulative discounted reward.  Q 根据 minibatch 样本来计算.

   notice  与标准 RL 的运行方式不同：我们在每个时隙 t 中使用 NN 进行多次推理（即产生多个动作）； 每次推理后 input state 都会发生变化； 我们只在时间段中的所有推理完成后观察reward 并更新 NN 一次。 我们可以在一个时隙 t 中获取多个sample，并将每个样本中的奖励设置为在 t 中完成所有推理后观察到的奖励（1） 

(其实稳定是之后的事情, 最重要的是先搞出RL 的框架)

  我们进一步采用了多种技术来稳定在线 RL，加速策略收敛，并提高所获得策略的质量。 

**actor-critic** 我们使用 actor-critic 算法 [46]（如图 6 所示）改进了基于梯度的基本策略强化学习，以加快策略网络的收敛速度。 基本思想是 让Q减去一个函数 Q(a, s; θ) − Vπ(s, θ)，其中 Vπ(s, θ) 是一个价值函数,  表示期待的reward   representing the expected reward over the actions drawn using policy π(a | s; θ) at all times starting from time slot. 

这可以显示特定行动的优劣.(为什么?要计算的是在某一个状态 s 采取某一个动作 a 的时候，优势函数有多大。 critic 就是估计advantage的. )

   还确保梯度的方差小得多，从而使policy学习更加稳定。 价值函数由价值网络评估， 网络结构和policy NN 相同，只是其最终输出层是一个没有任何激活函数的线性神经元[46]，并产生价值函数 Vπ(s,  θ)。 输入状态也和策略网络相同。 我们使用temporal  difference 方法[46]训练价值网络。 这里具体的实现要看 46号文献(简单来说, 时序差分强化学习方法是**每一个step就更新一次** ，（比如我们的贪吃蛇游戏，贪吃蛇每移动一次（或几次）就进行更新）。相对来说，时序差分强化学习方法比蒙特卡洛强化学习方法更新的频率更快。时序差分强化学习能够在知道一个小step后就进行学习，相比于蒙特卡洛强化学习，其更加**快速、灵活**。)

​	**job-aware探索**。 为了通过 RL 获得好的策略，我们需要确保充分探索行动空间（即可以充分产生导致良好回报的行动）； 否则，RL 可能会收敛到较差的局部最优策略 [60] [46]。 我们首先采用一种常用的熵entropy探索方法，通过在梯度计算中添加熵正则化项 来更新策略网络[46]。  这样策略网络的参数 θ 朝着更高熵的方向更新（意味着探索更多的动作空间）

​	在训练期间，由于不了解工作语义，我们观察到大量不必要的或糟糕的探索（例如，为工作分配多个worker但 0 PS）。 为了提高探索效率，我们采用了另一种基于omega-greedy 方法的技术[55]。 在使用策略网络的每次推理中，我们检查输入状态：如果输入状态属于我们已经识别的不良状态之一.  那么 概率为 1 − omega应用策略网络产生的资源分配决策， omega概率，我们丢弃来自策略网络的输出，采用指定的动作并观察该动作的奖励。

   差输入状态集包括三种情况：（i）存在一个要调度的作业，该作业已分配给多个worker但没有 PS；  (ii) 存在一份工作分配了多个 PS 但没有worker；  (iii) 存在一项工作，其分配的worker (w) 和 PSs (u) 的分配数量差异太大，即 w/u > 阈值或 u/w > 阈值（在我们的实验中阈值为 10） 我们对这些输入状态中的每一个手动指定的操作是：（i）为该作业分配一个 PS；  (ii) 为该工作再分配一名worker；  (iii) 为该工作再分配一名 PS 或一名worker，使其worker/PS 数量更加均衡。
	**experience replay**。 众所周知，样本之间的相关性会阻止 actor-critic 模型收敛到一个好的策略 [55]。 在我们的在线 RL 中，当前的策略网络确定了以下训练样本，例如，如果策略网络发现分配更多的worker可以提高奖励，那么以下样本序列将由该策略产生的样本序列主导； 这可能会导致糟糕的反馈循环，从而阻止对具有更高奖励的样本进行探索。 

  为了减轻观察到的样本序列中的相关性，我们在 actor-critic 框架中采用了经验回放 [47]。 具体来说，我们维护一个重放缓冲区来存储在最新时隙中收集的样本。 在每个时间段的末尾，我们选择一个小批量的样本，而不是使用在这个时间段收集的所有样本samples from the replay buffer to compute the gradient updates, where the samples could be from multiple previous time slots.

## 5 dynamic scaling

 现在tf,caffe, mxnet都不支持 动态调整resource, 一个方法, optimus用检查点, 终止工作把global 模型参数保存为检查点,	用新的PS 和worker部署重新启动job, 这样会有延迟. 

资源如果一个小时拓展一次, 那么开销很大. 我们改进了mxnet框架来实现动态热缩放. 

​	Though node addition and deletion are supported in system design in the literature , existing opensource distributed machine learning frameworks (e.g., TensorFlow [15], MXNet [20], Caffe [5]) do not support dynamic worker/PS adjustment in a running job. To adjust the number of workers/PSs in a job, a simple and general approach is checkpointing (e.g., Optimus [49]): terminate a training job and save global model parameters as a checkpoint image; then restart the job with a new deployment of PSs and workers, and the saved model parameters. Checkpointing and restarting add additional delay to the training process [50]. For example, it takes 1 minute to checkpoint and stop training, and another 5 minutes to completely restore training of a DSSM model [52], due to data re-preprocessing before training starts. The overhead is significant when the frequency of resource scaling is high (e.g., every hour). The other approach is to resize resources without terminating training process. As an example, we improve the MXNet framework [20] to enable dynamic “hot” scaling. 

每个PS维护参数的一个子集, PS变了, 需要PS之间migrated 全局参数 , 并通知worker 把参数更新发送给新的PS.  难点1. 正确性, 参数跨PS移动时 保持全局模型参数一致,  worker 不会发错 2 . 高性能, 中断少,PS负载均衡.  

**Challenges**. In the parameter server architecture, each PS maintains a subset of the parameters in the global model. When the number of PSs changes, the global parameters need to be migrated among the PSs (for load balancing), and workers should be informed in time to send parameter updates to the correct PSs. When the number of workers changes, the new connections between new workers and the PSs should be established. The key challenges are: (1) correctness, i.e., a consistent copy of the global model parameters should be maintained while parameters are moved across the PSs, and workers always send gradients to correct PSs; (2) high performance, i.e., we should ensure that interruption to training is minimized and the PSs are load balanced.

缩放步骤, MXNet 加了 coordinator 协调器,   和DL2 一起处理新worker加入和现有worker的终止, 

**Scaling Steps.** We add a coordinator module into the MXNet framework, which works with DL2 scheduler to handle joining of new workers or PSs and termination of existing ones. We demonstrate our design using the case of adding a new PS into an existing job. The steps are shown in Fig. 7.

第一步注册, 第二步参数分配, 第三步 参数迁移 第四步 worker更新参数映射PS

1/ 注册,  PS启动, 向 coordinator 发送inc server 请求 来注册自己, PS接收 1它在job中的id , 2它负责维护的全局参数, 和3当前PS,worker的列表 to 建立联系,    之后PS开始工作, 等待worker发的参数更新和进一步的instruction. 

1) registration When a new PS is launched, it registers itself with the coordinator by sending an “INC SERVER” request message. The PS will then receive its ID in the job, the global parameters it is responsible to maintain, and the current list of workers and PSs to establish connections with. After that, the PS starts functioning, awaiting workers’ parameter updates and further instructions from the coordinator (e.g., parameter migration).

2  参数分配, coordinator 更新 PS和worker列表, 计算分配给新PS的参数, 采用best fit : 把每个PS部分参数move过去,  让每个PS参数数量差不多, 同时 minimizing参数移动. 

*2) Parameter assignment.* Upon receiving a registration request, the coordinator updates its list of workers and PSs, and computes parameter assignment to the new PS. A best-fit algorithm is adopted: move part of the parameters on each existing PS to the new PS, such that all PSs maintain nearly the same number of parameters, while minimizing parameter movement across the PSs. 

为了在迁移参数时 保持全局模型参数的一致副本, 我们维护了一个版本计数器,     PS来说是参数更新的次数,  worker来说, pull 参数同时拿到 版本计数器.   根据 版本计数器和  round trip time 来计算 scaling clock 就是啥时候迁移参数.  

​	In order to keep a consistent copy of global model parameters when migrating parameters among PSs, we maintain a version counter for parameters. For PSs, the version counter is the number of parameter updates; for workers, the version counter is received from PSs when pulling updated parameters. To decide when PSs should migrate parameters, we calculate a scaling clock based on current version counter and round trip time between the coordinator and PSs/workers.

协调器向所有PS和worker 发送  新参数分配和 scaling clock.

​	The coordinator sends new parameter assignment among PSs and the scaling clock to all PSs and workers.

3) Parameter migration. At each PS, when the version counter of parameters reaches the scaling clock received from the coordinator, the PS moves its parameters to the new PS according to the parameter assignment decisions received. Once parameter migration among all PSs is completed, the coordinator notifies all workers to resume training. 

4) Worker update. At each worker, once its version counter equals the scaling clock received from the coordinator, the worker suspends its push/pull operations and awaits notification for completion of parameter migration. Upon notification from the coordinator, the workers update their parameter-PS mapping, establish connections with the new PS, and resume the training process.
*3) 参数迁移。* 当参数的版本计数器达到从协调器接收到的缩放时钟时，PS 根据接收到的参数分配decisions 将其参数移动到新的 PS。  所有PS参数迁移完成后,  协调器notifies 所有worker 继续训练 

*4) worker更新。* 在每个 worker 上，一旦其版本计数器等于从协调器接收到的缩放时钟，worker 就会暂停其推/拉操作并等待参数迁移完成的通知。 收到协调器的notification后，worker更新他们的参数-PS 映射，与新的 PS 建立连接，并恢复训练过程。

​	In case of removing a PS, the scheduler chooses the PS to be removed by keeping the load balanced among the physical machines. The chosen PS sends a removal request to the coordinator. Similar steps as 2)3)4) above are then carried out, except that parameters in the removed PS are moved to other PSs, using the best-fit algorithm.    重新  第二步参数分配, 第三步 参数迁移 第四步 worker更新参数映射PS 

​	新加worker1的话,  协调器把当前的参数-PSmapping关系发给worker1, 然后 通知所有PS有新worker. 复制了训练数据集后, worker开始工作. 移除worker1的话, 广播给所有PS, 还要调整mini batch 大小.让总batch 相同.    sends the current parameter-PS mapping in the response to the worker’s registration message. It also notifies all PSs the addition of the new worker for building connections. The worker starts operation after training dataset is copied. For worker removal, the scheduler chooses the worker to be removed by keeping the load balanced across physical machines. The coordinator receives a removal request from the worker, and then broadcasts it to all workers and PSs for updating their node lists. The mini-batch size of workers is adjusted so as to keep total batch size the same.

## 6 评估

### 6.1 DL2 Implementation

  We implement DL2 as a custom scheduler on Kubernetes [9]. We run each training job using the MXNet framework [20]. Workers and PSs are running on Docker containers. Training data of jobs are stored in HDFS 2.8 [3]. The scheduler constantly queries cluster resources and job states (e.g., training speeds) and instructs deployment of a new job or resource adjustment in an existing job via Kubernetes API server. Mapping the cluster and job states to a scheduling decision takes less than 3ms. 

  For each new job, DL2 launches its coordinator, workers, and PSs on machines decided by the default placement strategy of the cluster (i.e., load balancing). The coordinator is informed of the workers and PSs in the job via Kubernetes API. When a worker/PS container is launched on a machine, an agent in the container starts execution. It queries the readiness of other containers of the same job via Kubernetes API and starts user-provided training scripts after all other containers are ready. The agent also monitors the training status, e.g., the number of trained steps, accuracy, and training speed.

我们将 DL2 实现为 Kubernetes [9] 上的自定义调度程序。 我们使用 MXNet 框架 [20] 运行每个训练作业。  Worker 和 PS 在 Docker 容器上运行。 作业的训练数据存储在 HDFS 2.8 [3] 中。 调度器不断查询集群资源和作业状态（例如训练速度），并通过 Kubernetes API 服务器指示部署新作业或在现有作业中调整资源。 将集群和作业状态映射到调度决策所需的时间不到 3 毫秒。

 对于每个新作业，DL2 在由集群的默认放置策略（即负载平衡）决定的机器上启动其协调器、工作器和 PS。 协调器通过 Kubernetes API 获知作业中的worker和 PS。 当在机器上启动 worker/PS 容器时，容器中的代理开始执行。 它通过 Kubernetes API 查询同一作业的其他容器的准备情况，并在所有其他容器准备就绪后启动用户提供的训练脚本。 代理还监控训练状态，例如训练步数、准确性和训练速度。

### 6.2 Methodology 

**Testbed.** Our testbed includes 13 GPU/CPU servers connected by a Dell Networking Z9100-ON 100GbE switch. Each server has one Intel E5-1660 v4 CPU, two GTX 1080Ti GPUs, 48GB RAM, one MCX413A-GCAT 50GbE NIC, one 480GB SSD, and one 4TB HDD. Each server runs Ubuntu 14.04 LTS and Docker 17.09-ce.

**Trace.** We use patterns from a 75-day real-world job trace collected from a large production DL cluster with a few thousands of GPUs and thousands of jobs, to drive our testbed experiments and simulation studies. Fig. 8 (a) shows the job arrival rate (number of jobs arrived per time slot, i.e., 20 minutes) during a typical week. Fig. 8 (b) shows the distribution of job duration: over a half of jobs run for more than an hour and some for days; the average job duration is 147 minutes.

​	Due to security and privacy concerns of the company, the job source code is not available, and we do not know job details (e.g., model architecture). So we select 8 categories of ML models for experiments, from official MXNet tutorials [10], with representative application domains, different architectures and parameter sizes [10], as shown in Table 1. Each worker in different jobs uses at most 2 GPUs and 1-4 CPU cores, and each PS uses 1-4 CPU cores. 

这个就是trace.py里面的self.resr worker 和resr ps.

​	In both testbed experiments and simulations, the jobs are submitted to the cluster following the dynamic pattern in Fig. 8 (a) (with arrival rates scaled down). Upon an arrival event, we randomly select a model from Table 1 and vary its required number of training epochs (tens to hundreds) to generate a job variant, following job running time distribution of the real-world trace (scaled down). For models training on large datasets (e.g., ImageNet [8]), we downscale the datasets so that the training can be finished in a reasonable amount of time. In experiments, 30 jobs are submitted to run in our testbed; in simulations, 500 servers are simulated, and 200 jobs are submitted in the simulated cluster

这里有讲testbed 随机选一个模型, 产生一个job变量. 根据 fittng的function 改变 required epoch.   

**Training setting.** Our DL-based scheduler is implemented using TensorFlow [15]. The neural network is trained using Adam optimizer [34] with a fixed learning rate of 0.005 for offline supervised learning and 0.0001 for online reinforcement learning, mini-batch size of 256 samples, reward discount factor γ = 0.9, exploration constant  = 0.4, entropy weight β = 0.1, and an experience replay buffer of 8192 samples. The network has 2 hidden layers with 256 neurons each. These hyper-parameters (neural network structure, learning rate, mini-batch size, etc.) are chosen based on a few empirical training trials. We refer to one update of the neural network at the end of each time slot as one step in this section



**Baseline**

主导资源公平性（DRF）[24]：它根据主导资源的公平性为工作分配资源。 默认情况下，我们使用 DRF 作为现有调度器，用于指导 DL2 中的监督学习，因为它在现有集群调度器中被广泛采用，例如 YARN [58]、Mesos [31]

 • 俄罗斯方块 TETRIS[27]：优先将资源分配给剩余完成时间最短、资源打包效率最高的作业。  

代码实现中,计算分数.

```python
mean_resr_score[job] = np.sum(resr) * (1 - job.progress / job.num_epochs)
mean_align_score[job] = np.sum((pm.NUM_RESR_SLOTS - used_resrs) * resr)
NUM_RESR_SLOTS = 8  # number of available resource slots on each machine  (插槽的数量- 用了的数量)*job.resr_worker
node, used_resrs = node_used_resr_queue.get()
resr = job.resr_worker
```

 • Optimus [49]：它是一个定制的DL 工作负载调度器，它为深度学习作业构建性能模型以估计剩余训练时间，并采用贪婪启发式调度作业。 

 • OfflineRL：离线强化学习算法采用纯离线训练，在与 DL2 中的在线 RL 相同的训练设置下，除了训练数据由模拟环境中的分析性能模型 [49] 生成（我们不使用trace，因为它 不包含对离线培训产生的所有决定的反馈）

 在适当的情况下，我们使用单独的训练数据集和验证数据集。 Both include job sequences generated using the job arrival and duration distributions from the trace. 生成数据集时随机种子是不同的，以确保它们是不同的。

### 6.3 Performance

We first compare the performance of DL2 with baselines and show the overhead of dynamic scaling using testbed experiments. 

**Comparison**. Fig. 9 shows that DL2 improves average job completion time by 44.1% when compared to DRF. Tetris performs better than DRF but worse than DL2: once it selects a job with the highest score in terms of resource packing and remaining completion time, it always adds tasks to the job until the number of tasks reaches a user-defined threshold. When compared to Optimus, DL2 achieves 17.5% higher performance, since Optimus’ estimation of training speed is inaccurate due to cluster interference and evolved MXNet framework (e.g., communication does not overlap with backward computation in Optimus’ model). DL2 also outperforms OfflineRL by37.9% due to its online training using realistic feedback.

​    For a better understanding of DL2’s performance gain,Fig. 10 shows how the validated performance keeps improving during the training process, when the policy NN is trained using offline supervised learning only (green curve), online RL only (cyan curve), and offline supervised learning followed by online RL (green+blue). The average job completion time shown at each time slot (i.e., step) is computed over job sequence in the validation dataset, using the policy network trained (on the training dataset) at the current step. We see that with pure online RL, it takes hundreds of steps to achieve the same performance of DRF; with offline supervised learning, the performance quickly converges to a point that is close to DRF’s performance within tens of steps (i.e., model updates); as we continue training the NN using online RL, the performance further improves a lot. The performance of DRF is fixed as its strategy does not change over time. Besides smaller job completion time, we also observe that DL2 has higher CPU and GPU utilization (similar observation as in [49]).

### 6.4Generality

**Training completion time variation.** To see how DL2 handles practical performance variation (which white-box schedulers may not handle well), we vary the training speeds in each type of jobs to simulate variation in the training completion time of the same type of jobs (the total numbers of epochs to train remain the same). In Fig. 13, the variation indicates how the training speed deviates from the average speed (which can be faster or slower by the respective percentage). We see that Optimus is more sensitive to the variation, as it can be easily stuck in local optimum: its scheduling relies on the convexity of the performance model, but training speed variation often breaks convexity. The average job completion time shown in all simulation figures is in time slots.



**Other scheduling strategies for supervised learning.** We change the default DRF used in supervised learning of DL2 to two other heuristics, First-In-First-Out (FIFO) and Shortest-Remaining-Time-First (SRTF). Fig. 16 shows average job performance when DL2 uses each of these strategies in its supervised learning phase, when the NN trained only using supervised learning, or using both supervised learning and online RL, is evaluated on the validation dataset. In both cases, the performance is significantly improved with DL2, beyond what the existing scheduling strategy in the cluster can achieve (41.3% speedup in the case of SRTF).(代码上srtf和fifo好像很简单, 就push的依据换了一下, srtf 就是 看progress,  )

### 6.5 Training Design 

SL loss function. We evaluate three common loss functions for supervised learning, i.e., Mean Square, Cross Entropy (the default) and Absolute Difference [13]. We observe similar performance with these loss functions, while adopting Cross Entropy achieves the best performance. This is because Mean Square or Absolute Difference emphasize incorrect or suboptimal output, while only the correct or optimal output contributes to the loss when using Cross Entropy.6.5 训练设计 SL 损失函数。 我们评估了监督学习的三种常见损失函数，即均方、交叉熵（默认值）和绝对差值 [13]。 我们观察到与这些损失函数相似的性能，而采用交叉熵实现了最佳性能。 这是因为均方或绝对差强调不正确或次优的输出，而在使用交叉熵时，只有正确或最佳的输出才会导致损失。

Reward function. We evaluate another reward function with DL2, which sets the reward of each action (that adds some worker/PS to a job) as the normalized number of epochs trained by the job in the time slot. We find that its performance is 29.1% worse. Our default reward function considers all jobs’ progress, enabling the policy network to learn to schedule from a global perspective. 

Actor-critic. To see how the actor-critic algorithm affects training, we remove the value network but only train the policy network. As widely adopted in RL community, we use the exponential moving average of rewards as a baseline in place of the output of the value network in gradient computation of the policy network. As shown in Table 2,

reward 函数。 我们使用 DL2 评估另一个奖励函数，它将每个动作的奖励（向工作添加一些worker/PS）设置为工作在时间段内训练的归一化时期数(不一样吗? 默认奖励函数是啥? )。 我们发现它的性能差了 29.1%。 我们的默认奖励函数会考虑所有工作的进度，使策略网络能够学习从全局角度进行调度。 

 为了了解 actor-critic 算法如何影响训练，我们移除了价值网络，只训练了策略网络。 正如 RL 社区广泛采用的那样，我们在策略网络的梯度计算中使用奖励的指数移动平均值作为基准，代替价值网络的输出。 

This is because the average reward is not always an effective baseline; in some cases, even the optimal action leads to a lower reward than the average reward. 

**Job-aware exploration**. We examine how exploration contributes to the performance. From Table 2, we see that without exploration the performance is 28.8% worse, as online RL is stuck in a local optimal policy. 

**Experience replay.** We disable experience replay and see how performance changes. Table 2 shows that the average job completion time is degraded by 39.6%, indicating that experience replay is critical for training. 

**Federated training.** Federated training enables multiple clusters to learn a global DL2 model collaboratively. We study how the number of clusters affects the policy training, by implementing the A3C [46] algorithm, which trains a global policy NN using multiple DL2 schedulers with different training datasets, each for one cluster. Fig. 18 shows that the global performance remains stable when we increase the number of clusters. We have also observed that with more clusters, the policy NN converges much faster due to the use of more training datasets: if there are x clusters, the NN converges almost x times faster. The preliminary result also suggests the possibility of dividing a single massive cluster into loosely coupled sub-clusters where each runs a DL2 scheduler for resource allocation, if scalability issue arises.

 这是因为平均奖励并不总是有效的基线； 在某些情况下，即使是最佳动作也会导致比平均奖励更低的奖励。

 工作意识探索。 我们研究了探索对性能的贡献。 从表 2 中我们可以看到，如果没有探索，性能会差 28.8%，因为在线 RL 陷入了局部最优策略。 

体验回放。 我们禁用体验重放并查看性能如何变化。 表 2 显示平均作业完成时间降低了 39.6%，表明经验回放对于训练至关重要。

federated 联合训练。 联合训练使多个集群能够协作学习全局 DL2 模型。 我们通过实施 A3C [46] 算法来研究集群的数量如何影响策略训练，该算法使用具有不同训练数据集的多个 DL2 调度程序训练全局策略NN，每个调度程序用于一个集群。 图 18 显示，当我们增加集群数量时，全局性能保持稳定。 我们还观察到，对于更多的集群，由于使用了更多的训练数据集，策略 NN 的收敛速度要快得多：如果有 x 个集群，则 NN 的收敛速度几乎是 x 倍。 初步结果还表明，如果出现可扩展性问题，可以将单个大型集群划分为松散耦合的子集群，每个子集群都运行 DL2 调度程序以进行资源分配。

resource utilization. DL2 starts from offline supervised learning, to ensure basic scheduling performance comparable to the existing cluster scheduler, and then runs in the live DL cluster to make online scheduling decisions, while improving its policy through reinforcement learning using live feedback. Our testbed experiments and large-scale trace-driven simulation verify DL2’s low scaling overhead, generality in various scenarios and outperformance over hand-crafted heuristics.资源利用率。  DL2 从离线监督学习开始，保证基本调度性能与现有集群调度器相媲美，然后在实时 DL 集群中运行以做出在线调度决策，同时通过使用实时反馈的强化学习改进其策略。 . 我们的测试平台实验和大规模跟踪驱动模拟验证了 DL2 的低扩展开销、各种场景中的通用性以及优于手工启发式算法的性能。

## 7 Discussion and Future Directions

​	**更多调度功能**  可以调整学习目标, 比如 incorporate resource fairness by adding a quantified fairness term in the reward function.

​	**all reduce 架构** caffe 和cntk 支持, worker 可以直接交换模型参数.   把PS相关删除应该就可以支持了

**job placement** 我们用的是默认的placement 策略, 但是worker和PS的placement 也可以用RL决定,  用NN来同时产生 allocation和polacemnt 很难, 因为action space 太大了, 但是我们可以用hierachicalNN model  . 这也就是老师要我去做的.  但是如果直接把他和harmony  连起来就很糟糕, 收敛性不好, 我在看pollux 2021是否能有联合训练的好方法. 看看是怎么调度的. 

**实际部署**。 在实际部署中，可能需要考虑以下两个问题：（1）恶意攻击欺骗NN输入;  (2) 检测异常调度的NN监控。 随着安全研究的进步和对NN的更深入了解，这些都是值得探索的有趣方向。

## 8 Related Work

 很多人用来 DRL , 有的静态资源分配 ,    有的根据任务执行顺利来调度基于图的并行job. 	 但是他们没有研究 job执行同时调整资源. 

​	Deep reinforcement learning in system research. A number of recent studies use DRL for resource allocation, device placement, and video streaming. Mao et al. [39] and Chen et al. [21] use DRL for job scheduling in cloud clusters, to minimize average job slowdown. Their NNs select the jobs (single-task jobs) to run with static resource allocation. The NNs are trained offline: multiple job arrival sequences are used as training examples; each example is repeatedly trained for multiple epochs. Mao et al. [41][42] learn an NN to schedule graph-based parallel jobs as in Spark, in terms of parallelism level and execution order of tasks in the jobs, using offline training. Adjustment of resources during job execution is not in the scope of the above studies.

Mirh他们也是优化计算图的placement ,   Xu他们用DRL选择网络结点的路由,  mao 用DRL动态决定视频流速率. 但是他们没有用在线RL来改进NN ,都是离线用自己建模的分析模型或模拟器生成的数据来训练. 

Mirhoseini et al. [45][44] use DRL to optimize placement of a computation graph, to minimize running time of an individual TensorFlow job. Xu et al. [66] use DRL to select routing paths between network nodes for traffic engineering. Mao et al. [40] dynamically decide video streaming rates in an adaptive streaming system with DRL. All these studies resort to offline RL training, using data generated by analytical models or simulators. In contrast, we use offline supervised learning to prepare our NN and then online RL to further improve the NN.

SLAQ 用在线fitting 调度经典ML , 但是没有人用使用PS架构的分布式ML job.  optimus提出了 基于在线fitting 资源性能模型的动态资源scheduler.   但是这些都依赖详细建模和简化假设.   gandiva 把job 迁移到更合适的gpu , gpu共享的资源分配也是一个探索方向.  (好像PS架构现在用的也没有all reduce多)

**ML cluster scheduling.** SLAQ [67] adopts online fitting to estimate the training loss of convex algorithms, for scheduling jobs training classical ML models. Dorm [54] uses a utilization-fairness optimizer to schedule ML jobs. These work do not focus on distributed ML jobs using the parameter server architecture. Optimus [49] proposes a dynamic resource scheduler based on online-fitted resource-performance models. Bao et al. [18] design an online scheduling algorithm for DL jobs. These studies rely on detailed modeling of DL jobs and simplified assumptions in their design. Gandiva [63] exploits intra-job predictability to time-slice GPUs efficiently across multiple jobs, and dynamically migrate jobs to better-fit GPUs. They do not consider resource allocation adjustment; Resource allocation with GPU sharing will be an intriguing future direction to explore.

## 9 Conclusions

​	We present DL2, a DL-driven scheduler for DL clusters, which expedites job completion globally with efficient resource utilization. DL2 starts from offline supervised learning, to ensure basic scheduling performance comparable to the existing cluster scheduler, and then runs in the live DL cluster to make online scheduling decisions, while improving its policy through reinforcement learning using live feedback. Our testbed experiments and large-scale trace-driven simulation verify DL2’s low scaling overhead, generality in various scenarios and outperformance over hand-crafted heuristics.

