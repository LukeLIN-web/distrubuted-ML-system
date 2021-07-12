A Deep Learning-driven Scheduler for Deep Learning Cluster

A 尝试解决什么问题

用深度学习技术

B方法有哪些关键元素

C我可以怎么自己用这方法

D 

越来越多的公司部署了机器学习 (ML) 集群，在其中训练深度学习 (DL) 模型以提供各种 AI 驱动的服务。 有效的资源调度对于最大限度地利用昂贵的 DL 集群至关重要。 现有的集群调度器要么对 ML 工作负载特征不可知，要么使用基于操作员对特定 ML 框架和工作负载的理解的调度启发式，这些方法效率较低或不够通用。 在本文中，我们展示了可以采用 DL 技术来设计通用且高效的调度程序。 DL2 是 DL 集群的 DL 驱动调度程序，通过动态调整分配给作业的资源大小来针对global 训练作业远征。 DL2 提倡联合监督学习和强化学习方法：基于现有集群调度器产生的作业轨迹，通过离线监督学习对神经网络进行预热； 然后将神经网络插入实时 DL 集群，通过在 DL 作业的整个训练过程中进行的强化学习进行微调，并用于以在线方式决定作业资源分配。 通过在准备监督学习阶段应用现有集群调度器做出的过去决策，我们的方法可以实现从现有调度器的平滑过渡，并在最小化平均训练完成时间方面提供高质量的调度器。 我们在 Kubernetes 上实施 DL2，并在 MXNet 上的 DL 作业中启用动态资源扩展。 广泛的评估表明，DL2 在平均作业完成时间方面优于公平调度程序（即 DRF）44.1% 和专家启发式调度程序（即 Optimus）17.5%。

## 1 介绍

近年来，基于深度学习的技术在各个领域取得了突破，例如机器翻译 [16]、图像分类 [35] 和语音识别 [28]。 大公司已经部署了具有数万到数千个昂贵 GPU 服务器的 ML 集群，并在一个或多个上运行分布式训练作业,分布式 ML 框架（例如 TensorFlow [15]、MXNet [20]、Petuum [65] 和 PaddlePaddle [11]），以获得需要其 AI 驱动服务的 DL 模型。 即使使用并行训练，训练 DL 模型通常也非常耗费时间和资源。 有效的资源调度对于运行具有多个训练作业的共享 DL 集群至关重要，以便最好地利用昂贵的资源并加快完成训练。 当今的 ML 集群中存在两大阵营调度器。在第一个阵营中，通用云/集群调度器被应用，并且可能被定制，用于分布式 ML 作业调度。 例如，Google 使用 Borg [59] 作为其 DL 集群调度器； 微软、腾讯和百度使用自定义版本的类似 YARN 的调度程序 [58] 来管理 DL 作业。 使用的代表性调度策略包括先进先出 (FIFO) 和显性资源公平 (DRF) [24]。 这些调度器根据用户规范分配资源，并且不会在训练期间调整资源分配。 正如我们将在 §2.2 中看到的，为作业设置合适的资源量是困难的，静态资源分配会导致集群中的资源利用不足。

In the second camp, recent studies have proposed white-box heuristics for resource allocation in ML clusters [67][49][18]. Typically they tackle the problem in two steps: set up analytical models for DL/MLworkloads, and propose scheduling heuristics accordingly for online resource allocation and adjustment. Designing heuristics requires a deep understanding ofML frameworks and workloads, and the analytical model is tightly coupled with the ML framework implementation (e.g., a new feature or optimization in evolving ML frameworks may invalidate the analytical model) [49]. Further, the modeling typically does not consider interference in a multi-tenant cluster, where in average 27.3% performance variation may happen (§2.2)

In this paper, we pursue a DL cluster scheduler that
does not depend on expert heuristics and explicit modeling, resorting to a black-box end-to-end approach enabled by modern learning techniques. We propose DL2, a deep learning-driven scheduler for deep learning clusters, that dynamically adjusts resource allocation to training jobs on the go. DL2 learns resource allocation policies through experience using deep reinforcement learning (DRL): the policy neural network takes the current system state as input, produces resource allocation decisions for all the current training jobs and gradually improves the decisions based on feedback. However, merely applying off-the-shelf RL algorithms to scheduling does not produce high-quality decisions, and careful design according to the problem nature is in need.在本文中，我们寻求一种不依赖于专家启发式和显式建模的 DL 集群调度程序，采用现代学习技术支持的黑盒端到端方法。 我们提出了 DL，一种用于深度学习集群的深度学习驱动的调度程序，它动态调整资源分配以训练-

Existing DRL applications in resource scheduling scenarios [39][41] [42] (§8) use simulators to generate training data for offline training, and apply trained models for resource scheduling in a live system. The core of such a simulator is typically an explicit performance model as mentioned above, and hence the inaccuracy of the simulator may lead to low-quality trained model. Instead of extensive offline training over large simulation, DL2 takes a different approach: we bootstrap the model using minimal offline supervised learning with any available historical job traces and decisions of any existing scheduling strategy employed in the cluster; then we use online training with feedback from ongoing decision making in a live system, with carefully designed techniques to guide model convergence to high-quality decisions, which minimize average job completion time in the cluster.

资源调度场景中的现有 DRL 应用程序, 用模拟器生成用于离线训练的训练数据，并在实时系统中应用训练模型进行资源调度。 此类模拟器的核心通常是如上所述的显式性能模型，因此模拟器的不准确性可能会导致低质量的训练模型。 DL2 没有对大型模拟进行广泛的离线训练，而是采用了一种不同的方法：我们使用最小的离线监督学习引导模型，其中包含任何可用的历史作业跟踪和集群中采用的任何现有调度策略的决策； 然后我们使用在线训练和实时系统中正在进行的决策制定的反馈，使用精心设计的技术来引导模型收敛到高质量的决策，从而最大限度地减少集群中的平均工作完成时间。

In summary, we make the following contributions in
DL2:  In contrast to previous DL cluster scheduling approaches that require analytical performance model and job profiling, DL2 adopts a more generic design, i.e., using DRL to schedule DL workloads. Instead of simulation-driven RL model training, we adopt online training with real feedback from online resource allocation (§2).总之，我们在 DL 中做出了以下贡献： 与之前需要分析性能模型和作业分析的 DL 集群调度方法相比，DL 采用了更通用的设计，即使用 DRL 来调度 DL 工作负载。 我们采用具有来自在线资源分配的真实反馈的在线培训，而不是模拟驱动的 RL 模型培训（第 2 节）。

我们发现直接将简单的 RL 方法应用于我们的在线调度程序培训通常会导致错误的决策。 为了避免在线 RL 开始时的错误决策，我们在准备离线监督学习阶段应用 DL 集群中现有调度程序所做的过去决策。 我们的方法可以实现从现有调度程序的平滑过渡，并自动学习超越现有调度程序性能水平的更好调度程序（§3）。 为了优化在线 RL，特别是针对 DL 作业调度，我们提出了作业感知探索以在动作空间中进行有效探索，并采用额外的训练技术（例如，演员评论算法、经验回放）进行样本高效学习（第 4 节）。我们在流行的分布式 ML 框架 MXNet [20] 中设计和实现弹性扩展，以实现动态工作人员/参数服务器调整（§5）。 我们将 DL2 与 Kubernetes [9] 集成，并使用测试台实验和受控模拟仔细评估 DL2，由从生产 DL 集群收集的 DL 作业模式驱动。 评估结果表明，DL2 显著 outperforms other representative schedulers in various scenarios, e.g., 44.1% improvement in average job completion time as compared to the widely adopted DRF scheduler. We also demonstrate DL2’s scaling overhead and generality

## 2背景

### 2.1 PS框架

我们专注于参数服务器 (PS) 架构 [37]，该架构在分布式 ML 学习框架中广泛用于并行训练，例如 MXNet [20]、TensorFlow [15]、PaddlePaddle [11] 和 Angel [32]。 请注意，DL2 也可以扩展到 all-reduce，如第 7 节所述。 在 PS 架构中，模型，例如深度神经网络 (DNN)，在多个参数服务器 (PS) 之间进行分区，训练数据集在工作人员之间进行分割（即，在代表性数据并行训练模型中）。 每个worker的数据分区被分成minibatch； 每个工作人员在本地处理一个小批量并计算模型参数变化，通常表示为梯度。 梯度被推送到保持全局模型参数的 PS。 我们专注于同步训练，其中工人的训练进度是同步的，并且 PS 在每次迭代中收到所有工人的梯度后更新全局模型。 更新的参数被发送回工作人员。 工作人员通过使用更新后的参数处理下一个小批量来开始下一个训练迭代/步骤。 在整个数据集中的所有小批量处理完一次后，就完成了一个训练时期。 输入数据集通常训练多个时期，直到模型收敛。

### 2.2motivation

The typical workflow for a user to train a model in a DL cluster is as follows: The user specifies how many PSs and workers she/he wishes to use and the amount of resources (e.g., GPU, CPU) each PS/worker needs, and then submits the job to the scheduler (e.g., Borg [59], YARN [58], Mesos [31]). The scheduler allocates PSs and workers to the job according to both user demand and its scheduling strategy, and the allocated resources then remain fixed over the entire training course of the job. This workflow has two limitations, as illustrated below. 

**Difficulty in setting the right worker/PS numbers.**  有更多的PS和work的时候, 训练速度如何提高?  How does a job’s training speed improve when more PSs and workers are added to the job? We train 3 classical models, i.e., ResNet-50 [30], VGG-16 [53] and Seq2Seq [23], in our testbed of 6 machines (see §6 for hardware details), and measure their training speeds (in terms of the number of samples trained per unit time), when increasing the number of workers and keeping the number of PSs equal to the worker number. Each worker uses 1 GPU, 4 CPU cores, 10GB memory and each PS has 4 CPU cores, 10GB memory. In Fig. 1, the speedup is calculated by dividing the training speed achieved using multiple workers/PSs (they are deployed onto physical machines in a load-balanced fashion) by the training speed obtained using one worker and one PS colocated on a single machine. We observe a trend of decreasing return, i.e., adding PSs/workers does not improve the training speed linearly. This is because more communication overhead is incurred when there are more PSs or workers. 因为有更多的通信开销.

On the other hand, is an equal number of PSs and workers (as a general rule of thumb) always the best? We fix the total number of PSs and workers to be 12 and measure the training speed of two models under different combinations of PS/worker numbers (i.e., 4:8, 6:6, 8:4) [49]. Fig. 2 shows that Seq2Seq achieves highest training speed when there are 4 PSs and 8 workers, while VGG-16 is trained fastest with 6 PSs and 6 workers.

From the above, we see that it is challenging to reason about which job will have the largest marginal gain from extra resources and what the best PS-to-worker ratio is, as they are affected by many factors, e.g., allocated resources, models. Existing schedulers largely side-step this problem and leave it to the user to decide how many PSs/workers to use.

静态资源分配。 GPU 集群资源通常没有被充分利用：当一个训练作业完成时，它释放的资源（例如，昂贵的 GPU）可能会空闲，而不是被仍在运行的剩余作业利用。 图 3 显示了生产 DL 集群在 24 小时间隔内的 GPU 利用率，该集群具有大约 1000 个 P100 GPU 卡（由于匿名要求而删除了公司名称），其作业跟踪将用于我们的评估（第 6 节）。 我们看到 GPU 利用率水平随时间显着变化，当集群负载低/高时，为训练作业中的动态资源扩展/缩减提供了机会。 我们提倡在训练作业中随着时间的推移动态调整 worker/PS 数量，以最大限度地利用 DL 集群中的可用资源来加快作业完成。 有了这个，我们进一步不要求用户提交 他们的worker和PS数量,the number of workers/PSs they want to use in their jobs (who nonetheless may not be at the best position to decide that), but will decide the best worker/PS numbers for each user at each time based on both global resource availability and individual jobs’ performance.

**White-box heuristics.** There have been existing studies which explicitly model detailed relationship between the training speed and resources within jobs, and design scheduling heuristics based on the resource-speed model, e.g., SLAQ [67], Optimus [49] and OASiS [18]. They have two limitations. First, in order to derive an accurate performance model, the modeling process is coupled tightly with ML framework implementation, and remodeling is often needed when the framework changes (e.g., adding new features or adopting optimization). For example, Optimus models computation and communication as two separate procedures during one training step; its model needs to be rebuilt when new features are incorporated into ML frameworks, e.g., overlapping backward computation with communication, gradient compression [20].其次，**在不考虑多租户 GPU 集群中的干扰的情况下构建显式性能模型。** 例如，SLAQ和 Optimus [49] 假设 PS 上没有网络拥塞，而 OASiS [18] 和 Optimus [49] 假设可用带宽是一个常数。 但是，我们观察到训练相同模型的速度可能会有很大差异。 Second, explicit performance models are built without considering interference in multi-tenant GPU clusters. 白盒启发式。 已有研究明确地对作业内的训练速度和资源之间的详细关系进行建模，并基于资源速度模型设计调度启发式. 它们有两个限制。 首先，**为了推导出准确的性能模型，建模过程与 ML 框架实现紧密耦合，当框架发生变化（例如，添加新功能或采用优化）时，通常需要重构。** 例如，Optimus 在一个训练步骤中将计算和通信建模为两个独立的过程； 当新功能被纳入 ML 框架时，它的模型需要重建，例如，将反向计算与通信重叠，梯度压缩 [20]。  此外，对 ML 作业之间的干扰进行显式建模也非常困难 [17]，因为每个额外的维度（神经网络结构、并行架构、运行时隔离等）都会增加复杂性。

In contrast to white-box model-based schedulers, we
resort to a black-box approach and design an RL-based resource scheduler: it automatically learns end-to-end resource allocation policy without requiring expert heuristics and without explicitly modeling the ML framework, the workload, and the interference.与基于白盒模型的调度器相比，我们采用黑盒方法并设计了基于 RL 的资源调度器：它自动学习端到端的资源分配策略，无需专家启发式方法，也无需对 ML 框架进行显式建模 、工作量和干扰。

### 2.3 Deep Reinforcement Learning

DRL has been widely used for sequential decision making in an unknown environment, where the agent learns a policy to optimize a cumulative reward by trial-and-error interactions with the environment . In each iteration, the agent observes the current state of the environment and then chooses an action based on the current policy. The environment moves to a new state and reveals the reward, and the policy is updated based on the received reward.

现有的基于 DRL 的资源分配调度器 为离线 DRL 模型训练生成大量跟踪，通常通过为作业构建显式资源性能模型并使用它来估计作业进度 基于分配的资源，在离线仿真环境中。 模型重建（由于 ML 系统更改）和性能模型的不准确性（由于干扰）的需要降低了 DRL 策略学习的质量（见图 9）。 另一种可能性是使用可用的历史轨迹进行离线 DRL 训练。 然而，由于资源分配的决策空间很大（与资源量呈指数关系），历史轨迹通常不包括对 DRL 策略产生的所有可能决策的反馈

Therefore, instead of offline training in a simulated environment, we advocate online RL in the live cluster, exploiting true feedback for resource allocation decisions produced by the DRL agent, to learn a good policy over time. Pure online learning of the policy network model from scratch can result in poor policies at the beginning of learning (see Fig. 10). To avoid poor initial decisions and for the smooth transition from an existing scheduler, we adopt offline supervised learning to bootstrap the DRL policy with the existing scheduling strategy.因此，我们提倡在实时集群中进行在线强化学习，而不是在模拟环境中进行离线训练，利用对 DRL 代理产生的资源分配决策的真实反馈，随着时间的推移学习一个好的策略。 从头开始纯在线学习策略网络模型可能会导致学习开始时的策略不佳（见图 10）。 为了避免糟糕的初始决策和从现有调度器的平滑过渡，我们采用离线监督学习来引导 DRL 策略与现有调度策略。

## 3DL

DL2 的最终目标是在实时DL 集群中找到最佳资源分配策略，并最小化所有并发作业的平均作业完成时间。

### 3.1 DL 集群

在具有多个 GPU 服务器的 DL 集群中，随着时间的推移提交 DL 训练作业。 每个作业都运行一个分布式 ML 框架（例如我们的实验中的 MXNet），以通过重复训练其数据集来学习特定的 DL 模型。 提交作业后，用户，即作业所有者，提供她/他分别运行每个工人和每个 PS 的资源需求，以及要运行的训练时期总数。 例如，一个worker至少 1 个 GPU 和一个 PS 需要许多 CPU 内核。 可以基于专家知识或工作经历来估计实现模型收敛的总训练时期数（例如，模型的损失或准确性的收敛）。

根据资源可用性和训练速度，每个作业可能会从一个时间段到另一个时间段运行不同数量的工作人员和 PS（由调度程序决定）。 对于同步训练，为了保证相同的训练结果（模型）同时改变工人的数量，我们调整每个工人的小批量大小，以便用户指定的作业中的总批量大小仍然保持不变  对于异步训练，每个工人的小批量大小保持不变，而工人数量不同（因为全局批量大小等于每个工人的批量大小）。

### 3.2 DL Scheduler

Our DL-based scheduler, DL2, adopts joint offline and online learning of a policy neural network (NN) for making resource allocation decisions to concurrent jobs in the cluster. An overview of DL2 is given in Fig. 5. 

#### Offline supervised learning. 

For warm-up, we use supervised learning to train the policy NN, to initialize a policy whose performance is as good as the existing scheduler in the DL cluster. A small set of historical job runtime traces collected from the cluster are used for supervised learning, to allow the NN to produce similar decisions as made by the existing scheduler. This step is a must due to the poor performance of applying online RL directly (see Fig. 10).3.2 DL2 调度器我们基于 DL 的调度器 DL2 采用策略神经网络 (NN) 的联合离线和在线学习，为集群中的并发作业做出资源分配决策。 图 5 给出了 DL2 的概述。离线监督学习。 对于热身，我们使用监督学习来训练策略神经网络，以初始化一个策略，其性能与 DL 集群中现有的调度程序一样好。 从集群中收集的一小组历史作业运行时跟踪用于监督学习，以允许 NN 生成与现有调度程序所做的类似决策。 由于直接应用在线 RL 的性能不佳，这一步是必须的（见图 10）。

#### Online reinforcement learning. 

Online RL works in a time-slotted fashion; each time slot is a scheduling interval, e.g., 1 hour. At the beginning of a scheduling interval, the policy NN takes the information of all the concurrent jobs as input state, and produces the numbers of workers and PSs for each job. The concurrent jobs include new jobs arrived in the previous time slot (after previous scheduling) and jobs which were submitted earlier and whose training has not been completed yet. Workers and PSs are placed on physical machines following the placement policy in the cluster, such as load balancing [51]. Jobs’ training progress is observed at the end of each time slot, and used as the reward to improve the policy network

## 4 detailed design

### 4.1 Policy Neural Network

State. The input state to the policy NN is a matrix 

- x 表示在作业中训练的 DL 模型JxL的矩阵，其中 J 是我们正在调度的时间段中并发作业的最大数量的上限，L 是集群中任何时候训练作业类型的最大数量。 我们将 DL 作业训练类似的 DNN 架构视为我们输入中的相同类型。 例如，基于相同的预训练模型的微调作业很常见1，它们可以被视为同一类型。

- d 表示  一个 J 维向量，编码每个作业在集群中运行的time slot数，用于所有作业。 例如，di 是作业 i 运行的time slot数。

- e 一个 J 维向量，编码为每个作业训练的剩余 epoch 数。  ei 是用户指定的总训练时期数之间的差值

- r一个 J 维向量，表示当前时间段内已分配给每个作业的主导资源的价值网络价值量。 例如，ri 是分配给作业 i 的主导资源量（与集群中资源的总容量相比，作业占用的资源类型最多），这是通过在该时间段内通过推理做出的资源分配决策分配给作业 i 的。

- w和u 它们中的每一个都是一个 J 维向量，其中第 i 项是当前时隙中分配给作业 i 的工人 (PS) 的数量。 

  状态不同组件中并发作业的信息根据作业的到达时间进行排序。 输入状态不直接包括集群中可用的资源容量； 我们的调度程序可以处理集群中随时间变化的整体资源容量。

**action** NN 产生策略 π : π(a | s; θ) → [0, 1]，这是动作空间上的概率分布。  a 代表一个动作，θ 是神经网络中的当前参数集。 一个简单的设计是允许每个动作指定分配给所有并发作业的工人/PS 的数量； 这导致了一个指数级大的动作空间，包含所有可能的工人/PS 编号组合。 大的action space会导致大量的训练成本和缓慢的收敛

  为了加快神经网络的学习，我们简化了动作定义，并允许神经网络通过每个推理从以下 3 × J + 1 个动作中输出一个动作：(i) (i, 0)，意味着分配一个工人到工作 i, (ii) (i, 1), 为作业 i 分配一个 PS, (iii) (i, 2), 为作业 i 分配一个工人和一个 PS, (iv) 一个无效动作，表示停止分配资源 当前时隙（因为分配更多资源不一定会导致更高的训练速度 [49]）。 由于每个推理只输出要分配给 J 个作业之一的增量资源，因此我们允许对 NN 进行多次推理，以在每个时隙中生成完整的资源分配决策集：在生成一个动作后，我们更新状态 s， 然后使用神经网络产生另一个动作； 重复推理直到资源被使用完 或产生无效动作。  void 操作表明进一步为作业分配资源不再提高训练速度。 

  虽然我们在每个时间段内为每个作业重新生成工人/PS 编号，但对于在前一个时间段内运行的作业，我们比较新的和以前的编号并执行动态缩放以仅调整部署编号（§5  ）。

  **神经网络NN架构**。 输入状态矩阵 s 连接到一个全连接层，使用 ReLU [48] 函数进行激活。 这一层的神经元数量与状态矩阵的大小成正比。 该层的输出聚合在一个隐藏的全连接层中，然后连接到最终的输出层。 最后的输出层使用 softmax 函数 [25] 作为激活函数。  NN 架构是基于经验训练试验设计的。

### 4.2 Offline Supervised Learning

In offline supervised learning, we use stochastic gradient descent (SGD) [56] to update parameters θ of the policy NN to minimize a loss function, which is the cross entropy of the resource allocation decisions made by the NN and decisions of the existing scheduler in the traces [38]. The NN is repeatedly trained using the trace data, e.g., hundreds of times as in our experiments, such that the policy produced by the NN converges to the policy of the existing scheduler

在离线监督学习中，我们使用随机梯度下降 (SGD) [56] 来更新策略神经网络的参数 θ 以最小化损失函数，这是神经网络做出的资源分配决策和现有决策的交叉熵 跟踪中的调度程序 [38]。  NN 使用跟踪数据重复训练，例如，在我们的实验中进行数百次训练，以便 NN 产生的策略收敛到现有调度程序的策略

### 4.3 在线强化学习奖励。

  DL2 的目标是最小化整个集群中的平均作业完成时间。 工作完成时间将是观察的自然奖励，但只有当工作完成时才能知道，这很可能是数百个时间段之后。 奖励的显着反馈延迟对于在线 RL 来说是不可接受的，因为延迟的奖励几乎没有提供改进早期决策的指导。 我们设计了每时隙奖励以通过工作流程收集更多奖励样本，以便更频繁地更新 RL 模型以加速收敛。 每个时间段的奖励是并发作业在该时间段内训练的归一化时期数的总和，其中在作业 i (ti) 中训练的时期数在为作业训练的总时期数上归一化

基本原理是作业在一个时间段内运行的 epoch 越多，完成所需的时间段就越少，因此最大化累积奖励相当于最小化平均作业完成时间。 规范化是为了防止偏向大型工作。

基于策略梯度的学习。 在在线强化学习中，通过离线监督学习获得的策略神经网络使用 REINFORCE 算法 [62] 进一步训练，以最大化预期累积折扣奖励 E[ 求和∞ t=0 γtrt]，其中 γ ∈ (0, 1) 是 折扣系数。 我们将问题建模为具有长期影响的非线性问题，而不是具有一轮独立反馈的传统线性模型，例如上下文老虎机 [36]，因为不同时隙中的动作是相关的。  REINFORCE 算法通过对 E[ 求和∞ t=0 −γtrt] 执行 SGD 来更新策略网络的参数 θ。 梯度为：

  其中 Q 值 Q(a, s; θ) 表示在给定状态 s 中按照策略 π(·; θ) 采取的行动 a 的“质量”，计算为选择后获得的预期累积折扣奖励 在状态 s 之后的动作 a 跟随 π(·; θ)。 每个 Q 值都可以使用小批量样本计算（根据经验）[56]。 每个样本是一个四元组，(s, a, s, r)，其中 s 是在状态 s 中采取动作 a 后的新状态。 

  请注意，我们的系统与标准 RL 的运行方式不同：我们在每个时隙 t 中使用 NN 进行多次推理（即，产生多个动作）； 每次推理后输入状态都会发生变化； 我们只在时间段中的所有推理完成后观察奖励并更新 NN 一次。 我们可以在一个时隙 t 中获取多个样本，并将每个样本中的奖励设置为在 t 中完成所有推理后观察到的奖励（1）。 

  我们进一步采用了多种技术来稳定在线 RL，加速策略收敛，并提高所获得策略的质量。 

**actor-critic** 我们使用 actor-critic 算法 [46]（如图 6 所示）改进了基于梯度的基本策略强化学习，以加快策略网络的收敛速度。 基本思想是替换方程中的 Q 值。  2 具有优势，Q(a, s; θ) − Vπ(s, θ)，其中 Vπ(s, θ) 是一个价值函数，表示对使用策略 π(a |  s; θ) 从时隙 t 开始的所有时间。 与当前状态下根据 π(a | s; θ) 采取行动的预期回报相比，该优势表明特定行动要好得多。 利用计算策略梯度的优势可确保梯度的方差小得多，从而使策略学习更加稳定。 价值函数由价值网络评估，该网络具有与策略网络相同的神经网络结构，只是其最终输出层是一个没有任何激活函数的线性神经元[46]，并产生价值函数 Vπ(s,  θ)。 输入状态价值网络与策略网络相同。 我们使用时间差异方法[46]训练价值网络。
**job-aware探索**。 为了通过 RL 获得好的策略，我们需要确保充分探索行动空间（即可以充分产生导致良好回报的行动）； 否则，RL 可能会收敛到较差的局部最优策略 [60] [46]。 我们首先采用一种常用的熵探索方法，通过在梯度计算中添加熵正则化项βθH(π(·|s;θ))来更新策略网络[46]。 通过这种方式，策略网络的参数 θ 朝着更高熵的方向更新（意味着探索更多的动作空间）。

在训练期间，由于不了解工作语义，我们观察到大量不必要的或糟糕的探索（例如，为工作分配多个工人但 0 PS）。 为了提高探索效率，我们采用了另一种基于-greedy 方法的技术[55]。 在使用策略网络的每次推理中，我们检查输入状态：如果输入状态属于我们已经识别的不良状态之一，概率为 1 − ，我们应用策略网络产生的资源分配决策，并且 概率，我们丢弃来自策略网络的输出，但采用指定的动作并观察该动作的奖励。

   差输入状态集包括三种情况：（i）存在一个要调度的作业，该作业已分配给多个工人但没有 PS；  (ii) 存在一份工作分配了多个 PS 但没有工人；  (iii) 存在一项工作，其分配的工人 (w) 和 PSs (u) 的分配数量差异太大，即 w/u > 阈值或 u/w > 阈值（在我们的实验中阈值为 10）。 我们对这些输入状态中的每一个手动指定的操作是：（i）为该作业分配一个 PS；  (ii) 为该工作再分配一名工人；  (iii) 为该工作再分配一名 PS 或一名工人，使其工人/PS 数量更加均衡。
**experience replay**。 众所周知，样本之间的相关性会阻止 actor-critic 模型收敛到一个好的策略 [55]。 在我们的在线 RL 中，当前的策略网络确定了以下训练样本，例如，如果策略网络发现分配更多的工人可以提高奖励，那么以下样本序列将由该策略产生的样本序列主导； 这可能会导致糟糕的反馈循环，从而阻止对具有更高奖励的样本进行探索。 

  为了减轻观察到的样本序列中的相关性，我们在 actor-critic 框架中采用了经验回放 [47]。 具体来说，我们维护一个重放缓冲区来存储在最新时隙中收集的样本。 在每个时间段的末尾，我们选择一个小批量的样本，而不是使用在这个时间段收集的所有样本samples from the replay buffer to compute the gradient updates, where the samples could be from multiple previous time slots.

## 5 dynamic scaling

Though node addition and deletion are supported in system design in the literature , existing opensource distributed machine learning frameworks (e.g., TensorFlow [15], MXNet [20], Caffe [5]) do not support dynamic worker/PS adjustment in a running job. To adjust the number of workers/PSs in a job, a simple and general approach is checkpointing (e.g., Optimus [49]): terminate a training job and save global model parameters as a checkpoint image; then restart the job with a new deployment of PSs and workers, and the saved model parameters. Checkpointing and restarting add additional delay to the training process [50]. For example, it takes 1 minute to checkpoint and stop training, and another 5 minutes to completely restore training of a DSSM model [52], due to data re-preprocessing before training starts. The overhead is significant when the frequency of resource scaling is high (e.g., every hour). The other approach is to resize resources without terminating training process. As an example, we improve the MXNet framework [20] to enable dynamic “hot” scaling. 

**Challenges**. In the parameter server architecture, each PS maintains a subset of the parameters in the global model. When the number of PSs changes, the global parameters need to be migrated among the PSs (for load balancing), and workers should be informed in time to send parameter updates to the correct PSs. When the number of workers changes, the new connections between new workers and the PSs should be established. The key challenges are: (1) correctness, i.e., a consistent copy of the global model parameters should be maintained while parameters are moved across the PSs, and workers always send gradients to correct PSs; (2) high performance, i.e., we should ensure that interruption to training is minimized and the PSs are load balanced.
**Scaling Steps.** We add a coordinator module into the MXNet framework, which works with DL2 scheduler to handle joining of new workers or PSs and termination of existing ones. We demonstrate our design using the case of adding a new PS into an existing job. The steps are shown in Fig. 7.

1) registration When a new PS is launched, it registers itself with the coordinator by sending an “INC SERVER” request message. The PS will then receive its ID in the job, the global parameters it is responsible to maintain, and the current list of workers and PSs to establish connections with. After that, the PS starts functioning, awaiting workers’ parameter updates and further instructions from the coordinator (e.g., parameter migration).

*2) Parameter assignment.* Upon receiving a registration request, the coordinator updates its list of workers and PSs, and computes parameter assignment to the new PS. A best-fit algorithm is adopted: move part of the parameters on each existing PS to the new PS, such that all PSs maintain nearly the same number of parameters, while minimizing parameter movement across the PSs. 

  In order to keep a consistent copy of global model parameters when migrating parameters among PSs, we maintain a version counter for parameters. For PSs, the version counter is the number of parameter updates; for workers, the version counter is received from PSs when pulling updated parameters. To decide when PSs should migrate parameters, we calculate a scaling clock based on current version counter and round trip time between the coordinator and PSs/workers.

The coordinator sends new parameter assignment among PSs and the scaling clock to all PSs and workers.

3) Parameter migration. At each PS, when the version counter of parameters reaches the scaling clock received from the coordinator, the PS moves its parameters to the new PS according to the parameter assignment decisions received2. Once parameter migration among all PSs is completed, the coordinator notifies all workers to resume training. 4) Worker update. At each worker, once its version counter equals the scaling clock received from the coordinator, the worker suspends its push/pull operations and awaits notification for completion of parameter migration. Upon notification from the coordinator, the workers update their parameter-PS mapping, establish connections with the new PS, and resume the training process.协调器向所有 PS 和工人发送 PS 之间的新参数分配和缩放时钟。
*3) 参数迁移。* 在每个 PS，当参数的版本计数器达到从协调器接收到的缩放时钟时，PS 根据接收到的参数分配决定将其参数移动到新的 PS。 一旦所有 PS 之间的参数迁移完成，协调器通知所有工人恢复训练。  

*4) 工人更新。* 在每个 worker 上，一旦其版本计数器等于从协调器接收到的缩放时钟，worker 就会暂停其推/拉操作并等待参数迁移完成的通知。 根据协调器的通知，工作人员更新他们的参数-PS 映射，与新的 PS 建立连接，并恢复训练过程。

In case of removing a PS, the scheduler chooses the PS to be removed by keeping the load balanced among the physical machines. The chosen PS sends a removal request to the coordinator. Similar steps as 2)3)4) above are then carried out, except that parameters in the removed PS are moved to other PSs, using the best-fit algorithm.

nator sends the current parameter-PS mapping in the response to the worker’s registration message. It also notifies all PSs the addition of the new worker for building connections. The worker starts operation after training dataset is copied. For worker removal, the scheduler chooses the worker to be removed by keeping the load balanced across physical machines. The coordinator receives a removal request from the worker, and then broadcasts it to all workers and PSs for updating their node lists. The mini-batch size of workers is adjusted so as to keep total batch size the same.

## 6 评估

### 6.1 DL2 Implementation

  We implement DL2 as a custom scheduler on Kubernetes [9]. We run each training job using the MXNet framework [20]. Workers and PSs are running on Docker containers. Training data of jobs are stored in HDFS 2.8 [3]. The scheduler constantly queries cluster resources and job states (e.g., training speeds) and instructs deployment of a new job or resource adjustment in an existing job via Kubernetes API server. Mapping the cluster and job states to a scheduling decision takes less than 3ms. 

  For each new job, DL2 launches its coordinator, workers, and PSs on machines decided by the default placement strategy of the cluster (i.e., load balancing). The coordinator is informed of the workers and PSs in the job via Kubernetes API. When a worker/PS container is launched on a machine, an agent in the container starts execution. It queries the readiness of other containers of the same job via Kubernetes API and starts user-provided training scripts after all other containers are ready. The agent also monitors the training status, e.g., the number of trained steps, accuracy, and training speed.

我们将 DL2 实现为 Kubernetes [9] 上的自定义调度程序。 我们使用 MXNet 框架 [20] 运行每个训练作业。  Worker 和 PS 在 Docker 容器上运行。 作业的训练数据存储在 HDFS 2.8 [3] 中。 调度器不断查询集群资源和作业状态（例如训练速度），并通过 Kubernetes API 服务器指示部署新作业或在现有作业中调整资源。 将集群和作业状态映射到调度决策所需的时间不到 3 毫秒。

 对于每个新作业，DL2 在由集群的默认放置策略（即负载平衡）决定的机器上启动其协调器、工作器和 PS。 协调器通过 Kubernetes API 获知作业中的工作人员和 PS。 当在机器上启动 worker/PS 容器时，容器中的代理开始执行。 它通过 Kubernetes API 查询同一作业的其他容器的准备情况，并在所有其他容器准备就绪后启动用户提供的训练脚本。 代理还监控训练状态，例如训练步数、准确性和训练速度。

### 6.2 Methodology 

**Testbed.** Our testbed includes 13 GPU/CPU servers
connected by a Dell Networking Z9100-ON 100GbE switch. Each server has one Intel E5-1660 v4 CPU, two GTX 1080Ti GPUs, 48GB RAM, one MCX413A-GCAT 50GbE NIC, one 480GB SSD, and one 4TB HDD. Each server runs Ubuntu 14.04 LTS and Docker 17.09-ce [7]. 

**Trace.** We use patterns from a 75-day real-world job trace collected from a large production DL cluster with a few thousands of GPUs and thousands of jobs, to drive



### 6.4Generality

Training completion time variation. To see how DL2 handles practical performance variation (which white-box schedulers may not handle well), we vary the training speeds in each type of jobs to simulate variation in the training completion time of the same type of jobs (the total numbers of epochs to train remain the same). In Fig. 13, the variation indicates how the training speed deviates from the average speed (which can be faster or slower by the respective percentage). We see that Optimus is more sensitive to the variation, as it can be easily stuck in local optimum: its scheduling relies on the convexity of the performance model, but training speed variation often breaks convexity. The average job completion time shown in all simulation figures is in time slots.

### 6.5 Training Design 

SL loss function. We evaluate three common loss functions for supervised learning, i.e., Mean Square, Cross Entropy (the default) and Absolute Difference [13]. We observe similar performance with these loss functions, while adopting Cross Entropy achieves the best performance. This is because Mean Square or Absolute Difference emphasize incorrect or suboptimal output, while only the correct or optimal output contributes to the loss when using Cross Entropy.6.5 训练设计 SL 损失函数。 我们评估了监督学习的三种常见损失函数，即均方、交叉熵（默认值）和绝对差值 [13]。 我们观察到与这些损失函数相似的性能，而采用交叉熵实现了最佳性能。 这是因为均方或绝对差强调不正确或次优的输出，而在使用交叉熵时，只有正确或最佳的输出才会导致损失。

Reward function. We evaluate another reward function with DL2, which sets the reward of each action (that adds some worker/PS to a job) as the normalized number of epochs trained by the job in the time slot. We find that its performance is 29.1% worse. Our default reward function considers all jobs’ progress, enabling the policy network to learn to schedule from a global perspective. 

Actor-critic. To see how the actor-critic algorithm affects training, we remove the value network but only train the policy network. As widely adopted in RL community, we use the exponential moving average of rewards as a baseline in place of the output of the value network in gradient computation of the policy network. As shown in Table 2,

奖励功能。 我们使用 DL2 评估另一个奖励函数，它将每个动作的奖励（向工作添加一些工人/PS）设置为工作在时间段内训练的归一化时期数。 我们发现它的性能差了 29.1%。 我们的默认奖励函数会考虑所有工作的进度，使策略网络能够学习从全局角度进行调度。 

 为了了解 actor-critic 算法如何影响训练，我们移除了价值网络，但只训练了策略网络。 正如 RL 社区广泛采用的那样，我们在策略网络的梯度计算中使用奖励的指数移动平均值作为基准，代替价值网络的输出。 如表 2 所示，



with the value network, the performance is 21.1% better. This is because the average reward is not always an effective baseline; in some cases, even the optimal action leads to a lower reward than the average reward. 

**Job-aware exploration**. We examine how exploration contributes to the performance. From Table 2, we see that without exploration the performance is 28.8% worse, as online RL is stuck in a local optimal policy. 

**Experience replay.** We disable experience replay and see how performance changes. Table 2 shows that the average job completion time is degraded by 39.6%, indicating that experience replay is critical for training. 

**Federated training.** Federated training enables multiple clusters to learn a global DL2 model collaboratively. We study how the number of clusters affects the policy training, by implementing the A3C [46] algorithm, which trains a global policy NN using multiple DL2 schedulers with different training datasets, each for one cluster. Fig. 18 shows that the global performance remains stable when we increase the number of clusters. We have also observed that with more clusters, the policy NN converges much faster due to the use of more training datasets: if there are x clusters, the NN converges almost x times faster. The preliminary result also suggests the possibility of dividing a single massive cluster into loosely coupled sub-clusters where each runs a DL2 scheduler for resource allocation, if scalability issue arises.使用价值网络，性能提高 21.1%。 这是因为平均奖励并不总是有效的基线； 在某些情况下，即使是最佳动作也会导致比平均奖励更低的奖励。 工作意识探索。 我们研究了探索对性能的贡献。 从表 2 中我们可以看到，如果没有探索，性能会差 28.8%，因为在线 RL 陷入了局部最优策略。 体验回放。 我们禁用体验重放并查看性能如何变化。 表 2 显示平均作业完成时间降低了 39.6%，表明经验回放对于训练至关重要。 联合训练。 联合训练使多个集群能够协作学习全局 DL2 模型。 我们通过实施 A3C [46] 算法来研究集群的数量如何影响策略训练，该算法使用具有不同训练数据集的多个 DL2 调度程序训练全局策略神经网络，每个调度程序用于一个集群。 图 18 显示，当我们增加集群数量时，全局性能保持稳定。 我们还观察到，对于更多的集群，由于使用了更多的训练数据集，策略 NN 的收敛速度要快得多：如果有 x 个集群，则 NN 的收敛速度几乎是 x 倍。 初步结果还表明，如果出现可扩展性问题，可以将单个大型集群划分为松散耦合的子集群，每个子集群都运行 DL2 调度程序以进行资源分配。

resource utilization. DL2 starts from offline supervised learning, to ensure basic scheduling performance comparable to the existing cluster scheduler, and then runs in the live DL cluster to make online scheduling decisions, while improving its policy through reinforcement learning using live feedback. Our testbed experiments and large-scale trace-driven simulation verify DL2’s low scaling overhead, generality in various scenarios and outperformance over hand-crafted heuristics.资源利用率。  DL2 从离线监督学习开始，保证基本调度性能与现有集群调度器相媲美，然后在实时 DL 集群中运行以做出在线调度决策，同时通过使用实时反馈的强化学习改进其策略。 . 我们的测试平台实验和大规模跟踪驱动模拟验证了 DL2 的低扩展开销、各种场景中的通用性以及优于手工启发式算法的性能。

