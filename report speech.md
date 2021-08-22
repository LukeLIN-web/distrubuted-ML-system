### report speech 

各位老师好!

我是林炬乙

很高兴能在这里展示我暑期科研的成果.  我科研项目的主题是利用hierarchical reinforcement learning to jointly allocate GPU resources and place jobs. 

首先我们来介绍一下我们的背景. 

Nowadays, leading IT companies operate ML clusters, where deep learning (DL) models are trained for providing various AI-driven services. Efficient resource scheduling is essential for maximal utilization of expensive DL clusters. 

 For example, allocating too many GPUs may result in inefficient resource usage, while allocating too few GPUs may result in long runtimes and unused resources. Such decisions are especially difficult to make in a shared-cluster setting, since optimal choices are dynamic and depend on the cluster load while a job is running.
	Some schedulers require users to manually configure their jobs, which if done improperly, can greatly degrade training performance and resource efficiency. Such decisions are especially difficult to make in a shared-cluster setting, since optimal choices are dynamic and depend on the cluster load while a job is running.
Even though recent elastic schedulers can automatically select an appropriate amount of resources for each job,  the  problem is that the workers and PSs may well be distributed onto different physical servers, when they cannot be completely hosted on one server, or to maximize resource fragment utilization on servers. However, even without over-subscription of resources, co-located ML jobs on the same server may interfere with each other negatively and experience performance unpredictability.
	Google brain uses a recurrent NN policy network to predict the placement of operations in a computational graph. It is based on data flow instead of PS architecture. However, it cannot solve the following problems: 
	Firstly, it still uses the resources (PS/worker) allocated by TensorFlow or job owner, which is less efficient or not general enough.

​	Secondly,  the planner can only optimize the training time for a target model (e.g., a TensorFlow graph) If we meet another model, we need to retrain this planner. Further, the modeling typically does not consider interference in a multiple-type job cluster. On the contrary, we want to find the efficient placement of different types of jobs. It is more generic and efficient. 
​	Pollux uses adaptively co-optimizing inter-dependent factors to improve scheduling performance. But it doesn't take co-located interference into account. We want to co-optimize resource allocation and device placements.
​	Using one NN to produce both resource allocation and placement decisions is challenging, mainly because of the significantly larger action space. That is why we want to use hierarchical RL to schedule.



The another problem is that the workers and PSs may well be distributed onto different physical servers, when they cannot be completely hosted on one server, or to maximize resource fragment utilization on servers. However, even without over-subscription of resources, co-located ML jobs on the same server may interfere with each other negatively and experience performance unpredictability.

Google brain uses  a  recurrent  neural  network  policy network  to  predict  the  placement  of  operations  in  a  computational graph. It is based on data-flow instead of PS architecture. However, it cannot solve the following problems: 

resource   allocation It  still  use  the  resources  (PS/worker)allocated by TensorFlow or job owner, which are less efficient or not general enough

Pollux uses adaptively co-optimizing inter-dependent factors to improve scheduling performance.

当前资源分配的问题: 

Most existing schedulers expect users to specify the number of resources for each job, often leading to inefficient resource use. Some recent schedulers choose job resources for users, but do so without awareness of how DL training can be re-optimized to better utilize the provided resources.

placement方面)

当不能放在一个server的时候, 或者为了最大化服务器利用,   worker和PS 可以分到不同服务器,  

The workers and PSs may well be distributed onto different physical servers, when they cannot be completely hosted on one server, or to maximize resource fragment utilization on servers. However, even without over-subscription of resources, co-located ML jobs on the same server may interfere with each other negatively and experience performance unpredictability.

Hierarchical reinforcement learning

Hierarchical Reinforcement Learning is come up with when the simple RL cannot solve some more realistic and complex problems, or when the agent is unlikely to directly learn effective strategies from the low-level primitive action.

Reinforcement Learning can not solve many problems which have too sparse rewards. It almost impossible for agent to learn good and robust policy by exploring environment randomly. Hierarchical Reinforcement Learning could solve these problems.

In recent years, in deep RL, although DNN has enhanced the generalization ability of value / policy function in the large state space problem, it will still face the same problem when encountering the complex learning task with sparse reward or the Life-long learning task. Naturally, there are some methods of hierarchical DRL to expand hierarchical RL to deep environment.

In this work, we focus on joint PS/worker number decision and placement with some hierarchical reinforcement learning design. 

We use Deep Reinforcement Learning with Model free learning,  Model free learning does not model the real environment. The agent can only perform actions through certain strategies in the real environment, wait for rewards and state migration, and then update the behaviour strategy according to these feedback information. In this way, it iterates repeatedly until the optimal strategy is learned.

进度:

Read previous work and find potential limitations.

design the hierarchical RL and make a theoretical comparison.

now: implement our design and test it in experiment.

That's all. Thank you for listening!





### 陶瓷介绍中文

如今，大多数领先的 IT 公司都运营 ML 集群，在其中训练深度学习模型以提供各种 AI 驱动的服务。 有效的资源调度对于最大限度地利用昂贵的 DL 集群至关重要。
   一些调度器要求用户手动配置他们的作业，如果配置不当，会大大降低训练性能和资源效率。 在共享集群中做出这样的决定特别困难，因为最佳选择是动态的并且取决于集群负载。
   尽管最近的弹性调度器可以为每个作业自动选择适当数量的资源，但问题是工作线程和 PS 很可能分布在不同的物理服务器上，当它们不能完全托管在一台服务器上时，或者最大限度地提高资源碎片利用率 在服务器上。 然而，即使没有过度订阅资源，同一服务器上的协同定位 ML 作业也可能会相互干扰并体验性能不可预测性。
   谷歌大脑使用循环神经网络策略网络来预测运算在计算图中的位置。 它基于数据流而不是PS架构。 但是，它不能解决以下问题：首先，它仍然使用TensorFlow或作业所有者分配的资源（PS/worker），效率较低或不够通用。
    其次，规划器只能优化目标模型（例如，TensorFlow 图）的训练时间，如果遇到另一个模型，则需要重新训练此规划器。 此外，建模通常不考虑多类型工作集群中的干扰。 相反，我们希望找到不同类型工作的有效安置。 它更通用，更高效。
   Pollux 使用自适应协同优化相互依赖的因素来提高调度性能。 但它没有考虑同地干扰。 我们希望共同优化资源分配和设备放置。
   使用一个 NN 来产生资源分配和放置决策具有挑战性，主要是因为动作空间要大得多。 这就是我们要使用分层 RL 进行调度的原因。



Nowadays, most leading IT companies operate ML clusters. Efficient resource scheduling is essential for maximal utilization of expensive DL clusters. 
	Some schedulers require users to manually configure their jobs, which if done improperly, can greatly degrade training performance and resource efficiency. Such decisions are especially difficult to make in a shared-cluster, since optimal choices are dynamic and depend on the cluster load.
	Even though recent elastic schedulers can automatically select an appropriate amount of resources for each job,  the problem is that the workers and PSs may well be distributed onto different physical servers, when they cannot be completely hosted on one server, or to maximize resource utilization on servers. However, even without over-subscription of resources, co-located ML jobs on the same server may interfere with each other negatively.
	Google brain uses a recurrent NN policy network to predict the placement of operations in a computational graph. It is based on data flow instead of PS architecture. However, the planner can only optimize the training time for a target model (e.g., a TensorFlow graph) If we meet another model, we need to retrain this planner. Further, the modeling typically does not consider interference in a multiple-type job cluster. On the contrary, we want to find the efficient placement of different types of jobs. It is more generic and efficient. 
	Pollux uses adaptively co-optimizing inter-dependent factors to improve scheduling performance. But it doesn't take co-located interference into account. We want to co-optimize resource allocation and device placements.

​	Using one NN to produce both resource allocation and placement decisions is challenging, mainly because of the significantly larger action space. That is why we want to use hierarchical RL to schedule.

   We could input the current information of the job and cluster, the first level allocate of resources, the second level of additional input cluster available resources and  the current placement. The final output of the allocated resources and placement.   Then,  we calculate reward for the first and second levels.
   The figure is shown. Some of the inputs are from the cluster API and some are from the job owner. The input from the job owner can be fixed in the experiment. At first we need train two networks separately. It helps convergence.

Now I have read previous work and find potential limitations, have written the motivation and background of my paper.  I have finished the simulator. 

design the hierarchical RL and make a theoretical comparison.

now: implement our design and test it in experiment.

That's all. Thank you for listening!



