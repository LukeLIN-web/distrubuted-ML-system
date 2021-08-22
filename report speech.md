report speech 













中文

各位老师好!

我是林炬乙

很高兴能在这里展示我暑期科研的成果.  我科研项目的主题是利用hierarchical reinforcement learning to jointly allocate GPU resources and place jobs. 

首先我们来介绍一下我们的背景. 

Nowadays leading IT companies operate ML clusters, where deep learning (DL) models are trained for providing various AI-driven services. Efficient resource scheduling is essential for maximal utilization of expensive DL clusters. 

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









进度:

Read previous work and find potential limitations.

design the hierarchical RL and make a theoretical comparison.

now: implement our design and test it in experiment.



That's all. Thank you for listening!