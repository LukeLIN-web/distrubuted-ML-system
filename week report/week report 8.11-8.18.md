8.5 -8.11

## Summary:

### Progress:

I summarize some top conference papers in recent years. 

I investigate the selection of networks and update algorithms in deep reinforcement learning in recent years. At present, the best are SAC and PPO. In fact, omega -exploration, entropy regularization, and experience playback are also used in DL2.

#### What are the problems in the past DRL methods?

They took a long time to make a decision or cannot optimize co-located multiple jobs.

Mirhoseini et al. 2017 proposed to solve the device placement problem using a reinforcement learning approach, based on the policy gradient method. Unfortunately, the standard policy gradient method is known to be inefficient, as it performs one gradient update for each data sample. With the vanilla policy gradient method, it took 27 hours over a cluster of 160 workers to find a placement that outperforms an existing heuristic. Such training costs are prohibitive and hardly acceptable by machine learning practitioners. 

Spotlight 2018 can improve the reinforcement learning of the policy gradient, the modeling is faster than Google's solution, and the state transition model is more stable than Google's DRL. But this cannot optimize co-located multiple jobs.

There are also some fancy but unrelated papers on preemptive scheduling.

We could optimize co-located multiple jobs.

#### What is the new heterogeneity placement?

Narayanan et al., 2020 proposed Gavel, a heterogeneity-aware cluster scheduler that can optimize makespan. But they did not propose a new scheduling strategy or performance optimization in this work.

How to use it?

This can be used as a baseline or to allow the device placement network to converge to the specified strategy first. What's more, we can use deep reinforcement learning to allocate resources in the first level, and use this scheduler as the second level to place jobs. It could make decisions faster than Mirhoseini et al. 2017 

#### What are the problems in recent schedulers?

**Tiresias: A GPU Cluster Manager for Distributed Deep Learning** **nsdi** **2019** It does not consider which device the job is suitable for, the scheduling mechanism and the target policy are strongly coupled, and it is difficult to support other more complex strategies. Moreover, it requires the user to submit the number of GPUs, which belongs to non-scale-adaptive scheduling.

We want to decouple target policy from scheduling mechanism.

**Pollux: Co-adaptive Cluster Scheduling for** **Good-put-Optimized Deep Learning osdi2021** We have the same idea: Currently we are also unaware of the dependence of resource allocation and placement, so we need to adjust them at the same time. The weakness is that it did not consider device placement.  

#### What we could learn from past design?

**Gandiva: Introspective Cluster Scheduling for Deep Learning 2018osdi18-xiao** We can refer to its dynamic migration, GPU allocation, and device placement methods.

**An Efficient and Non-Intrusive GPU Scheduling Framework for Deep Learning Training Systems 2020** The input progress information is used to allocate the GPU. The input and output are the same as our design. We can also do non-intrusive scheduling.



Dear Prof. Wu,

I find that 



“为了达到xxx的目的，我先xxx，然后xxx，最后xxx。中间我遇到了xxx问题，所以得到或者没得到xxx结果。我为了解决这个问题，好刻苦的！我读了xxx，问了xxx。书里面说xxx，别人给我的意见是xxx。总结下来，关于这个问题，我猜测原因可能是xxx和xxx，所以我试了一下xxx。发现还是不管用，我觉得原因是xxx。基于此，我下一周想继续试一试xxx。但我不确定，因此想在这里和大家讨论一下。”



  Mirhoseini et al. 2017 took a long time to make a decision (27 hours over a cluster of 160 workers).  Ziyue Luo suggests me reading some new papers from sigcomm, infocom and find a good design.

 I summarize these papers . **Heterogeneity-Aware Cluster Scheduling Policies for DL Workloads osdi2020** can transform many other common cluster scheduling policies to corresponding heterogeneity-aware allocations. I used to think that we can use deep reinforcement learning to allocate resources in the first level and use this scheduler as the second level to place jobs. But it cannot be expressed using a single linear program. Another problem is that every job’s placement decision must be greedy---wants the best performance. In the end, there will be problems when competing for resources. Different device placement will affect communication efficiency. Due to the inter-dependence between resource allocation and device placement, they should be trained jointly with each other. I want to find their careful feedback designs. 

I am studying the way to adjust resources and placement jointly from  Pollux2021.   



