## week report 15-20 July

### 7.15

Background knowledge.

1. I have learned the difference between model parallelism, data parallelism, and pipeline parallelism.  I have learned PS architecture: the workers compute gradients and send gradients to PS, PS use optimizer like SGD to update model parameters. 
2.   I have learned synchronous training and asynchronous training.  In synchronous training, the PS doesn't update parameters until it received all worker's gradients. On the contrary, asynchronous training allows the PS sends parameters to one worker before it received all worker's gradients. Asynchronous training is apparently faster but it may cause stale gradients so that it doesn't have high stability and model accuracy. 

DRL . I have read the problems and common application place of DRL. In our paper about harmony, the input is states,  the output is placement decision. The reward is carefully designed normalized speed. 

背景知识。
    1. 我已经了解了模型并行、数据并行和管道并行之间的区别。 我学习了 PS 架构：worker 计算梯度并将梯度发送到 PS，PS 使用像 SGD 这样的优化器来更新模型参数。
      2.我学过同步训练和异步训练。 在同步训练中，PS 在收到所有 worker 的梯度之前不会更新参数。 相反，异步训练允许 PS 在收到所有工人的梯度之前将参数发送给一个工人。 异步训练显然更快，但它可能会导致过时的梯度，因此它没有高稳定性和模型准确性。
       我已经阅读了DRL的问题和常见应用场所。 在我们关于的论文中，输入是状态，输出是放置决策。 奖励是精心设计的标准化速度。

### 7.16

1. DRL .    I have learned **the specific way to optimize DRL** such as job-aware, actor-critic, and experience replay. **Actor-critic** introduces a baseline function dependent on the state to ensure a much lower variance in the estimation of the policy gradient. **experience replay** uses FIFO buffer to store samples from multiple past intervals instead of the last interval, it could reduce the correlation among the samples so that accelerate converge. **job-aware Exploration** adds an **entropy regularization term** to gradient calculation and **uses omega-greedy** to ensure that the action space is adequately explored. The further learning could read "V Minch, Asynchronous Methods for Deep Reinforcement Learning. 2016" 
2. DL2.  I have read the design about DL2. Input state is a matrix, including job type, the time job has run, remaining epochs, allocated resources, allocated workers, allocated PSs.  Output is a policy. Each inference has 3xJ +1 actions. I need to learn hierarchical mode carefully. I think that I need follow this paper "A HIERARCHICAL MODEL FOR DEVICE PLACEMENT"

### 7.17

DRL   

1.   I have learned the challenges of using DRL:  Historical traces can not cover a whole huge action space. That is why we design a NN for reward modeling. The input of reward NN is historical tagged traces, the output is reward prediction.
2.   I have read the design of DRL. The reward design seems identical in DL2 and harmony. r= epoch in this interval/ all needed epoch

The problem in placement.    

I have learned the reason for co-located interference.  This is because different jobs use the same underlying resources such as CPU caches, disk IO, network IO, and buses(PCIe/QPI)

I have learned the difference between bin packing and load balancing, bin packing tries to make full use of the servers, load balancing places one worker or PS on the least loaded machine. 

other

1. I have read harmony's architecture. The point above is also a part of the harmony.  Fig 6 in the paper could clearly show the workflow.
2. Although it doesn't closely related to our topic since harmony's implement uses HDFS and docker, I take a look at lecture 6.824 about GFS and a Hadoop book. I have practice a TensorFlow demo with a docker container.

### 7.18

Summary

|                     | DL2                                                          | optimus                                                      | harmony                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| placement           | no                                                           | analytical performance models for jobs. Based on these models, Optimus design a simple yet effective method for dynamically allocating resources. | DRL output placement while take interference into consideration. NN provide reward prediction. |
| resource allocation | dynamically allocating resources, but DL2 doesn't need build accurate performance models, it focus on online dynamic resource allocation with NN and offline supervised warm-up. | Based on the performance model, we design a marginal gain-based resource allocation algorithm | no                                                           |

1.  Optimus designs an efficient heuristic to solve resource allocation optimization problems,  designs a placement scheme based on a theorem and some principles.
2.   resource allocation methods.      I learned a way to find stragglers and a parameter assign algorithm to assign parameter blocks. If a parameter block size is very small, assign it to the PS with the least number of updated requests.

### 7.19

 Evaluation.   

1.  How to test the scheduler's performance?  The first is a comparison with baselines such as DRF or Tetris. The second is **resource adjustment overhead**. The third is **testing scalability** by scheduling a large number of jobs(4000 in Optimus paper) in a cluster with thousands of nodes.
2.   How to examine scheduler's sensitivity?  The first is examining prediction error. The second is **different training mode**. The third is **different job arrival processes**.

other

1. resource allocation methods.   How to **dynamically** allocate resources? Optimus uses checkpoint. To reduce checkpoint overhead, we could set a threshold of checkpointing times for each job to limit the restarting frequency. 
2.  In Optimus-related work, Azalia et al. [50] use a model-free deep reinforcement learning method to achieve model parallelism that maximizes the training speed of a given model in a single machine. Maybe I need to learn why it is yet to be general and efficient. This is because the difference between a single machine and cluster? or model parallelism is different from resource allocation?

7.20

1. read some abstracts from MLSys 2021. "Value Function-Based Performance Optimization of Deep Learning Workloads" seems related to our work but it focuses on underlying languages like Halide and NN compiler like TVM. It isn't related to distribution system performance optimization.
2. DL2 needs submitted job types as features like vgg, resnet. Will it affect the generality? What if the job owner submits another job type? Or we don't submit the job type.(其实也是可以的, 实验中有做, 但是我当时没有仔细看. )
3. I read "A hierarchical model for device placement", which solves the placement problem with the hierarchical model. Group is FNN which reads in information about each operation and its context within the graph, to predict the group to which that operation should be assigned placer is a sequence-to-sequence model that reads in the embedding of the group and predicts the device placement for that group. How to use DRL to solve this problem?    In DL2,  DRL could output resource allocation. In harmony, DRL could output new job placement.

Summary

|        | DL2                                                          | hierarchical model                              |                           harmony                            |
| ------ | ------------------------------------------------------------ | :---------------------------------------------- | :----------------------------------------------------------: |
| input  | job type,time has run, remaining epoch, allocated resources, # of workers/PS | all operation(grouper)  group embedding(placer) | job type,resource demand, # of workers/PS ,available resources on the servers, placement of existing workers and PSs |
| output | allocation one worker/PS of new arrival job                  | placing one group on server.                    | action: placing one worker/PS of new arrival job n on server m |



21晚meeting

我的想法: 

因为层次, 单个模型,  没有考虑colocated,  分组 , group op

DL2 有resource allocation 没有placement. 

harmony , 有placement , 没有resource allocation,NN reward 预测. 

下一次一定要做个ppt ,展示自己很短. 

这周做了啥. 

我花了太多时间看DRL 各种优化方法, 看的比做的多太多,这样不好, 应该看多少,就做多少, 这样学习扎实, 就和上课一样.  
