## week report 12-19 July

### 7.15

Background knowledge.

1. I have learned the difference between model parallelism, data parallelism, and pipeline parallelism.  I have learned PS architecture: the workers compute gradients and send gradients to PS, PS use optimizer like SGD to update model parameters. 
2.   I have learned synchronous training and asynchronous training.  In synchronous training, the PS doesn't update parameters until it received all worker's gradients. On the contrary, asynchronous training allows the PS sends parameters to one worker before it received all worker's gradients. Asynchronous training is apparently faster but it may cause stale gradients so that it doesn't have high stability and model accuracy. 

DRL . I have read the problems and common application place of DRL. In our paper about harmony, the input is states,  the output is placement decision. The reward is carefully designed normalized speed. 

### 7.16

1. DRL .    I have learned **the specific way to optimize DRL** such as job-aware, actor-critic, and experience replay. **Actor-critic** introduces a baseline function dependent on the state to ensure a much lower variance in the estimation of the policy gradient. **experience replay** uses FIFO buffer to store samples from multiple past intervals instead of the last interval, it could reduce the correlation among the samples so that accelerate converge. **job-aware Exploration** adds an **entropy regularization term** to gradient calculation and **uses omega-greedy** to ensure that the action space is adequately explored. The further learning could read "V Minch, Asynchronous Methods for Deep Reinforcement Learning. 2016" 
2. DL2.  I have read the design about DL2.  I need to learn hierarchical mode carefully. I think that I need follow this paper "A HIERARCHICAL MODEL FOR DEVICE PLACEMENT"

### 7.17

DRL 

1.   I have learned the challenges of using DRL:  Historical traces can not cover a whole huge action space. That is why we design a NN for reward modeling. The input of reward NN is historical tagged traces, the output is reward prediction.
2.   I have read the design of DRL. The reward design seems identical in DL2 and harmony. r= epoch in this interval/ all needed epoch

Problem in placement.    

1. I have learned the reason for co-located interference.  This is because different jobs use the same underlying resources such as CPU caches, disk IO, network IO, and buses(PCIe/QPI)
2. I have learned the difference between bin packing and load balancing, bin packing tries to make full use of the servers, load balancing places one worker or PS on the least loaded machine. 

other

1. I have read harmony's architecture. The point above is also a part of the harmony.  Fig 6 in the paper could clearly show the workflow.
2. Although it doesn't closely related to our topic, since harmony's implement uses HDFS and docker, I take a look at lecture 6.824 about GFS and a Hadoop book. I have practice a TensorFlow demo with a docker container.

### 7.18

Summary

|                     | DL2                                                          | optimus                                                      | harmony                                                      |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| placement           | no                                                           | analytical performance models for jobs. Based on these models, Optimus design a simple yet effective method for dynamically allocating resources. | DRL output placement while take interference into consideration. NN provide reward prediction. |
| resource allocation | dynamically allocating resources, but DL2 doesn't need build accurate performance models, it focus on online dynamic resource allocation with NN and offline supervised warm-up. | Based on the performance model, we design a marginal gain-based resource allocation algorithm | no                                                           |

1.  Optimus designs an efficient heuristic to solve resource allocation optimization problem,  designs a placement scheme based on a theorem and some principle.
2.   resource allocation methods.      I learned a way to find straggler and a parameter assign algorithm to assign parameter blocks. If a parameter block size is very small, assign it to the PS with the least number of updated requests.

### 7.19

 Evaluation.   

1.  How to test scheduler's performance?  The first is comparison with baselines such as DRF or Tetris . The second is **resource adjustment overhead**. The third is **testing scalability** by scheduling a large number of jobs(4000 in Optimus paper) in a cluster with thousands of nodes.
2.   How to examine scheduler's sensitivity?  The first is examining prediction error. The second is **different training mode** . The third is **different job arrival processes** .

other

1. resource allocation methods.   How to **dynamically** allocate resource? Optimus use checkpoint. To reduce checkpoint overhead, we could set a threshold of checkpointing times for each job to limit the restarting frequency. 
2.  In Optimus related work, Azalia et al. [50] use a model-free deep reinforcement learning method to achieve model parallelism that maximizes training speed of a given model in a single machine. Maybe I need to learn why it is yet to be general and efficient. This is because the difference between a single machine and cluster? or model parallelism is different from resource allocation?

