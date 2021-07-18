## week report 12-19 July

7.15

1. I have learned the difference between model parallelism, data parallelism and pipeline parallelism.  I have learned PS architecture: the workers compute gradients and send gradients to PS, PS use optimizer like SGD to update model parameters.   
2. I have learned the synchronous training and asynchronous training.  In synchronous training, the PS doesn't update parameters until it received  all worker's gradients. On the contrary, asynchronous training allow the PS sends parameter to one worker before it received all worker's gradients. Asynchronous training is apparently faster but it may cause stale gradients so that it doesn't have high stability and model accuracy. 
3. I have read the problems and common application place of DRL. In our paper about harmony, input is state,  output is placement decision. reward is carefully designed normalized speed. 

7.16

1. I have learned the specific way to optimize DRL such as job-aware, actor-critic and experience replay.  The further learning could read  "V Minch, Asynchronous Methods for Deep Reinforcement Learning. 2016" 
2. I have read the design about DL2 .  I need learn hierarchical mode carefully.

7.17

1. I have learned the reason of co-located interference.  This is because different jobs use same underlying resources such as CPU caches, disk IO, network IO, and buses(PCIe/QPI)
2. 

