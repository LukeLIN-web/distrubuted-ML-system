A HIERARCHICAL MODEL FOR DEVICE PLACEMENT 2018

摘要

引入了一个层次模型把计算图placement. 特别是在异构环境中. 把 graph operation assign给groups 把这些group分给device.  group和device allocation 是联合学习的.  方法trained with policy gradient. 

## 1 introduction & RELATED WORK

device placement 可以定义为learning 在可用device之间 划分graph的问题.  graph partitioning methods 是CS well-studied subject, 传统的方法可以作为natural baseline.  我们用scotch进行了实验. scotch是一个成熟的graph分割开源库. 但是结果令人失望, 可能因为成本函数的 non-stationarity 非平稳性,(从变量的角度看，**平稳性（stationarity）**意味着条件概率分布要不随时间变化，这样才能用用以前的值来预测未来的值)

操作数大时, RNN 学习cost高, 只能用于小graph少于1000 nodes, 而且需要人把来graph分成group. 

我们提出, 不用人工分组的方法. 用two-level hierarchical network. 第一个模型把graph的操作分组, 第二个模型把分组放在设备上. group一个feed forward NN 前馈网络, graph中读取每个操作和上下文信息, 预测应该分配哪个组. Placer 是一个序列到序列模型, 读取 the embedding of the group(不知道embedding啥意思) 并预测该group的placement.  用RL 对两级网络联合训练. (老师想做的是一级resource allocation, 一级placement. )我们的方法是端到端的. 不需要人工分group. 

我们的model 可以处理非常大的graph, 可以发现 多个设备上的non-trivial (具有一定复杂度，需要一定脑力活动、加工过程才能得到的)placement.  这个方法可以学习环境的特性, 包括硬件中计算和通信的复杂折衷 trade-off. 

## 2 method 

两个子NN 联合训练.  叫做hierarchy层次规划器.  运行时间= fp + bp + update parameter,  因为reward 不可微分, 用策略梯度来training 层次规划器, forward Grouper and recurrent Placer. 

Grouper把操作分组, 所有操作分组完毕, 用每个member operation产生 an embedding for that group. 然后把这个嵌入作为输入传递给Placer.  

placer 计算每个组的placement, 给每个device 分配0个或多个group. 然后把每个操作放置在其组被分配到的设备上来. 

grouper是前馈模型, 后跟一个softmax 层, 输出大小等于组的数量. 

placer是一个序列到序列模型, 具有LSTM. 和content-based attention mechanism to predict placement.

我们首先生成 op embedding, 作为输入传递给grouper .

op embedding 是什么?  三个向量 , 

1. 嵌入op type 信息的向量, 比如matmul,conv2d, sum , 看作语言建模任务, 学习一个size = 20的op 类型embedding, 包括200个最常用的TF操作. 
2.  包含每个op的输出大小和输出数的向量, 输出边 <= 6, 输出大小<= 4个元素, 通过逐个读取op输出插入输出操作shape来填充这个向量, 如果不够用-1填充
3. 包含该op的邻接信息的向量, 用bfs 遍历graph, 传入最多6传出最多6, 然后填充向量, 

为了生成placer的输入, 取每个组并连接3个向量来创建其组嵌入, 

1. 包含组中每个op类型的计数的向量
2. 计算该组中op的输出shape 总数的向量,大小为16
3.  包含group 邻接信息的向量, 向量大小是组数,实验中为256, 如果有边就是1. 

placer的 RNN encoder 每次读一个group embedding,  产生M 个hidden state, M 作为超参数, M = 组的个数, placer的解码器RNN 每个时间step 预测一个设备,设备以 于input group embedding 相同的顺序返回. i.e.,  第一组中的op 放置在第一解码器step 返回的 device上.  每个设备都有可训练embedding, 作为下一个解码器step的输入.

每一步t , 解码器用attention 来关注编码器的状态, 训练时, 解码器从placer的softmax中每一步采样要给设备dt,  为了让激活不陡峭并允许模型探索我们也用了一些方法.  

用放置决策来放置模型, 下一节, 我们描述一种policy gradient 方法来训练层次规划器.

RL :  planner 优化, 根据grouper和placer的决策. 预测设备放置的每个训练步骤d , 把d的reward定义为  Rd = -sqrt(r) planner应让Rd的期望值最大化,  

我们需要优化cost function

可以得到cost function的两个偏导

### 分布式训练 

我们的框架有一个在多个控制器之间共享的PS, 所有控制器用同一组参数, 异步更新policy. 一个控制器和k个工作结点通信. 一个worker只和一个控制器交互.

每个worker 执行控制器给定的placement 并报告. 所有worker都placement好了 , 控制器就会计算梯度. 

## 3 experiments

应用于CV  NLP 的模型. 

baseline 用 启发式和基于RL的和两个更简单的方法比较, 1 无分组直接放置每个操作的前馈模型2  把random grouper 放入一个学习过的placer中.

### measuring reward:

reward时目标tf模型的一个训练step的运行时间的负平方根.  无效的placement  比如 因为内存限制, 就是-10. 

