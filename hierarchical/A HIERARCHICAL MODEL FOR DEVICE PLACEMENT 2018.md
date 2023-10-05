A HIERARCHICAL MODEL FOR DEVICE PLACEMENT 2018

前身是Device Placement Optimization with Reinforcement Learning , 这篇文章拓展成了hierarchy. 加了group.

计算图, data flow和  PS placement的区别是什么? 效果更好?  处理更复杂?  不是一个维度的问题, 计算图的placement还和计算中的通信顺序有关,那是model parallel的方式,你还要决策op的scheduling. 

ps placement简单很多,就是影响 parameter synchronization的通信.

加了group好处是什么?

可以自动分组. 减少operation, 避免梯度爆炸.  我们的设计应不会有梯度爆炸, 因为我们allocation不会过大, 还是一样的大小.多一个异构gpu就复杂挺多,可以先考虑最简单的同构的,他是把一个计算图,也就是具体op的graph做place,和我们的维度不一样,是完全不同的问题

```http
https://q.uiver.app/?q=WzAsMjYsWzEsNywib3AxIl0sWzMsNywib3AyIl0sWzUsNywib3AzIl0sWzMsNiwiZW1iZWRkaW5nMiJdLFszLDksImNvbXB1dGluZ1xcXFxncmFwaCJdLFsxLDYsImVtYmVkZGluZzEiXSxbNSw2LCJlbWJlZGRpbmczIl0sWzYsNywib3AuLi4iXSxbMCw2LCJvcFxccXVhZCBlbWJlZGRpbmciXSxbMSw1LCJncm91cFxccXVhZCBpZCJdLFszLDUsImdyb3VwXFxxdWFkIGlkIl0sWzUsNSwiZ3JvdXBcXHF1YWQgaWQiXSxbMCwzLCJncm91cFxccXVhZCBlbWJlZGRpbmciXSxbMCwyLCJoaWRkZW5cXHF1YWQgc3RhdGUiXSxbMSwzLCJnMSJdLFszLDMsImcyIl0sWzQsMywiZzMiXSxbMSwyLCLlj6MiXSxbMywyLCLlj6MiXSxbNCwyLCLlj6MiXSxbNSwyLCLlj6MiXSxbNSwwLCJkZXZpY2VcXHF1YWQgZm9yXFxcXGdyb3VwMSJdLFswLDEsImF0dGVudGlvbiJdLFswLDAsInNvZnRtYXgiXSxbNiwyLCLlj6MiXSxbNiwwLCJkZXZpY2VcXHF1YWQgZm9yXFxcXGdyb3VwMiJdLFsxLDNdLFs0LDFdLFs0LDBdLFs0LDJdLFswLDVdLFsyLDZdLFs0LDddLFs1LDldLFszLDEwXSxbNiwxMV0sWzEwLDE0LCIiLDEseyJsZXZlbCI6Mn1dLFs5LDE1LCIiLDAseyJsZXZlbCI6Mn1dLFsxMSwxNiwiIiwwLHsibGV2ZWwiOjJ9XSxbMTQsMTddLFsxNywxOF0sWzE1LDE4XSxbMTgsMTldLFsxNiwxOV0sWzE5LDIwXSxbMjAsMjFdLFsyNCwyNV0sWzIwLDI0XV0=
```



解决了什么问题?

把计算图placement, 但是是不是只有一个model , 不能多个model?

核心:

它用的是基于数据流的分布式服务, 和harmony的 PS 不一样.  他是划分op的, 然后会自动加上send和rece算子, 它这个好像是自动分配resource的?  

这个就是模型并行, 好像没有数据并行?  他是自动分配的. 

分层训练,一层FNN 一层seqtoseq with LSTM。 第一个把所有op分组,然后把each group分配到设备上个 

它的问题:

1. 只能一个模型产生了op然后优化, 是非常局限的. 不能多个模型一起placement,他的device placement和 harmony不是一种, 这篇文章是异构的GPU TPU 分配, harmony是co-jobs 的问题.   
2.  

摘要

引入了一个层次模型把计算图placement. 特别是在异构环境中. 把 graph operation assign给groups 把这些group分给device.  group和device allocation 是联合学习的.  方法trained with policy gradient. 

## 1 introduction & RELATED WORK

device placement 可以定义为learning 在可用device之间 划分graph的问题.  graph partitioning methods 是CS well-studied subject, 传统的方法可以作为natural baseline.  我们用scotch进行了实验. scotch是一个成熟的graph分割开源库. 但是结果令人失望, 可能因为成本函数的 non-stationarity 非平稳性,(从变量的角度看，**平稳性（stationarity）**意味着条件概率分布要不随时间变化，这样才能用用以前的值来预测未来的值)

colocRL 用RNN预测 graph中op的位置, 用策略梯度方法优化计算速度. 操作数大时, RNN 学习cost高, 只能用于小graph少于1000 nodes, 而且需要人把来graph分成group. 

我们提出, 不用人工分组的方法. 用two-level hierarchical network. 第一个模型把graph的操作分组(问题 :  这个graph的op 是哪里来的? 是tf自动生成的吗? 是的), 第二个模型把分组放在设备上. group一个feed forward NN 前馈网络, graph中读取每个操作和上下文信息, 预测应该分配哪个组. Placer 是一个序列到序列模型, 读取 the embedding of the group(不知道embedding啥意思) 并预测该group的placement.  用RL 对两级网络联合训练. (老师想做的是一级resource allocation, 一级placement. )我们的方法是端到端的. 不需要人工分group. 

我们的model 可以处理非常大的graph, 可以发现 多个设备上的non-trivial (具有一定复杂度，需要一定脑力活动、加工过程才能得到的)placement.  这个方法可以学习环境的特性, 包括硬件中计算和通信的复杂折衷 trade-off. 

## 2 method 

两个子NN 联合训练.  叫做hierarchy层次规划器.  运行时间= fp + bp + update parameter,  因为reward 不可微分, 用策略梯度来training 层次规划器, forward Grouper and recurrent Placer. 

Grouper把操作分组, 所有操作分组完毕, 用每个member operation产生 an embedding for that group. 然后把这个嵌入作为输入传递给Placer.  

placer 计算每个组的placement, 给每个device 分配0个或多个group. 然后把每个操作放置在其组被分配到的设备上来. 

grouper是前馈模型, 后跟一个softmax 层, 输出大小等于组的数量. 

placer是一个序列到序列模型, 具有LSTM. 和content-based attention mechanism to predict placement.

我们首先生成 op embedding, 作为输入传递给grouper。

op embedding 是什么?  三个向量

1. 嵌入op type 信息的向量, 比如matmul,conv2d, sum , 看作语言建模任务, 学习一个size = 20的op 类型embedding, 包括200个最常用的TF操作. 
2.  包含每个op的输出大小和输出数的向量, 输出边 <= 6, 输出大小<= 4个元素, 通过逐个读取op输出插入输出操作shape来填充这个向量, 如果不够用-1填充.
3. 包含该op的邻接信息的向量, 用bfs 遍历graph, 传入最多6传出最多6, 然后填充向量, 

为了生成placer的输入, 取每个组并连接3个向量来创建其组嵌入, 

1. 包含组中每个op类型的计数的向量
2. 计算该组中op的输出shape 总数的向量,大小为16
3.  包含group 邻接信息的向量, 向量大小是组数,实验中为256, 如果有边就是1. 

placer的 RNN encoder 每次读一个group embedding,  产生M 个hidden state, M 作为超参数, M = 组的个数, placer的解码器RNN 每个时间step 预测一个设备,设备以 于input group embedding 相同的顺序返回. i.e.,  第一组中的op 放置在第一解码器step 返回的 device上.  每个设备都有可训练embedding, 作为下一个解码器step的输入.

每一步t , 解码器用attention 来关注编码器的状态, 训练时, 解码器从placer的softmax中每一步采样要给设备dt,  为了让激活不陡峭并允许模型探索我们也用了一些方法.  

用放置决策来放置模型, 下一节, 我们描述一种policy gradient 方法来训练层次规划器.

### RL学习训练

  planner 优化一个目标模型的训练时间比如一个tf 图(所以它必须要有专门的模型?  不能分配job, 不能多个模型同时训练? 我觉得不行  ), 根据grouper和placer的决策. 预测设备放置的每个训练步骤d , 把d的reward定义为  Rd = -sqrt(r) planner应让Rd的期望值最大化

我们需要优化cost function

可以得到cost function的两个偏导

### 分布式训练 

我们的框架有一个在多个控制器之间共享的PS, 所有控制器用同一组参数, 异步更新policy. 一个控制器和k个工作结点通信. 一个worker只和一个控制器交互. (图可以看2017年的文章)

每个worker 执行控制器给定的placement 并报告. 所有worker都placement好了 , 控制器就会计算梯度. 

## 3 experiments

应用于CV  NLP 的模型. 

baseline 用 启发式和基于RL的和两个更简单的方法比较, 1 无分组直接放置每个操作的前馈模型2  把random grouper 放入一个学习过的placer中.

### measuring reward:

reward时目标tf模型的一个训练step的运行时间的负平方根.  无效的placement  比如 因为内存限制, 就是-10. (这里不会有delay reward吗? 有的, 他是跑好几个才有， 训练好了才上线。 )

运行时间 = 一个step的时间=  fp+ bp + update parameter.

怎么减少测量方差?  每个predicted placement 运行10steps. 放弃前5个, 选后5个的中值.

经验发现 sqrt 比单位数或对数更好. 

改变奖励, 我们也可以优化其他指标, 比如推理速度, 吞吐量和网络拥塞. 

设备,  tf r1.3

### 策略NN 的结构:

group是FNN with a hidden size of 64, placer是序列到序列模型, LSTM hiddent size of 256.  我们用两层LSTM形成 biLSTM 编码器. 用单向 LSTM作为解码器, 

group的softmax输出大小 = group数, group= 256时得到最好效果. placer中unrolled step = 组数,  

placer的softmax输出大小= 可用硬件设备数

### 训练细节:

用Adam 优化器(优点:  更新方向考虑了历史梯度,  对step 长度利用累加的梯度平方值进行修正 ,  信息累加按照指数形式衰减)训练, 固定学习率为0.1

为了鼓励探索, 前500个policy 训练step中加入了 整体分布的noise. 

因为大部分placement都是invalid, 我们只在 500个之后有效的placement才 update policy  .  we prevent 策略收敛到 和无效放置相关的奖励. 

### 和colocRL比较

运行了不同GPU怎么办? 没法比,  we will discuss the relative improvements achieved by our approach.

我们对于最佳启发式改进了 60%, 它改进了19%, 4和8层NMT没有结果可能因为模型不能处理这么多op.

colocRL 假设操作必须colocated,  我们的粒度可以很细, 高度并行,  把展开的LSTM一层 每个step放在multiple GPU,  而colocRL只能 colocated all op in a step.也就是都放一个device上

### 分析

展示了部分placement, 非常复杂, 高度并行化, (感觉有点像超线程), LSTM, attention 和softmax 所有展开step都分布在多个GPU上. 

一开始op均匀分配给所有256个组,  但是 grouper 最终会收敛到只用所有模型中的一小部分组, 说明FNN学会了对计算图进行分区, 把应该放在同一设备上的op  分组在一起.  

输入给placer的组嵌入顺序有关系吗?  没啥关系, <7%速度改变

### 运行开销

对于每个模型,我们训练一个新的策略. (不同的模型都要重新训练,这是缺点  )

计算reward , 运行目标模型5个step, 使用median 运行时间. 

简单起见, 先训练 层次规划器planner, 然后 训练目标模型. 为了提高效率，我们可以使用实际训练步骤的运行时间作为奖励，将层次规划器的训练与目标模型的训练交织在一起。(就是说顺便也训练一下目标模型.)

### Alternative Policy Architectures:

把分层规划器和下面两种比较, 

一 只有FNN , 独立预测,  FNN输出直接就是预测位置而不是输出groups, 

不能为更大的benchmark 找到有效的placement. 

另一个是随机分组的planner.

实现证明没有 分组planner好.

## 4 conclusion

提出层次化, 把计算图的op放到设备上 

利用策略梯度法对planner 参数进行优化.  可以拓展到over 8万操作的计算图. 

端到端, 高度并行.

