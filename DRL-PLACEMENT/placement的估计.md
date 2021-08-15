placement的估计



不同placment主要反映 通信带宽不同,  反应出来就是 gradient  synchronize的时间不一样, 

可以根据model的size , 通信的方式(PS 还是all reduce)    , 网络带宽( intra or inter, PCIE nvlink or 以太网) 

我要查些什么?

PS 通信量是多少? 

allreduce 通信量是多少? 

具体考虑那个带宽? 是整体的带宽还是bottleneck 带宽? 

总结一个通信估计的公式. 

  





