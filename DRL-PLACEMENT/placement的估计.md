placement的估计



不同placment主要反映 通信带宽不同,  反应出来就是 gradient  synchronize的时间不一样, 

可以根据model的size , 通信的方式(PS 还是all reduce)    , 网络带宽( intra or inter, PCIE nvlink or 以太网) 

我要查些什么?

PS 通信量是多少? 

allreduce 通信量是多少? 

具体考虑那个带宽? 是整体的带宽还是bottleneck 带宽? 

总结一个通信估计的公式. 





sync_mode 在DL的trace只有dist_sync一种. 

### 怎么计算model size?

fp16 parameters 是2bytes, gradients是2bytes.

fp32, parameters 4 bytes

gradients 4 bytes

optimizer state : momentum+ variance  8 bytes

那一个参数大概是16bytes. 然后 * 参数个数 是size

但是optimizer state取决于你的方法,不一定是4 bytes

只通信gradients, 那就是4bytes.通信量就是4bytes * 模型参数个数. 



### 网络带宽

We compare NVLink 2.0’s ➁ performance to GPU (PCI-e 3.0 ➀) and CPU interconnects (Intel Xeon Ultra Path Interconnect (UPI) ➂, IBM POWER9 X-Bus ➃), CPU memory (Intel Xeon ➄, IBM POWER9 ➅), and GPU memory (Nvidia V100➆).

PCI 

NVlink2.0 最快, 63GiB 比PCI-e 3.0 要快5-10倍. 

对于 NVlink和 内存, 瓶颈在NVlink. 约

![](https://pic4.zhimg.com/v2-70d4474994fb66c836cc50e286e9a3ab_b.jpg)

P100搭载的NVLink 1.0，每个P100有4个NVLink通道，每个拥有40GB/s的双向带宽，每个P100可以最大达到160GB/s带宽。nvidia官网上 NVLINK1.0

V100搭载的NVLink 2.0，每个V100,NVLink通道达到6个，每个通道达到50G的双向带宽，因而每个V100可以最大达到300GB/s的带宽。也就是nvidia官网上NVLINK2.0

  ```python
      self.inter_bws = [91.875, 233.0, 59.5, 145.875, 120.125, 60.75, 92.125, 10.375] # MB/s   外部带宽
  ​    self.intra_bws = [306.5, 427.75, 63.0, 1082.125, 181.125, 159.625, 65.625, 22.875] # MB/s 内部带宽
  ```



GPU 和GPU 连接用NVlink3.0，600GB/s 。 Nvidia官网可以看到NVLINK3.0. https://www.nvidia.com/en-us/data-center/nvlink/

CPU 和GPU 用 PCIe， 16GB/s

CPU和CPU 用QPI

网卡 NIC 经过交换机到另一个网卡， 10Gb/s

测出来的速度不代表理论上限
