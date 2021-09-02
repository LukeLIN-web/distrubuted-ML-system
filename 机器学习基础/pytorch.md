pytorch

自己学习到的, 可以用来查函数名字

pytorch和tensorflow 区别http://www.xyu.ink/1785.html

随机种子`manual_seed`

### 求导

```python
y.backward()
x.grad  就知道梯度了
x.grad.zero_() #  把0写入梯度,也就是清零.
# 我们一般不会求微分矩阵, 而是每个样本单独计算的偏导数之和
y = x * x 
y.sum().backward() 
```



### network

输出网络参数`named_parameters()`

```python
for name,param in policy_net.net.named_parameters():

​    logger.info(*f*"name: {name}, param: {param.shape}")
```

### 检查点

```python
#比如每隔2个epoch保存一次
    if epoch% 2 == 0：
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint,'cnn_model.pth')#这里后面这个是你想要设置的checkpoint文件的名字，ubuntu上默认保存为zip文件

# load	
    if pm.POLICY_NN_MODEL is not None:
		checkpoint = tr.load(pm.POLICY_NN_MODEL)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint(['epoch'])
```



### optimizer类



### loss类

```
"Mean_Square":
"Cross_Entropy":  Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。
"Absolute_Difference":
```

自定义损失函数  [pytorch系列12 --pytorch自定义损失函数custom loss function_墨流觞的博客-CSDN博客_pytorch 自定义loss](https://blog.csdn.net/dss_dssssd/article/details/84103834)

MSELoss 

```
criterion = nn.MSELoss()
loss = nn.MSELoss()
```

```python
output, loss = net.get_sl_loss(np.stack(inputs), np.vstack(labels))
# stack 叠起来, vstack 
```



检查全连接层的参数:  print(net[2].state_dict())

X = torch.rand(size=(2, 4))

```python
tensor.double() 可能会显示tensor(xxx, dtype =float64)# 64位就是double.
tensor.float() 可能会显示tensor(xxx) # 有小数点就是float
```
