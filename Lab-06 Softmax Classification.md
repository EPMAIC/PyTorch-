

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# 항상 똑같은 결과를 보장하기 위한 seed
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7f80f40da8d0>




```python
'''
Discrete Probability Distribution (이산확률분포 - 정수로 딱 떨어지는 확률들)

주사의 확률 분포 PMF 1,2,3,4,5,6 각각에서만 1/6
-> uniform distribution(각 확률이 같음)

가위바위보와 같이 각 확률이 동일하지 않을 수 도 있음
P(주먹|가위) : 가위를 냈을 때 주먹을 낼 확률
'''
```


```python
# softmax : max값을 soft하게 뽑아줌

z= torch.FloatTensor([1,2,3]) # |z|=(3, )
# 기존에는 max를 뽑는다면 argmax max(0, 0, 1)

hypothesis = F.softmax(z, dim=0) # 비율에 따라서 1이 되는 값으로 부드럽게 나타내줌 
print(hypothesis)
```

    tensor([0.0900, 0.2447, 0.6652])



```python
hypothesis.sum() # 역시 모두 합해보면 1이 나옴
```




    tensor(1.)




```python
'''
#Cross Entropy
2개의 확률 분포가 주어졌을 때 확률 분포가 얼마나 비슷한지 나타낼 수 있는 수치
(사실 이해가 잘 안된다 갑자기 뭐라는거야? ㅋㅋ)
아무튼 cross entropy를 최소화 하면 모델의 확률 분포 함수가 원래 확률 분포 P에 근사하게 된다고 한다
그래서 cross entropy를 최소화하는게 목표다
'''
# Cross Entropy Loss (low-level)

z = torch.rand (3, 5, requires_grad=True) # rand함수로 random하게 (3, 5) 짜리 Z를 생성 
hypothesis = F.softmax(z, dim=1) # dim 1에 대해서 softmax를 함
print(hypothesis) # prediction y가 될 것
```

    tensor([[0.1146, 0.1571, 0.2537, 0.2496, 0.2250],
            [0.1871, 0.1989, 0.1878, 0.1718, 0.2545],
            [0.1955, 0.1750, 0.2596, 0.1615, 0.2083]], grad_fn=<SoftmaxBackward>)



```python
y = torch.randint(5, (3,)).long() # 랜덤하게 생성한 정답
print(y) # classes = 5 samples = 3 // 각각의 샘플(행)에 대해서 정답 인덱스를 구한 것
# 위에서 정답은 0.1146 , 0.2545 , 0.1750
```

    tensor([0, 4, 2])



```python
y_one_hot = torch.zeros_like(hypothesis) # |y_one_hot| = (3,5)
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # unsqueeze |y| = (3, ) -> (3,1)
'''
scatter 함수? 하여간 아까처럼 0,4,2 에 1이 찍혀 있는 모습
'''
```




    tensor([[1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 1., 0., 0.]])




```python
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
# (3, 5)*(3, 5) -> (3, 5) // dim 1에 대해 sum -> (3, ) // mean -> scalar
print(cost)
```

    tensor(1.6277, grad_fn=<MeanBackward1>)



```python
# log softmax 함수가 제공 되므로 torch.log(f.softmax()) 할 필요 x
F.log_softmax(z, dim=1)
```




    tensor([[-2.1659, -1.8511, -1.3716, -1.3880, -1.4916],
            [-1.6762, -1.6151, -1.6722, -1.7616, -1.3686],
            [-1.6322, -1.7428, -1.3485, -1.8233, -1.5686]],
           grad_fn=<LogSoftmaxBackward>)




```python
# nll_loss (negative log likelihood _ loss)

#low level
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

```




    tensor(1.6277, grad_fn=<MeanBackward1>)




```python
#high level 뒤에 있는 sum 과 mean 을 생략
F.nll_loss(F.log_softmax(z, dim=1), y)
```




    tensor(1.6277, grad_fn=<NllLossBackward>)




```python
# F.cross_entropy 간소화의 결정체 nll_loss 와 log softmax 를 결합
#단 neural network는 softmax 이전의 값을 필요로 할 수 있음

F.cross_entropy(z, y)
```




    tensor(1.6277, grad_fn=<NllLossBackward>)




```python
#Training with Low-level Cross Entropy Loss

# Dataset
x_train = [[1, 2, 1, 1], # |x_train| = (m, 4)
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_train = [2, 2, 2, 1, 1, 1, 0, 0] # |y_train| = (m, )
# 4차원의 어떤 벡터를 받아 어떤 클래스인지 예측
# One Hot vector로 나타냈을 때 1이 있는 위치의 index값 (정수, discrete 하므로)

x_train = torch.FloatTensor(x_train) # |y_train| = (m)
y_train = torch.LongTensor(y_train)  # y_train이 Long형이어야 F.cross_entropy
```


```python
# 모델 초기화
W = torch.zeros((4,3), requires_grad = True) # smaples = m
b = torch.zeros(1, requires_grad = True)     # classes = 3
                                              # dim = 4
# Optimizer 설정
optimizer = optim.SGD([W,b], lr = 0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    # cost 계산 (1)
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) # matmul
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1),1) # one hot vector
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
    
    # H(x) 개선
    # back propagation을 통해 얻은 gradient로 gradient descent 실행
    # Cross_entropy 함수 minimize하여 실제 확률분포 P에 근사
    optimizer.zero_grad()
    cost.backward()
    optimizer.step() 
    
    # 100회 당 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost : {:6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/1000 Cost : 1.098612
    Epoch  100/1000 Cost : 0.761050
    Epoch  200/1000 Cost : 0.689991
    Epoch  300/1000 Cost : 0.643229
    Epoch  400/1000 Cost : 0.604117
    Epoch  500/1000 Cost : 0.568255
    Epoch  600/1000 Cost : 0.533922
    Epoch  700/1000 Cost : 0.500291
    Epoch  800/1000 Cost : 0.466908
    Epoch  900/1000 Cost : 0.433507
    Epoch 1000/1000 Cost : 0.399963



```python
#Training with F.cross_entropy

# 모델 초기화
W = torch.zeros((4,3), requires_grad = True)
b = torch.zeros(1, requires_grad = True)  

# Optimizer 설정
optimizer = optim.SGD([W,b], lr = 0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    
    # cost 계산 (2)
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train) 
    # scatter가 필요없다 one_hot 벡터를 만들어 주는 과정이 생력되어 간편해짐
    
    # H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100회 당 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost : {:6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```

    Epoch    0/1000 Cost : 1.098612
    Epoch  100/1000 Cost : 0.761050
    Epoch  200/1000 Cost : 0.689991
    Epoch  300/1000 Cost : 0.643229
    Epoch  400/1000 Cost : 0.604117
    Epoch  500/1000 Cost : 0.568255
    Epoch  600/1000 Cost : 0.533922
    Epoch  700/1000 Cost : 0.500291
    Epoch  800/1000 Cost : 0.466908
    Epoch  900/1000 Cost : 0.433507
    Epoch 1000/1000 Cost : 0.399962

