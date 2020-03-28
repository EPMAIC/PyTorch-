

```python
import torch

x_train = torch.FloatTensor([[1], [2], [3]]) # 입력
y_train = torch.FloatTensor([[1], [2], [3]]) # 출력
# 3쌍의 dataset 각각 입출력이 동일
```


```python
# Hypothesis in Linear Regression

w = torch.zeros(1, requires_grad=True) # w는 하나의 matrix / x라는 input 벡터에 곱해진 후 b라는 Bias벡터에 더해져 예측값 H(x)계산
b = torch.zeros(1, requires_grad=True) # requires_grad=True -> PyTorch에게 w 와 b를 학습시킬 것이라고 알려줌
hypothesis = x_train * w + b
```


```python
# Simpler Hypothesis Function (No Bias)

w = torch.zeros(1, requires_grad=True)
hypothesis = x_train * w
```


```python
'''
입력과 출력이 동일한 dataset이 주어졌으므로 최적의 Hypothesis Function은 H(x)=x
즉 w=1 일때 dataset에 있는 모든 데이터의 정확한 값 예측
반대로 w=1 이 아닐때 학습의 목표는 w를 1로 급접 시키는 것
w 가 1에 가까울수록 더 정확한 모델이 되는 것
'''

'''
Cost functtion : Intuition
모델 예측값이 실제 데이터와 얼마나 다른지 나타내는 값
잘 학습된 모델일수록 cost가 작다
예시에서는 w=1 일때 cost=0 이므로 (1,0)이 꼭지점이 되는 이차함수의 꼴
(x축이 w y축이 Cost)
'''
'''
Cost functtion : MSE
Linear Regrssion 에서 쓰이는 cost funciton은 Mean Squad Error 줄여서 MSE
예측값과 실제값의 차이를 제곱한 평균
torch.mean 함수로 구현 
'''
'''
Gradient Descent
목표 : Cost functtion 최소화
기울기가 음수일때 w커지고 양수일때 작아져야
기울기가 가파를수록 cost가 큰 것이므로 w를 크게 조정
기울기가 평평하면 cost가 0에 가까운 것이니 w를 조금 바꿔야함
즉 w : w-a델타w(Gradient에 일정 상수 a를 곱한 값을 뺀다)

gradient = 2 * torch.mean((w * x_train - y_train) * x_train) # mean 함수로 dataset 전체의 gradient를 구한다
lr = 0.1 # 상수를 learning weight 라고 부르며 통상 lr로 줄여 쓴다
w -= lr * gradient # 정의한 상수대로 w를 업데이트
'''
hypothesis = x_train * w
```


```python
# dataset
x_train = torch.FloatTensor([[1], [2], [3]]) # 입력
y_train = torch.FloatTensor([[1], [2], [3]]) # 출력

# model 초기화
w = torch.zeros(1, requires_grad=True)

# Learning rate 설정
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * w
    
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((w * x_train - y_train) * x_train)
    
    print('Epoch {:4d}/{} w: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, w.item(), cost.item()
    ))
    
    # cost gradient로 H(x) 개선
    w -= lr * gradient
```

```python
# torch.optim 으로도 gradient descent 가능
# 시작할 때 Optimizer 정의 -> 학습 가능한 변수와 lr을 알아야 함

optimizer = torch.optim.SGD([w], lr=0.15) # w라는 하나의 학습 가능한 변수

optimizer.zero_grad() # gradient를 초기화
cost.backward() # cost function을 미분하여 각 변수들의 ggradient 계산 
optimizer.step() # 계산된 gradient를 토대로 Gradient Descent 시행
```


```python
# dataset
x_train = torch.FloatTensor([[1], [2], [3]]) # 입력
y_train = torch.FloatTensor([[1], [2], [3]]) # 출력

# model 초기화
w = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = torch.optim.SGD([w], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * w
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    print('Epoch {:4d}/{} w: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, w.item(), cost.item()
    ))
    
    # cost로 H(x) 개선
    optimizer.zero_grad() # gradient를 초기화
    cost.backward() # cost function을 미분하여 각 변수들의 ggradient 계산 
    optimizer.step() # 계산된 gradient를 토대로 Gradient Descent 시행
```

    Epoch    0/10 w: 0.000, Cost: 4.666667
    Epoch    1/10 w: 1.400, Cost: 0.746667
    Epoch    2/10 w: 0.840, Cost: 0.119467
    Epoch    3/10 w: 1.064, Cost: 0.019115
    Epoch    4/10 w: 0.974, Cost: 0.003058
    Epoch    5/10 w: 1.010, Cost: 0.000489
    Epoch    6/10 w: 0.996, Cost: 0.000078
    Epoch    7/10 w: 1.002, Cost: 0.000013
    Epoch    8/10 w: 0.999, Cost: 0.000002
    Epoch    9/10 w: 1.000, Cost: 0.000000
    Epoch   10/10 w: 1.000, Cost: 0.000000

