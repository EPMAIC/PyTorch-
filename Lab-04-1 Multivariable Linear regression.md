

```python
import torch
```


```python
'''
Simple Linear Regression -> 하나의 정보로부터 하나의 결론
H(x) = Wx + b

Multivariate Linear Regression -> 여러개의 정보로부터 결론을 도출
'''
# Dataset

x_train = torch.FloatTensor([[73, 80, 75],  # 5명의 학생. 3번의 쪽지시험
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]) 
y_train = torch.FloatTensor([[152], [185], [196], [142]]) # 한번의 기말시험

```


```python
'''
# Hypothesis function

입력변수가 3개라면 w도 3개 / 단순한 H의 정의
-> H(x) = x1_train * w1 + x2_train * w2 + x3_train * w3 + b


matmul()로 한번에 계산 가능 
-> 더간결, x의 길이가 바뀌어도 코드는 그대로, 속도도 빠름
hypothesis = x_train.matmul(w) + b

#Cost function:MSE

cost + torch.mean((hypothesis - y_train) ** 2)

#Gradient Descent

optimizer = torch.optim.SGD([w, b], lr=1e-5) # w라는 하나의 학습 가능한 변수

optimizer.zero_grad() # gradient를 초기화
cost.backward() # cost function을 미분하여 각 변수들의 ggradient 계산 
optimizer.step() # 계산된 gradient를 토대로 Gradient Descent 시행

'''
```


```python
# <Full code>

# Dataset

x_train = torch.FloatTensor([[73, 80, 75],  # 5명의 학생. 3번의 쪽지시험
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]) 
y_train = torch.FloatTensor([[152], [185], [196], [142]]) # 한번의 기말시험

# model 초기화

w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 정의
optimizer = torch.optim.SGD([w, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train.matmul(w) + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    
    # cost로 H(x) 개선
    optimizer.zero_grad() # gradient를 초기화
    cost.backward() # cost function을 미분하여 각 변수들의 ggradient 계산 
    optimizer.step() # 계산된 gradient를 토대로 Gradient Descent 시행
    
    print('Epoch {:4d}/{} hypothesis: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze(), detach(),
    ))
```


```python
# nn.Module

import torch.nn as nn

class MultivariableLinearRegressionModel1(nn.Module): # nn.Module을 상속해서 모델 생성
    def __init__(self):
        super().__init__()
        self.linear = nn.linear(3, 1) # 입력 차원: 3 출력 차원 : 1  알려주고
        
    def forward(self, x): # Forward 함수에서 hypothesis 계산을 어떻게 하는지만 알려주면 된다
        return self.linear(x)
    
hypothesis = model(x_train)
```


```python
# F.mse_loss

import torch.nn.functional as F

cost = F.mse_loss(prediction, y_train) # PyTorch가 제공하는 Cost Function
```


```python
# <Full code>

# Dataset

x_train = torch.FloatTensor([[73, 80, 75],  # 5명의 학생. 3번의 쪽지시험
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]) 
y_train = torch.FloatTensor([[152], [185], [196], [142]]) # 한번의 기말시험

# model 초기화

model = MultivariableLinearRegressionModel1()

# optimizer 정의
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(hypothesis, y_train)
    
    
    # cost로 H(x) 개선
    optimizer.zero_grad() # gradient를 초기화
    cost.backward() # cost function을 미분하여 각 변수들의 ggradient 계산 
    optimizer.step() # 계산된 gradient를 토대로 Gradient Descent 시행
    
    print('Epoch {:4d}/{} hypothesis: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze(), detach(),
    ))
```
