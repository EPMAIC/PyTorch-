

```python
import torch

'''
linear Regression 
-> 학습 data와 가장 잘 맞는 하나의 직선을 찾는 것
training dataset 과 test dataset (학습 이후 model이 잘 작동하는지 판별하기 위한 data) 비교

직선은 y = Wx + b 의 꼴로 나타낼수 있다
W는 weight b는 Bias
'''
```


```python
# 데이터 정의

x_train = torch.FloatTensor([[1], [2], [3]]) # 입력따로
y_train = torch.FloatTensor([[2], [4], [5]]) # 출력따로
```


```python
# Hypothesis 초기화

w = torch.zeros(1, requires_grad=True) # weight 와 bias 0으로 초기화(어떤 입력을 받아도 초기에는 0을 예측하는 것과 같음)
b = torch.zeros(1, requires_grad=True) # requires_grad=True -> PyTorch에게 w 와 b를 학습시킬 것이라고 알려줌
hypothesis = x_train * w + b
```


```python
# MSE

'''
linear regression 학습하려면 우리의 model이 얼마나 정답과 가까운지 알아야 함
그 수치를 cost 또는 loss 라고 하는데
Linear Regression에서는 Mean Squared Error 줄여서 MSE 라는 함수로 loss를 계산한다
단순히 우리의 예측값과 실제 training dataset의 y값 차이를 제곱하여 평균을 구한 것
-> torch.mean 함수로 구현
'''

cost = torch.mean((hypothesis - y_train)**2)
```


```python
# loss를 이용해서 model을 개선
# torch.optim 라이브러리에서 Stochastic Gradient Descent 줄여서 SGD 라는 기법 사용

# optimizer 정의
optimizer = torch.optim.SGD([w, b], lr=0.01) # 학습 시킬 데이터인 W, b를 list로 만들어 넣어주고 적당한 learning weight 도 넣어준다.

optimizer.zero_grad() # gradient를 초기화
cost.backward() # gradient 계산 
optimizer.step() # 계산된 gradient를 방향대로 weight와 bias, w 와 b를 계산한다
```


```python
# Full Training Course
'''
1.데이터 정의
2.Hypothesis 초기화
3.optimizer 정의

4. 반복
[Hypothesis 예측
cost 계산 
optimizer로 학습]
'''
#Full Training Code
x_train = torch.FloatTensor([[1], [2], [3]]) # 입력따로
y_train = torch.FloatTensor([[2], [4], [5]]) # 출력따로

w = torch.zeros(1, requires_grad=True) # hypothesis를 위해 w, b 초기화
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.01) # optimizer 정의

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * w + b # hypothesis 예측
    cost = torch.mean((hypothesis - y_train)**2) # cost 계산
    
    # optimizer로 학습
    optimizer.zero_grad() 
    cost.backward() 
    optimizer.step() 
```


```python
print(w,b)
```

    tensor([1.5002], requires_grad=True) tensor([0.6661], requires_grad=True)

