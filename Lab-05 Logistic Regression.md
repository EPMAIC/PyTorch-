

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
'''
Logistic classification -> 모두딥 1 강의 참고

Logistic Regression (binary classification)

m개의 셈플, d차원의 d사이즈의 1d 벡터 데이터셋 X (m,d)를 m개의 0과 1로 이루어진 정답을 구할 수 있도록 조정
정답이 1아니면 0 이므로 확률 P(x=1) = 1 - P(x=0)

weight parameter는 (d x 1)에 들어있다

즉 X (m,d)를 weight paramete에 곱해서 0과 1로 나태내어야 함
X 와 W를 곱한 후에 sigmoid 함수를 사용 0과 1에 근사
sigmoid -무한대는 0 +무한대는 1에 근사 (1/(1 + e^-x))

Hypothesis

    H(X) = 1/(1+e^-x*w) 
    
    |X*W| = (m, d) x (d, 1)
          = (m, 1)
          
    m개의 element를 가진 1d백터 -> 정답과 size가 같다
    따라서 이것을 sigmoid 함수에 넣어도 size는 동일하고
    그리하여 H(x)를 구할 수 있다
    또  H(X) = P(x=1;w) = 1 - P(x=0; w) 이기도 하다
    
Cost

 cost(W) = -1/m(ylog(H(x)) + (1 - y)(log(1 - H(x))))

Weight Update via Gradient Descent 
    
    W := W - a(W')cost(W) -> Gradient Descent 최소화
'''
```


```python
# 계속 똑같이 결과를 재연해주기 위해 Torch Seed 부여
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7f91000ca950>




```python
# Training Data Set
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]  # |x_data| = (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]] # |y_data| = (6, )
```


```python
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```


```python
print('e^1 equals: ', torch.exp(torch.FloatTensor([1]))) # exp -> 자연상수 e
```

    e^1 equals:  tensor([2.7183])



```python
W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
```


```python
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
```


```python
print(hypothesis)
print(hypothesis.shape)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<MulBackward0>)
    torch.Size([6, 1])



```python
# PyTorch에서 sigmoid 함수를 제공하므로 굳이 일일이 대입해줄 필요는 없다

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
```


```python
print(hypothesis)
print(hypothesis.shape)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<SigmoidBackward>)
    torch.Size([6, 1])



```python
# Cost Function

-(y_train[0]*torch.log(hypothesis[0]) + 
    (1-y_train[0]) * torch.log(1-hypothesis[0])) # log(1 - P(x = 1))

# y_train[0]는 0 아니면 1이므로 윗줄이나 아랫줄 중 하나만 살아남음
```




    tensor([0.6931], grad_fn=<NegBackward>)




```python
# 전체 샘플에 구현

losses = -(y_train*torch.log(hypothesis) + 
    (1-y_train) * torch.log(1-hypothesis))
print(losses)
```

    tensor([[0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931]], grad_fn=<NegBackward>)



```python
cost = losses.mean() #평균 취하기 ( -1/m)
print(cost)
```

    tensor(0.6931, grad_fn=<MeanBackward1>)



```python
# binary_cross_entropy 함수 : binary class인 경우에 대해서 cross entropy를 구하는 것 (슈벌 뭐라는겨 크로스 엔트로피?)

F.binary_cross_entropy(hypothesis, y_train)
```




    tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)




```python
# Whole Training Procedure

# modle 초기화
W = torch.zeros((2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# Optimizer 설정
optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1): # d는 그대로 2 m은 훨씬 많아진 경우
    
    # Cost 계산   
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad() # 초기화
    cost.backward() # back propagation
    optimizer.step() # cost 값을 minimize 하는 방향으로 gradient 이용, W b를 업데이트
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch,nb_epochs, cost.item()
        ))
```

    Epoch    0/1000 Cost: 0.693147
    Epoch  100/1000 Cost: 0.134722
    Epoch  200/1000 Cost: 0.080643
    Epoch  300/1000 Cost: 0.057900
    Epoch  400/1000 Cost: 0.045300
    Epoch  500/1000 Cost: 0.037261
    Epoch  600/1000 Cost: 0.031673
    Epoch  700/1000 Cost: 0.027556
    Epoch  800/1000 Cost: 0.024394
    Epoch  900/1000 Cost: 0.021888
    Epoch 1000/1000 Cost: 0.019852



```python
# Evaluation : training set에 대해 훈련을 했으므로 test 또는 validation 필요

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis[:5]) # 5개만 출력
```

    tensor([[2.7648e-04],
            [3.1608e-02],
            [3.8977e-02],
            [9.5622e-01],
            [9.9823e-01]], grad_fn=<SliceBackward>)



```python
prediction = hypothesis >= torch.FloatTensor([0.5]) # H가 0.5보다 크면 TRUE
print(prediction[:5])
```

    tensor([[0],
            [0],
            [0],
            [1],
            [1]], dtype=torch.uint8)



```python
print(prediction[:5])
print(y_train[:5]) # prediction을 정답과 비교
```

    tensor([[0],
            [0],
            [0],
            [1],
            [1]], dtype=torch.uint8)
    tensor([[0.],
            [0.],
            [0.],
            [1.],
            [1.]])



```python
correct_prediction = prediction.float() == y_train # prediction이 ByteTensor 였으므로 float로 변환하여 y_train과 비교
print(correct_prediction [:5])
```

    tensor([[1],
            [1],
            [1],
            [1],
            [1]], dtype=torch.uint8)



```python
# Higher Implementation with Class

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1) # linear layer는 W (m은 알수 없고 d는 알수 있다 8개의 element를 가진 1d 벡터일 것) 와 b 2개를 가지고 있다
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x): # 전체 함수를 의미
        return self.sigmid(self.linear(x))
```


```python
model = BinaryClassifier()
```


```python
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1) # model에 있던 W와 b, self.linear 파라미터가 iterator 형태로 대임(이터레이터?? )

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = model(x_train)
    
    # cost 계산
    cost + F.binary_cross_entropy(hypothesis, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad() 
    cost.backward() 
    optimizer.step() 
    
    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction =hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch,nb_epochs, cost.item(), accuracy * 100
        ))
```
