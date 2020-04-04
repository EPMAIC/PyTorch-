

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```


```python
# For reproducibility
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7f749ffffaf0>




```python
x_train = torch.FloatTensor([[1, 2, 1], # |x_train| = (m, 3) m개의 샘플이 각각 3개의 element를 가진 1d 벡터가 된다
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0]) # |y_train| = one_hot 벡터의 인덱스들을 m개 가지고 있음
```


```python
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]]) # |x_test| = (m', 3) element의 수가 일치해야 함(같은 분포로부터 얻어진 데이터)
y_test = torch.LongTensor([2, 2, 2]) # |y_test| = (m', )
```


```python
class SoftmaxClassifierModel(nn.Module): # nn.Moudule 상속
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x) # |x| = (m, 3) => (m, 3)
```


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=0.1) # SGD를 통해 훈련
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산 by cross_entropy
        cost = F.cross_entropy(prediction, y_train) # |y_train| = one_hot 벡터의 인덱스이므로 prediction과 비교

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
```


```python
def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(
         correct_count / len(y_test) * 100, cost.item()
    ))
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 2.257259
    Epoch    1/20 Cost: 1.917658
    Epoch    2/20 Cost: 1.565636
    Epoch    3/20 Cost: 1.378519
    Epoch    4/20 Cost: 1.285198
    Epoch    5/20 Cost: 1.245983
    Epoch    6/20 Cost: 1.222197
    Epoch    7/20 Cost: 1.205287
    Epoch    8/20 Cost: 1.191489
    Epoch    9/20 Cost: 1.179064
    Epoch   10/20 Cost: 1.167576
    Epoch   11/20 Cost: 1.156695
    Epoch   12/20 Cost: 1.146284
    Epoch   13/20 Cost: 1.136250
    Epoch   14/20 Cost: 1.126543
    Epoch   15/20 Cost: 1.117128
    Epoch   16/20 Cost: 1.107980
    Epoch   17/20 Cost: 1.099082
    Epoch   18/20 Cost: 1.090420
    Epoch   19/20 Cost: 1.081981



```python
test(model, optimizer, x_test, y_test)
```

    Accuracy: 0.0% Cost: 1.842805



```python
# train loss는 감소했지만 test loss가 증가한 상황 = Overfitting
```


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e5)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 2.203667
    Epoch    1/20 Cost: 878772.812500
    Epoch    2/20 Cost: 2427601.000000
    Epoch    3/20 Cost: 290101.156250
    Epoch    4/20 Cost: 1766273.000000
    Epoch    5/20 Cost: 1283428.625000
    Epoch    6/20 Cost: 979163.625000
    Epoch    7/20 Cost: 1888147.875000
    Epoch    8/20 Cost: 543226.125000
    Epoch    9/20 Cost: 1120994.875000
    Epoch   10/20 Cost: 1188116.125000
    Epoch   11/20 Cost: 1171351.125000
    Epoch   12/20 Cost: 1622522.875000
    Epoch   13/20 Cost: 738018.187500
    Epoch   14/20 Cost: 849119.875000
    Epoch   15/20 Cost: 872491.125000
    Epoch   16/20 Cost: 1521351.125000
    Epoch   17/20 Cost: 1325647.875000
    Epoch   18/20 Cost: 1085413.625000
    Epoch   19/20 Cost: 638798.250000



```python
# learning rate가 너무 크면 diverage 하면서 발산한다.
```


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-10)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 1.280268
    Epoch    1/20 Cost: 1.280268
    Epoch    2/20 Cost: 1.280268
    Epoch    3/20 Cost: 1.280268
    Epoch    4/20 Cost: 1.280268
    Epoch    5/20 Cost: 1.280268
    Epoch    6/20 Cost: 1.280268
    Epoch    7/20 Cost: 1.280268
    Epoch    8/20 Cost: 1.280268
    Epoch    9/20 Cost: 1.280268
    Epoch   10/20 Cost: 1.280268
    Epoch   11/20 Cost: 1.280268
    Epoch   12/20 Cost: 1.280268
    Epoch   13/20 Cost: 1.280268
    Epoch   14/20 Cost: 1.280268
    Epoch   15/20 Cost: 1.280268
    Epoch   16/20 Cost: 1.280268
    Epoch   17/20 Cost: 1.280268
    Epoch   18/20 Cost: 1.280268
    Epoch   19/20 Cost: 1.280268



```python
# learning rate이 너무 작아 cost가 거의 줄어들지 않았다.
```


```python
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 3.187324
    Epoch    1/20 Cost: 1.334307
    Epoch    2/20 Cost: 1.047911
    Epoch    3/20 Cost: 0.996043
    Epoch    4/20 Cost: 0.985740
    Epoch    5/20 Cost: 0.977224
    Epoch    6/20 Cost: 0.970065
    Epoch    7/20 Cost: 0.963589
    Epoch    8/20 Cost: 0.957562
    Epoch    9/20 Cost: 0.951825
    Epoch   10/20 Cost: 0.946302
    Epoch   11/20 Cost: 0.940942
    Epoch   12/20 Cost: 0.935719
    Epoch   13/20 Cost: 0.930613
    Epoch   14/20 Cost: 0.925613
    Epoch   15/20 Cost: 0.920711
    Epoch   16/20 Cost: 0.915902
    Epoch   17/20 Cost: 0.911182
    Epoch   18/20 Cost: 0.906546
    Epoch   19/20 Cost: 0.901994



```python
'''
Data Preprocessing (전처리)

|y_train| = (m, 2) 인데 1열의 원소들의 값은 크고 2열의 원소들의 값이 작을때
전처리 없이 MSE를 수행하면 학습은 1열에만 집중됨 
2열은 이미 작으므로 1열을 줄이는 것이 더 효율 적
Data Preprocessing은 두 열의 값을 비슷한 범위의 값으로 변환, 동등한 학습 가능

'''

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142] ]) # regression 문제 (class x) 입렵값이 실제값과 가까워지도록 훈련 => MSE loss
```


```python
# standard deviation

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma

print(norm_x_train)
```

    tensor([[-1.0674, -0.3758, -0.8398],
            [ 0.7418,  0.2778,  0.5863],
            [ 0.3799,  0.5229,  0.3486],
            [ 1.0132,  1.0948,  1.1409],
            [-1.0674, -1.5197, -1.2360]])



```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 3개의 차원, element 를 입력 받아 1개의 element를 뱉어주는 linear layer

    def forward(self, x):
        return self.linear(x)
    
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-1)

def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train) # |x_train| = (m, 3)

        # cost 계산
        cost = F.mse_loss(prediction, y_train) # |prediction| = (m, 1)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

train(model, optimizer, norm_x_train, y_train)
```

    Epoch    0/20 Cost: 29729.951172
    Epoch    1/20 Cost: 18889.082031
    Epoch    2/20 Cost: 12048.976562
    Epoch    3/20 Cost: 7699.845215
    Epoch    4/20 Cost: 4924.701660
    Epoch    5/20 Cost: 3151.020508
    Epoch    6/20 Cost: 2016.562988
    Epoch    7/20 Cost: 1290.709106
    Epoch    8/20 Cost: 826.216003
    Epoch    9/20 Cost: 528.952271
    Epoch   10/20 Cost: 338.703400
    Epoch   11/20 Cost: 216.939957
    Epoch   12/20 Cost: 139.006989
    Epoch   13/20 Cost: 89.125130
    Epoch   14/20 Cost: 57.196125
    Epoch   15/20 Cost: 36.757286
    Epoch   16/20 Cost: 23.672049
    Epoch   17/20 Cost: 15.293400
    Epoch   18/20 Cost: 9.927166
    Epoch   19/20 Cost: 6.488914

