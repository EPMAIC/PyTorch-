

```python
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
```


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', # root : MNIST 데이터의 위치
                          train=True, #  train : True = train set / False = test set
                          transform=transforms.ToTensor(), # transform : 어떤 transform을 적용해서 불러올 것인지 / ToTensor는 이미지의 H,W,C값을 파이토치의 C,H,W 순서와 값에 맞게 변환
                          download=True) # download : True 일시 root에 MNIST 데이터가 존재하지 않으면 다운 받음

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, # dataset : 어떤 data set을 로드 할 건지
                                          batch_size=batch_size, # batch_size : 몇개 씩 batch 할 것인지
                                          shuffle=True, # shuffle : True 일시 순서를 무작위로 섞음
                                          drop_last=True) # drop_last : True 일시 숫자가 맞지 않게 남는 데이터를 사용하지 않음

# MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True).to(device)
# linear layer의 입력은 784 (MNIST가 784개의 데이터 이므로) output은 10 (레이블이 0~9)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device) # 28 x 28을 view를 이용하여 784로 변환
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
```

    Epoch: 0001 cost = 0.535468459
    Epoch: 0002 cost = 0.359274179
    Epoch: 0003 cost = 0.331187516
    Epoch: 0004 cost = 0.316578060
    Epoch: 0005 cost = 0.307158172
    Epoch: 0006 cost = 0.300180733
    Epoch: 0007 cost = 0.295130193
    Epoch: 0008 cost = 0.290851504
    Epoch: 0009 cost = 0.287417084
    Epoch: 0010 cost = 0.284379601
    Epoch: 0011 cost = 0.281825185
    Epoch: 0012 cost = 0.279800713
    Epoch: 0013 cost = 0.277809024
    Epoch: 0014 cost = 0.276154339
    Epoch: 0015 cost = 0.274440855
    Learning finished



```python
# Test the model using test sets
with torch.no_grad(): # no_grad = gradient 계산 안할 것 test 이므로
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
```

    Accuracy: 0.8863000273704529
    Label:  8
    Prediction:  8





