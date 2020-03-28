

```python
import torch
import torch.optim as optim
```


```python
'''
데이터 존나 많은데(수백 수천만) 
grdient descent를 하려면 각 데이터 마다 cost를 구해야됨
오래걸리고 하드웨어 적으로 불가능
-> 데이터의 일부만 학습

# MiniBatch Gradient Descent 

전체 데이터를 minibatch로 균일하게 나눠서 학습
각 minibatch의 cost를 구해서 grdient descent
업데이트당 연산량 줄어들고 속도도 빠름
단 잘못된 방향으로 업데이트 할 가능성있음 -> 다소 러프하게 수렴값으로 접근
'''
```


```python
# PyTorch dataset : PyTorch에서 제공하는 모듈

from torch.utils.data import Dataset
# 모듈을 상속해 새로운 클래스 지정, 원하는 Dataset을 지정할 수 있게 됨

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152],[185],[180],[196],[142]]
   
    def __len__(self):  # dataset 의 총 데이터수
        return len(self.x_data)
   
    def __getitem__(self, idx):  # 어떤 index를 받았을 때 그에 상응하는 데이터 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x, y

dataset = CustomDataset()

# 이렇게 dataset을 구현했다면 dataloader 사용 가능

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size = 2, # 각 minibatch의 크기 / 통상적으로 2의 제곱수
    shuffle = True # 프로그램이 dataset의 순서를 학습하지 못하게 Epoch마다 학습되는 순서를 바꿔줌.
)

# 학습 구현

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):# enumerate(dataloader): minibatch 인덱스와 데이터를 받음
        x_train, y_train = samples
        
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch{}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader), 
            cost.item()
        ))
```
