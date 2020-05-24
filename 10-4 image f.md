

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow
%matplotlib inline

import torchvision
import torchvision.transforms as transforms
```


```python
# GPU 환경 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manuual_seed_all(777)
```


```python
trans= transforms.Compose([
                              transforms.Resize([64, 128]), # 높이 64, 너비 120
                              transforms.ToTensor()
])

train_path ='data/img'
train_data = torchvision.datasets.ImageFolder(root =train_path, transform = trans)
```


```python
data_loader = DataLoader(dataset=train_data, batch_size = 8, shuffle = True, num_workers=2)
```


```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*13*29, 120),
            nn.ReLU(),
            nn.Linear(120,2)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        print(out.shape) 
        out = self.layer2(out)
        print(out.shape) 
        out = out.view(out.shape[0], -1)
        print(out.shape) 
        out = self.layer3(out)
        return out
```


```python
#testing 
net = CNN().to(device)
test_input = (torch.Tensor(3,3,64,128)).to(device)
test_out = net(test_input)
```

    torch.Size([3, 6, 30, 62])
    torch.Size([3, 16, 13, 29])
    torch.Size([3, 6032])



```python
optimizer = optim.Adam(net.parameters(), lr=0.00005)
loss_func = nn.CrossEntropyLoss().to(device)
```


```python
total_batch = len(data_loader)

epochs = 7
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()
        
        avg_cost += loss / total_batch
        
    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))
print('Learning Finished!')
```

    torch.Size([8, 6, 30, 62])
    torch.Size([8, 16, 13, 29])
    torch.Size([8, 6032])



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-113-cb9dc0d6a8d5> in <module>
         10         optimizer.zero_grad()
         11         out = net(imgs)
    ---> 12         loss = loss_func(out, labels)
         13         loss.backward()
         14         optimizer.step()


    /opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
        487             result = self._slow_forward(*input, **kwargs)
        488         else:
    --> 489             result = self.forward(*input, **kwargs)
        490         for hook in self._forward_hooks.values():
        491             hook_result = hook(self, input, result)


    /opt/conda/lib/python3.6/site-packages/torch/nn/modules/loss.py in forward(self, input, target)
        902     def forward(self, input, target):
        903         return F.cross_entropy(input, target, weight=self.weight,
    --> 904                                ignore_index=self.ignore_index, reduction=self.reduction)
        905 
        906 


    /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction)
       1968     if size_average is not None or reduce is not None:
       1969         reduction = _Reduction.legacy_get_string(size_average, reduce)
    -> 1970     return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
       1971 
       1972 


    /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py in nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)
       1788                          .format(input.size(0), target.size(0)))
       1789     if dim == 2:
    -> 1790         ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
       1791     elif dim == 4:
       1792         ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)


    RuntimeError: Assertion `cur_target >= 0 && cur_target < n_classes' failed.  at /opt/conda/conda-bld/pytorch-cpu_1549626403278/work/aten/src/THNN/generic/ClassNLLCriterion.c:93

