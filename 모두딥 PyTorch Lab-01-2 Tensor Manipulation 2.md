

```python
import numpy as np
import torch
```


```python
# View(Reshape in numpy) : 텐서의 모양을 바꿔줌 ->> 준내 중요함

t = np.array([[[0, 1, 2],[3, 4, 5]],
             [[6, 7, 8],[9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape) # 3 x 2 짜리가 2개 있음
```

    torch.Size([2, 2, 3])



```python
print(ft.view([-1, 3])) # -1 : null 값(모르겠어!) / 앞에는 모르겠고 뒤에 두개의 차원중 두번째 차원은 3개의 Element를 가질래
print(ft.view([-1, 3]).shape) # 결과로 4 x 3 행렬이 나온다. (2, ,2, 3) -> (2 x 2, 3) 
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])
    torch.Size([4, 3])



```python
print(ft.view([-1, 1, 3])) # (2, 2, 3) -> (2 x 2, 1, 3)
print(ft.view([-1, 1, 3]).shape) # 변동이 심한 batcht size를 -1로 두면 나머지 값에 맞춰 element가 12개가 되되록 적절히 조절
# (2, 2, 3) (4, 3) (4, 1, 3) 모두 2 x 2 x 3 = 4 x 3 = 4 x 1 x 3 = 12
```

    tensor([[[ 0.,  1.,  2.]],
    
            [[ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.]],
    
            [[ 9., 10., 11.]]])
    torch.Size([4, 1, 3])



```python
#Squeeze : dimension의 element 개수가 1인 경우에 그 dimmension을 없애준다

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])



```python
print(ft.squeeze()) # (3, 1) -> (3, )
print(ft.squeeze().shape)

# ft.squeeze(dim=0) : 0 dimension에 대해 squeeze 실행. element 가 1개가 아니므로 아무일도 일어나지 않는다
# ft.squeeze(dim=0) : 0 dimension에 대해 squeeze 실행. element 가 1개이므로 ft.squeeze()와 같은 효과
```

    tensor([0., 1., 2.])
    torch.Size([3])



```python
#UnSqueeze : 반대로 내가 원하는 dimension에 1을 넣어준다 (따라서 원하는 dimension을 명시해 줘야 함)

ft = torch.Tensor([0, 1, 2]) # 벡터
print(ft.shape)
```

    torch.Size([3])



```python
print(ft.unsqueeze(0)) # dim=0 에 1을 넣어라 / (3, ) -> (1, 3)
print(ft.unsqueeze(0).shape)
```

    tensor([[0., 1., 2.]])
    torch.Size([1, 3])



```python
print(ft.view(1, -1)) # view로도 쉽게 구현 가능 / 앞 dim 에 1을 넣어라
print(ft.view(1, -1).shape)
```

    tensor([[0., 1., 2.]])
    torch.Size([1, 3])



```python
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])



```python
print(ft.unsqueeze(-1)) # dim = -1 마지막 dim 에 
print(ft.unsqueeze(-1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])



```python
# Type Casting 타입 바꿔주기

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
```

    tensor([1, 2, 3, 4])



```python
print(lt.float()) # .float()
```

    tensor([1., 2., 3., 4.])



```python
bt = torch.ByteTensor([True, False, False, True]) # ByteTensor는 True, False Boolean 값을 저장
print(bt)

'''
bt = (lt == 3 ) 과 같이 조건문 등을 수행 했을 때
bt = (0,0,1,0) 과 같이 ByteTensor 타입으로 텐서가 자동으로 선언된다
'''
```

    tensor([1, 0, 0, 1], dtype=torch.uint8)



```python
print(bt.long()) # True는 1 False는 0
print(bt.float())
```

    tensor([1, 0, 0, 1])
    tensor([1., 0., 0., 1.])



```python
# Concatenate 이어붙이기

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
```


```python
# x 와 y 의 size는 (2,2)
print(torch.cat([x, y], dim=0)) # dimension 0 에서 늘어남 -> (4 x 2) (0 dim 이 증가하도록)
print(torch.cat([x, y], dim=1)) # dimension 1 에서 늘어남 -> (2 x 4) (1 dim이 증가하도록)
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    tensor([[1., 2., 5., 6.],
            [3., 4., 7., 8.]])



```python
# Stacking :Concataion을 좀 더 편리하게 이용하게 해줌

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

# x, y, z 의 크기 = (2, )
```


```python
print(torch.stack([x, y, z])) # 위에서 부터 x, y, z를 쌓는다 (3, 2)로
print(torch.stack([x, y, z], dim=1)) # 쌓이는 dim을 지정을 해주면 그쪽으로 stack 된다 / 1 dim에 붙어서 (2, 3)
```

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    tensor([[1., 2., 3.],
            [4., 5., 6.]])



```python
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
# x, y, z를 dim 0에 unsqueeze 하여 (1,2)로 만들고 dim 0이 증가하도록 이어붙인 것과 같음
```

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])



```python
# Ones and Zeros / device(cpu, gpu 같은)간 차이와 관련해 필요..multiple GPU~

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]]) # size of x = (2,3)
print(x)
```

    tensor([[0., 1., 2.],
            [2., 1., 0.]])



```python
print(torch.ones_like(x))  # 1로만 가득찬 똑같은 size의 tensor
print(torch.zeros_like(x)) # 0으로 가득찬 똑같은 size의 tensor
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])



```python
# In-place Operation

x = torch.FloatTensor([[1, 2], [3, 4]])
```


```python
print(x.mul(2.)) # 일반적인 곱 x 에 2를 곱함
print(x) # 다시 x를 출력하면 원래 x가 나옴
print(x.mul_(2.)) # In-place Operation : _ 언더바가 붙음. 메모리에 새로 선언하지 않으면서 결과 값을 기존의 텐서에 넣음.
print(x) #  대체된 x값이 출력

'''
파이토치 피셜 가비지 콜렉터가 효율적으로 설계되서 In-place Operation 해도 속도의 이점이 크지 않을 수 있다~
'''
```

    tensor([[2., 4.],
            [6., 8.]])
    tensor([[1., 2.],
            [3., 4.]])
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[2., 4.],
            [6., 8.]])

