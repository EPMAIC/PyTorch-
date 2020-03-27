뭔지 모르겠네
-------------

```python
import numpy as np
import torch
```


```python
# 1D Array with Numpy

t = np.array([0., 1., 2., 3., 4., 5., 6.]) #배열의 index는 0부터
print(t)
```

    [0. 1. 2. 3. 4. 5. 6.]



```python
print('Rank of t: ', t.ndim) # dim : array 의 차원 -> t 는 1D 벡터 // 1D Vector 2D Matrix(행렬) 3D Tensor
print('shape of t: ', t.shape) #shape : array의 형태 -> 하나의 차원에 7개의 element
```

    Rank of t:  1
    shape of t:  (7,)



```python
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #Element -1 index -> 마지막에서부터 첫번째
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1]) #slicing [a:b] -> a 부터 b 이전까지
print('t[:2] t[3:]     = ', t[:2], t[3:])
```

    t[0] t[1] t[-1] =  0.0 1.0 6.0
    t[2:5] t[4:-1]  =  [2. 3. 4.] [4. 5.]
    t[:2] t[3:]     =  [0. 1.] [3. 4. 5. 6.]



```python
# 2D Array with Numpy

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
```

    [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]
     [10. 11. 12.]]



```python
print('Rank of t: ', t.ndim) # t 는 2D(2 dimmension, 2차원) Matrix(행렬)
print('shape of t: ', t.shape) # t는 4x3 행렬
```

    Rank of t:  2
    shape of t:  (4, 3)



```python
# 1D Array with PyTorch

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.]) # FloatTenasor 함수 -> Float형 Tensor 
print(t)
```

    tensor([0., 1., 2., 3., 4., 5., 6.])



```python
print('Rank of  t: ', t.dim())
print('shape of t: ', t.shape)
print('size of  t: ', t.size()) # shape
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) 
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1]) 
print('t[:2] t[3:]     = ', t[:2], t[3:])
```

    Rank of  t:  1
    shape of t:  torch.Size([7])
    size of  t:  torch.Size([7])
    t[0] t[1] t[-1] =  tensor(0.) tensor(1.) tensor(6.)
    t[2:5] t[4:-1]  =  tensor([2., 3., 4.]) tensor([4., 5.])
    t[:2] t[3:]     =  tensor([0., 1.]) tensor([3., 4., 5., 6.])



```python
# 2D Array with Pytorch

t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
```

    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])



```python
print(t.dim())
print(t.size())
print(t[:, 1]) # 1번째 차원(행)에서는 모두 ok / 2번째 차원(열)에서는 1 index 만 
print(t[:, 1].size()) #  크기가 4
print(t[:, :-1]) # 행에서는 모두 ok / 열에서는 처음부터 뒤에서 -1 index 전까지만
```

    2
    torch.Size([4, 3])
    tensor([ 2.,  5.,  8., 11.])
    torch.Size([4])
    tensor([[ 1.,  2.],
            [ 4.,  5.],
            [ 7.,  8.],
            [10., 11.]])



```python
# Broadcasting : 자동적으로 size를 맞춰 행연산을 수행하는 기능
```


```python
# Same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)
```

    tensor([[5., 5.]])



```python
# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3]]) # 3 -> [[3, 3]] 같은 크기로 자동 변환하여 연산  
print(m1 + m2)
```

    tensor([[4., 5.]])



```python
# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]]) #행벡터
print(m1)
m2 = torch.FloatTensor([[3], [4]]) #열벡터
print(m2)

print(m1 + m2)
```

    tensor([[1., 2.]])
    tensor([[3.],
            [4.]])
    tensor([[4., 5.],
            [5., 6.]])



```python
# Multiplication vs MatrixMultiplication

m1 = torch.FloatTensor([[1, 2], [3,4]])
m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1: ', m1.shape)
print('Shape of Matrix 2: ', m2.shape)

print(m1.matmul(m2)) # matmul : 행렬곱
print(m1.mul(m2)) # mul : Broadcasting 이 적용된 element wise곱 m2 가 [[1, 1], [2, 2]] 로 자동변환
```

    Shape of Matrix 1:  torch.Size([2, 2])
    Shape of Matrix 2:  torch.Size([2, 1])
    tensor([[ 5.],
            [11.]])
    tensor([[1., 2.],
            [6., 8.]])



```python
# Mean 평균

t = torch.FloatTensor([1, 2])
print(t.mean())
```

    tensor(1.5000)



```python
# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)
```

    Can only calculate the mean of floating types. Got Long instead.



```python
t = torch.FloatTensor([[1, 2], [3,4]])
print(t)
```

    tensor([[1., 2.],
            [3., 4.]])



```python
print(t.mean())       # 모든 element들의 평균
print(t.mean(dim= 0)) # 열 평균을 행 (index 0의 dimension = 첫번째 차원)으로  나열
print(t.mean(dim= 1)) # 행 평균을 열 (index 1의 dimension)으로 나열
print(t.mean(dim= -1))# 차원의 index 0:행 1:열 -> -1 = 열
```

    tensor(2.5000)
    tensor([2., 3.])
    tensor([1.5000, 3.5000])
    tensor([1.5000, 3.5000])



```python
# Sum

t = torch.FloatTensor([[1, 2], [3,4]])
print(t)
```

    tensor([[1., 2.],
            [3., 4.]])



```python
print(t.sum())       # 모든 element들의 합
print(t.sum(dim= 0)) # 열 원소들의 합을 행 (index 0의 dimension = 첫번째 차원)으로  나열
print(t.sum(dim= 1)) # 행 원소들의 합을 (index 1의 dimension)으로 나열
print(t.sum(dim= -1))
```

    tensor(10.)
    tensor([4., 6.])
    tensor([3., 7.])
    tensor([3., 7.])



```python
# Max and Argmax

t = torch.FloatTensor([[1, 2], [3,4]])
print(t)
```

    tensor([[1., 2.],
            [3., 4.]])



```python
print(t.max()) # max : 가장 큰 값을 반환
```

    tensor(4.)



```python
print(t.max(dim=0)) # 가장 큰 행(index 0)과 그 행의 인덱스를 반환
print('max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1]) # 3, 4 의 행 index는 1 이므로 [1,1] 반환    
```

    (tensor([3., 4.]), tensor([1, 1]))
    max:  tensor([3., 4.])
    Argmax:  tensor([1, 1])

