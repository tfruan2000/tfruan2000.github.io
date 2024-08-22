# 脚本代码

1. 求补码

```python
def get_twos_complement(num, bits):
    if num >= 0:
        return num
    else:
        return (1 << bits) + num

y1 = -2
complement_y1 = get_twos_complement(y1, 64)  # 计算64位补码
print(complement_y1)
print(bin(complement_y1))
print(hex(complement_y1))
```

2. 求不小于x的二次冪

```python
def get_next_power_of_two(x):
    if x & (x - 1) == 0: # 若 x 已是二次冪(二次表示只有一个是1)
        return x
    return 1 << (x - 1).bit_length()
```
