# np.cross的含义
由两个向量生成第三个向量
右手四指从 plane_x 转向 plane_z
大拇指方向就是 plane_y
```python
a = [ax, ay, az]
b = [bx, by, bz]

a × b = [
    ay*bz - az*by,
    az*bx - ax*bz,
    ax*by - ay*bx
]
结果向量长度
|a × b| = |a| * |b| * sin(theta)
```

# 根据一个向量如何计算另一个垂直向量
```python
    # 这个是重点,用plane_x减去plane_x在plane_z上的投影，两个向量相减就会得到垂直于plane_z的向量
    plane_x = plane_x - float(plane_x @ plane_z) * plane_z
```

# 通过平面ax+by_cz+d=0,怎么计算法向量


# 通过平面ax+by+cz+d = 0,怎么计算平面的俯仰角？


# 通过平面法向量怎么计算平面的俯仰角?