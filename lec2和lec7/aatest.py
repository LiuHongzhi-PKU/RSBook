# !/usr/bin/env python
# coding:utf-8
# autohr:wangbin

import numpy as np

a=np.zeros(10)
a[5]=1

b= np.linalg.norm(a)
print(b)