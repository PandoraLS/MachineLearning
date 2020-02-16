open with notepad++, or maybe not Compatible

Q1:"'range' object does not support item assignment"
>>> a=range(10)
>>> a
range(0, 10)
>>> del[a[1]]
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    del[a[1]]
TypeError: 'range' object doesn't support item deletion
A1:
>>> a=list(range(10))
>>> a
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> del[a[1]]
>>> a
[0, 2, 3, 4, 5, 6, 7, 8, 9]
>>>
You can refer to this:
http://blog.csdn.net/u011475210/article/details/78013342

I don't know how to Import Script from a Parent Directory!!!

