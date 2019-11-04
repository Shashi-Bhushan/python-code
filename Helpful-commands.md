Helpful commands:

- Time it
> `>>> import timeit`  
> `>>> min(timeit.repeat(lambda: len([]) == 0, repeat=100))`  
> 0.13775854044661884  
> `>>> min(timeit.repeat(lambda: [] == [], repeat=100))`  
> 0.0984637276455409  
> `>>> min(timeit.repeat(lambda: not [], repeat=100))`  
> 0.07878462291455435

- Check Assembly Code

> `>>> import dis`  
> `>>> dis.dis(lambda: len([]) == 0)`  
  1           0 LOAD_GLOBAL              0 (len)  
              2 BUILD_LIST               0  
              4 CALL_FUNCTION            1  
              6 LOAD_CONST               1 (0)  
              8 COMPARE_OP               2 (==)  
             10 RETURN_VALUE  

