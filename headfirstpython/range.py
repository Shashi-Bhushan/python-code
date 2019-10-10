import time

#if __name__ == '__main__':
start_time = time.time()

r = range(100000000)

exists = r[-1] in r

end_time = time.time()

print(f"1 Exists in Range {exists}. Time Taken : {end_time-start_time}")
