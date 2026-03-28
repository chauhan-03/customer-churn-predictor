arr=(2,3,4,5,6,6,8,5,4,677,5,4,433,34)

print(arr)

dict={}
for num in arr:
    dict[num] = dict.get(num, 0) + 1
print(dict)

# this is a new change