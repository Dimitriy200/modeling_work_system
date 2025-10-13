
mylist = [x*x for x in range(3)]
mygenerator = (x*x for x in range(3))

print(mylist)

# for i in mygenerator:
#     print(i+1)


print()

def gen():
    for i in range(3):
        yield i+2

exp_gen = gen()

print(next(exp_gen))
print(next(exp_gen))