


# N = input()
# print("nihao")
# print("nihao")
# print("nihao")
#
# str_in = input('用空格分隔多个数据:')
# num = [int(n) for n in str_in.split()]

N = 3
tab = [2,0,1]
ans = 1
minV = 0
maxV = 0
ft = 1
bk = 1
if N<2:
    print(0)
if tab[0]==0:
    ft = tab[1] if tab[1]!=0 else 1
if tab[-1]==0:
    bk = tab[-2] if tab[-2] != 0 else 1
for i in range(1,len(tab)-1):
#for i in range(len(tab)-2,0,-1):
    if tab[i]==0:
        if i==1:
            minV = tab[i-1]
            minV = max(tab[i + 1], minV)
        if i>1:
            minV = tab[i + 1]
        maxV = max(tab[i-1],tab[i+1])
        ans = ans * (maxV - minV + 1)

print((ft*ans*bk)%998244353)