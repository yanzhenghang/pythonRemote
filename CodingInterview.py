def Find(target, array):
    # write code here
    Row = len(array)
    Col = len(array[0])
    i = Row - 1
    j = 0
    while i>=0 and j < Col:
        if target == array[i][j]:
            return 1
        elif target < array[i][j]:
            i = i -1
        elif target > array[i][j]:
            j = j + 1
    return 0


print(Find(target=1,array = [[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]))


class Solution:
    # 139. Word Break
    def wordBreak(self, s:str, wordDict:list)->bool: #wordBreak(self, s: str, wordDict: list[str]) -> bool:

        if len(wordDict) <= 0:
            return False

        DP = (len(s) + 1) * [False]
        DP[0] = True

        for i in range(1, len(s)+1, 1):
            for j in range(0, i, 1):
                if DP[j] and (s[j:i] in wordDict):
                    DP[i] = True
                    break

        return DP[len(s)]

    #14. Longest Common Prefix
    def longestCommonPrefix(self, strs: list) -> str:#longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) <= 0:
            return ""
        elif len(strs)==1:
            return len(strs[0])
        cnt = 0
        Finished = False
        tmp = strs[0]
        valMin = 0x7FFFFFFF
        for tt in strs:
            valMin = min(valMin, len(tt))

        while 1:
            for t in strs[1:]:
                if cnt >= valMin:
                    Finished = True
                    break
                if tmp[cnt] != t[cnt]:
                    Finished = True
                    break
            if Finished :
                break
            cnt = cnt + 1
        return strs[0][0:cnt]

    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        N = len(s)
        if N == 0:
            return True
        stack = []
        for i in range(0,N):
            if s[i]=='(':
                stack.append(')')
            elif s[i]=='[':
                stack.append(']')
            elif s[i]=='{':
                stack.append('}')
            elif len(stack)==0 or stack.pop()!=s[i]:
                return False
        if len(stack)==0:
            return True
        else:
            return False

#139. Word Break
#动态规划
print( Solution().wordBreak("leetcode",["leet", "code"]) )
print( Solution().wordBreak("applepenapple",["apple", "pen"]) )
print( Solution().wordBreak("catsandog",["cats", "dog", "sand", "and", "cat"]) )

# 14. Longest Common Prefix
print( Solution().longestCommonPrefix(["flower","flow","flight"]) )
print( Solution().longestCommonPrefix(["a"]) )

# 20. Valid Parentheses
print( Solution().isValid("()") )

#190407 腾讯笔试2题  一般情况正确
def costnum(n, l):
    costsum = abs(l[0])
    for i in range(1,len(l)):
        l[i] += l[i-1]
        costsum += abs(l[i])
    return costsum
n = 3
l = [3,-4,1]
# n = int(input())
#l = [int(i) for i in input().split()]
print(costnum(n, l))