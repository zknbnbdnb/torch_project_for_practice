def kmp_match(s, p):
    '''KMP 算法主体'''
    i = 0
    j = 0
    next = get_next(p)
    while(i<len(s) and j<len(p)):
        if j == -1 or s[i] == p[j]:
            i += 1
            j += 1
        else:
            j = next[j]
    if j == len(p):
        return i-j
    return -1

def get_next(p):
    next = [-1]*len(p)
    next[1] = 0
    i = 1
    j = 0
    while i<len(p)-1:
        if j == -1 or p[i] == p[j]:
            i += 1
            j += 1
            next[i] = j
        else:
            j = next[j]
    return next

if __name__ == '__main__':
    haystack = "mississippi"
    needle = "issip"

    result = kmp_match(haystack, needle)
    print(result)