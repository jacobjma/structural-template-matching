def find(parents, index):
    parent = parents[index]
    while parent != parents[parent]:
        parent = parents[parent]
    parents[index] = parent
    return parent

def merge(ranks, parents, a, b):

    assert(a != b)
    assert(parents[a] == a)
    assert(parents[b] == b)
    if ranks[a] < ranks[b]:
        parents[a] = b
    elif ranks[a] > ranks[b]:
        parents[b] = a
    else:
        parents[b] = a
        ranks[a] += 1
        
def find_and_merge(ranks, parents, a, b):

    pa = find(parents, a)
    pb = find(parents, b)
    if pa != pb:
        merge(ranks, parents, pa, pb)
