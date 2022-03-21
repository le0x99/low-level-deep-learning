def shape(A): return (1, len(A)) if type(A[0]) != list else (len(A), len(A[0]))
def transpose(A): return list(map(list, zip(*A)))

def dot(A, B):
    if type(A[0]) == type(B[0]) == float:
        assert len(A) == len(B)
        return sum([ A[_]*B[_] for _ in range(len(A)) ])
    m1, n1 = len(A), len(A[0])
    m2, n2 = len(B), len(B[0])
    assert n1 == m2
    if n1 != m2:
        print(shape(A)," o ",shape(B))
    C = [ [ 0 for _ in range(n2) ] for __ in range(m1) ]
    for i in range(m1):  
        for j in range(n2):
            for k in range(m2):
                C[i][j] += A[i][k] * B[k][j]
    return C

def add(A, B):
    return [[A[0][_]+1*B[0][_] for _ in range(len(A[0]))]]
    


