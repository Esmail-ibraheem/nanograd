if __name__ == "__main__":
    from nanograd.nn.engine import Value

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a + b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    d += 3 * d + (b - a).sigmoid(5)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    print(f'{g.data:.4f}') 
    g.backward()
    print(f'{a.grad:.4f}') 
    print(f'{b.grad:.4f}')  
    print(f'{e.grad:.4f}')  


import nanograd.nn.train