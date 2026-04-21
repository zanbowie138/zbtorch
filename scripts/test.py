import zbtorch as zbt

a = zbt.Tensor([5,4,6], device="cuda")
print(a.device)

b = zbt.Tensor(5)
print(b.device)

device = ("cuda" if zbt.cuda.is_available() else "cpu")
c = zbt.Tensor(5, device)
print(c.device)