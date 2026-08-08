import anticipation.vocab as v

# dump every uppercase constant the module defines, with its value
for name in sorted(dir(v)):
    if name.isupper():
        print(f"{name:24s} = {getattr(v, name)}")