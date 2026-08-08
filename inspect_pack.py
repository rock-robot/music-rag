with open("packed/train.txt") as f:
    first = f.readline().split()

print("tokens in first block :", len(first))
print("first token           :", first[0])
print("first 10 tokens       :", first[:10])
print("any SEP (55025)?       :", "55025" in first)