import os

try:
    with open('.asdfasd') as f:
        pass
except FileNotFoundError:
    print('not found')

print(1)