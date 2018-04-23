# -*- coding:utf-8 -*-
import progressbar
import gmpy2


result_file = './result/1966000.txt'
# load result
with open(result_file) as f_in:
    G = f_in.readlines()

g = [gmpy2.mpz(data.strip('\n')[4:]) for data in G]
g = [abs(num) for num in g]
del G

# print(gmpy2.gcd(g[0], g[1]))

i = 0
N = 1966000
seq = ''


pgbar = progressbar.ProgressBar()
pgbar.start(N)
while i < N:
    if int(str(g[0])[-1]) % 2 != 0:
        seq += '1'
        g[0] = (g[0] + g[1]) // 2
    else:
        seq += '0'
        g[0] >>= 1
    i += 1
    pgbar.update(i)

pgbar.finish()

with open('sequence.txt') as f_in:
    ground_truth = f_in.readline()

print(seq == ground_truth[:N])
