# -*- coding: utf-8 -*-
"""
The inference of rational approximation.
Topsec Cup 2018
Author: hatuw
Mail: jiaxi_wu@ieee.org
"""

import gmpy2
import progressbar
import numpy as np
from multiprocessing import Pool
# import multiprocessing

sequence_file = './sequence.txt'
PROCESSES_NUM = 6

# # load the sequence data
with open(sequence_file, 'r') as sq_in:
    # for test: only load the top 1000 numbers
    # sequence = sq_in.readline()[:1000]
    sequence = sq_in.readline()[:10000]

test_seq = "0011100101"
# test_seq = "0001011001100"
sequence = test_seq

len_sequence = len(sequence)


# define phi(x)
phi = lambda x: np.max(np.abs(x))


# cal the bound
def cal_bound(*args):
    assert len(args) == 2, "the length of input must eq 2"
    ar1 = args[0]
    ar2 = args[1]
    temp_bound1 = (ar1[0] - ar1[1]) // (ar2[0] - ar2[1])
    temp_bound2 = -(ar1[0] + ar1[1]) // (ar2[0] + ar2[1])
    return np.sort([temp_bound1, temp_bound2])


def multipcs(*args):
    # Use multiprocess to accelerate and prevent memory error
    lowwer_bound = args[2]
    upper_bound = args[3]
    is_even = args[-1]

    # use multi processing
    pool = Pool(processes=PROCESSES_NUM)
    results = []
    pad_size = 1000
    tmp_low = lowwer_bound
    tmp_low = lowwer_bound + int(is_even)
    is_even = False
    tmp_up = lowwer_bound + pad_size
    while tmp_up <= upper_bound:
        ret = pool.apply_async(find_d,
            args=(args[0], args[1], tmp_low, tmp_up, is_even))
        results.append(ret)
        tmp_low += pad_size
        tmp_up += pad_size
        if tmp_up > upper_bound:
            tmp_up = upper_bound
        if tmp_low >= tmp_up:
            break

    pool.close()
    pool.join()

    return(np.min([x.get() for x in results]))


# define minimize(list)
# @check_bounds
def find_d(*args):
    """
    d: d is among the odd integers immediately
    less than(<) or greater than(>)
    (f1-f2)/(g1-g2) and -(f1,f2)/(g1+g2)
    """

    lowwer_bound = args[2] + int(args[-1])
    upper_bound = args[3]

    d_range = np.arange(lowwer_bound, upper_bound, 2)[:, np.newaxis]
    tmp_phi = np.max(np.abs(args[0] + d_range*args[1]), 1)
    d = d_range[np.where(tmp_phi == np.min(tmp_phi))]
    return d[0]


def controller(*args):
    # assert len(args) == 4, "It need tow args(f, g) to find d"
    diff = args[3] - args[2]
    is_even = int(str(args[2])[-1]) % 2 == 0
    if diff < 2:
        return args[2]+1 if is_even else args[2]
    elif diff > 10000:
        return multipcs(*args, is_even)
    else:
        return find_d(*args, is_even)


# validation
def test(g, sequence):
    pad = 0
    for index, val in enumerate(sequence):
        # pad += int(val) * 2**index
        if int(val) == 0:
            continue
        pad += int(val) << index

    if len_sequence < 10000:
        print("sequence: " + str(sequence))

    # gt = g[1]*pad % 2**(len(sequence))  # grount truth
    # rst = g[0] % (2**len(sequence))     # result
    gt = g[1]*pad % (2 << (len_sequence-1))  # grount truth
    rst = g[0] % (2 << (len_sequence-1))     # result

    if rst != gt:
        # print(gt, rst)
        # print("Error with g: ", g)
        print("[-]Incorrect!! >_<")
    else:
        filename = './result/{}.txt'.format(str(len_sequence).zfill(7))
        # filename = './result/20w-50w.txt'
        with open(filename, 'w') as out_f:
            out_f.write('g_0=' + str(g[0]) + '\n')
            out_f.write('g_1=' + str(g[1]))
        print("[+]Yeah!! Your result is correct!")


# find the first nonzero k(a_i)
for k, val in enumerate(sequence):
    if val == '1':
        break

# define alpha, f and g
a_k = 1
alpha = gmpy2.mpz(2**k)
f = np.array([gmpy2.mpz(0), gmpy2.mpz(2)])
g = np.array([gmpy2.mpz(2**k), gmpy2.mpz(1)])

i = k+1
pgbar = progressbar.ProgressBar()
pgbar.start(len_sequence)
while i < len_sequence:
    # alpha += int(sequence[i]) * 2**i
    if int(sequence[i]) == 1:
        # alpha += 2**i
        alpha += 2 << (i-1)

    # if (alpha*g[1] - g[0]) % (2**(i+1)) == 0:
    if (alpha*g[1] - g[0]) % (2 << i) == 0:
        # f *= 2
        f <<= 1

    elif phi(g) < phi(f):
        bound = cal_bound(f, g)
        payload = [f, g, bound[0], bound[1]]
        d = controller(*payload)
        # d = find_d(*payload)
        g, f = f+d*g, 2*g
    else:
        bound = cal_bound(g, f)
        payload = [g, f, bound[0], bound[1]]
        d = controller(*payload)
        # d = find_d(g, f, bound[0], bound[1])
        g, f = g+d*f, f*2
    i += 1
    # print("f: ", f, "g:", g)
    pgbar.update(i)

pgbar.finish()

if len(str(g[0])) < 10000:
    print("result_g: {}".format(g))
else:
    print("the length of result is greater than 1e4..")

test(g, sequence)
