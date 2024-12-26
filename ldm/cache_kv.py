import gc
import torch
from collections import defaultdict

# Flags
mode = None  # 'save', 'use'

# Cache K/V with dictionaries
''' what the dictionary looks like
{
  attention_layer_1: [ k/v_of_ref_1, ..., k/v_of_ref_n ],
  attention_layer_2: [ k/v_of_ref_1, ..., k/v_of_ref_n ], ...
}
'''
k = defaultdict(list)
v = defaultdict(list)


def clear_cache():
    while len(k) > 0:
        _, tmp_list = k.popitem()
        while len(tmp_list) > 0:
            del tmp_list[-1]
    while len(v) > 0:
        _, tmp_list = v.popitem()
        while len(tmp_list) > 0:
            del tmp_list[-1]
    n_gc = gc.collect()
    torch.cuda.empty_cache()