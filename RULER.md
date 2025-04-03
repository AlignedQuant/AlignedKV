Due to the size of the entire RULER evaluation set, we selected a few distinguishable subsets for evaluation, which consist of the following datasets: `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, and `qa_2`.

### `sequence length` = 64k Llama-3.1-8B-Instruct

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |   97.2   |  93.0  |   84.13 | 76.4 | 49.2   |
| AlignedKV         |                 |      |      |      |      |
| kivi-8bit         |                 |      |      |      |      |
| StreammingLLM-16k |                 |      |      |      |      |

### `sequence length` = 128k Llama-3.1-8B-Instruct

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |   74.6          |    54.56  |    75.0  |   71.6   |   41.8   |
| AlignedKV         |                 |      |      |      |      |
| kivi-8bit         |                 |      |      |      |      |
| StreammingLLM-16k |                 |      |      |      |      |

### `sequence length` = 64k Llama-3.2-3B-Instruct

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |         70.2        |   62.56   |   81.87   |  45.0    |  41.2    |
| AlignedKV         |         **70.2** | **63.2** | **83.53** | **44.6** | 41.0        |
| kivi-8bit         |         70.0 | 62.28 | 82.07 | 44.4 | **41.2** |
| StreammingLLM-16k |         21.6 | 18.04 | 53.73 | 27.2 | 24.2 |

### `sequence length` = 128k Llama-3.2-3B-Instruct

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |         57.0        |   41.64   |   62.6   |   39.6   |   35.2   |
| AlignedKV         | 56.8 | **42.4** | 61.93 | **41.0** | **35.6** |
| kivi-8bit         | **58.2** | 41.28 | **62.53** | 40.2 | 35.4 |
| StreammingLLM-16k | 10.2 | 7.48 | 28.2 | 22.6 | 23.4 |
