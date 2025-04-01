Due to the size of the entire RULER evaluation set, we selected a few distinguishable subsets for evaluation, which consist of the following datasets: `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, and `qa_2`.

### `sequence length` = 64k

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |                 |      |      |      |      |
| AlignedKV         |                 |      |      |      |      |
| kivi-8bit         |                 |      |      |      |      |
| StreammingLLM-16k |                 |      |      |      |      |

### `sequence length` = 128k

| Method            | niah_multikey_2 | vt   | fwe  | qa_1 | qa_2 |
| ----------------- | --------------- | ---- | ---- | ---- | ---- |
| baseline          |                 |      |      |      |      |
| AlignedKV         |                 |      |      |      |      |
| kivi-8bit         |                 |      |      |      |      |
| StreammingLLM-16k |                 |      |      |      |      |
