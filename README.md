# AlignedKV
AlignedKV: Reducing Memory Access of KV-Cache with Precision-Aligned Quantization

## Abstruct
Model quantization has become a crucial technique to address the issues of large memory consumption and long inference times associated with LLMs. Mixed-precision quantization, which distinguishes between important and unimportant parameters, stands out among numerous quantization schemes as it achieves a balance between precision and compression rate. However, existing approaches can only identify important parameters through qualitative analysis and manual experiments without quantitatively analyzing how their importance is determined. We propose a new criterion, so-called “precision alignment”, to build a quantitative framework to holistically evaluate the importance of parameters in mixed-precision quantization. Our observations on floating point addition under various real-world scenarios suggest that two addends should have identical precision, otherwise the information in the higher-precision number will be wasted. Such an observation offers an essential principle to determine the precision of each parameter in matrix multiplication operation. As the first step towards applying the above discovery to large model inference, we develop a dynamic KV-Cache quantization technique to effectively reduce memory access latency. Different from existing quantization approaches that focus on memory saving, this work directly aims to accelerate LLM inference through quantifying floating numbers. The proposed technique attains a 25% saving of memory access and delivers up to 1.3× speedup in the computation of attention in the decoding phase of LLM, with almost no loss of precision.

## Usage
### Prepare
```bash
cuda version: 11.8
python version: 3.10
torch version: 2.3.0
install gcc-10, g++-10 in /usr/bin/gcc-10, /usr/bin/g++-10
```
You can use the following command to install gcc-10 and g++-10
```bash
sudo apt install gcc-10 g++-10
```
### Installation
```bash
git clone https://github.com/AlignedQuant/AlignedKV.git
cd AlignedKV
pip install -r requirements.txt
```
### Run example:
```bash
python example.py
```
You can rewrite the example.py to modify the input data.
```python
# config
max_batch_size = 1
max_cache_len = 30

# input data
prompt = "Hey, are you conscious? Can you talk to me?"
```

## Run Experiment
Please switch the branch to `evaluate` to run the experiment.

You can read the `README.md` in the `evaluate` branch to get more information.
