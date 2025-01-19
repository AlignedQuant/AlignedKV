## V100

 [volta-architecture-whitepaper.pdf](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) 

|         **参数**         |      **Tesla V100**      |
| :----------------------: | :----------------------: |
|           GPU            |      GV100 (Volta)       |
|           SMs            |            80            |
|           TPCs           |            40            |
|     FP32 Cores / SM      |            64            |
|     FP32 Cores / GPU     |           5120           |
|     FP64 Cores / SM      |            32            |
|     FP64 Cores / GPU     |           2560           |
|    Tensor Cores / SM     |            8             |
|    Tensor Cores / GPU    |           640            |
|     GPU Boost Clock      |         1530 MHz         |
|     Peak FP32 TFLOPS     |           15.7           |
|     Peak FP64 TFLOPS     |           7.8            |
|    Peak Tensor TFLOPS    |           125            |
|      Texture Units       |           320            |
|     Memory Interface     |      4096-bit HBM2       |
|       Memory Size        |          16 GB           |
|      L2 Cache Size       |         6144 KB          |
| Shared Memory Size / SM  | Configurable up to 96 KB |
| Register File Size / SM  |          256 KB          |
| Register File Size / GPU |         20480 KB         |

- tensor core: 4\*4\*4

|                 GPU                 |     Kepler GK180      | Maxwell GM200 | Pascal GP100 |       Volta GV100        |
| :---------------------------------: | :-------------------: | :-----------: | :----------: | :----------------------: |
|         Compute Capability          |          3.5          |      5.2      |     6.0      |           7.0            |
|           Threads / Warp            |          32           |      32       |      32      |            32            |
|           Max Warps / SM            |          64           |      64       |      64      |            64            |
|          Max Threads / SM           |         2048          |     2048      |     2048     |           2048           |
|       Max Thread Blocks / SM        |          16           |      32       |      32      |            32            |
|      Max 32-bit Registers / SM      |         65536         |     65536     |    65536     |          65536           |
|        Max Registers / Block        |         65536         |     32768     |    65536     |          65536           |
|       Max Registers / Thread        |          255          |      255      |     255      |           255            |
|        Max Thread Block Size        |         1024          |     1024      |     1024     |           1024           |
|           FP32 Cores / SM           |          192          |      128      |      64      |            64            |
| Ratio of SM Registers to FP32 Cores |          341          |      512      |     1024     |           1024           |
|       Shared Memory Size / SM       | 16 KB / 32 KB / 48 KB |     96 KB     |    64 KB     | Configurable up to 96 KB |

## A100

 [nvidia-ampere-architecture-whitepaper.pdf](https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) 

|                   Feature                    | NVIDIA Tesla P100 |    NVIDIA Tesla V100     |        NVIDIA A100        |
| :------------------------------------------: | :---------------: | :----------------------: | :-----------------------: |
|                 GPU Codename                 |       GP100       |          GV100           |           GA100           |
|               GPU Architecture               |   NVIDIA Pascal   |       NVIDIA Volta       |       NVIDIA Ampere       |
|            GPU Board Form Factor             |        SXM        |           SXM2           |           SXM4            |
|                     SMs                      |        56         |            80            |            108            |
|                     TPCs                     |        28         |            40            |            54             |
|                FP32 Cores/SM                 |        64         |            64            |            64             |
|                FP32 Cores/GPU                |       3584        |           5120           |           6912            |
|         FP64 Cores/SM (excl. Tensor)         |        32         |            32            |            32             |
|        FP64 Cores/GPU (excl. Tensor)         |       1792        |           2560           |           3456            |
|                INT32 Cores/SM                |        NA         |            64            |            64             |
|               INT32 Cores/GPU                |        NA         |           5120           |           6912            |
|               Tensor Cores/SM                |        NA         |            8             |             4             |
|               Tensor Cores/GPU               |        NA         |           640            |            432            |
|               GPU Boost Clock                |     1480 MHz      |         1530 MHz         |         1410 MHz          |
| Peak FP16 Tensor TFLOPS with FP16 Accumulate |        NA         |           125            |          312/624          |
| Peak FP16 Tensor TFLOPS with FP32 Accumulate |        NA         |           125            |          312/624          |
| Peak BF16 Tensor TFLOPS with FP32 Accumulate |        NA         |            NA            |          312/624          |
|           Peak TF32 Tensor TFLOPS            |        NA         |            NA            |          156/312          |
|           Peak FP64 Tensor TFLOPS            |        NA         |            NA            |           19.5            |
|            Peak INT8 Tensor TOPS             |        NA         |            NA            |         624/1248          |
|            Peak INT4 Tensor TOPS             |        NA         |            NA            |         1248/2496         |
|        Peak FP16 TFLOPS (non-Tensor)         |       21.2        |           31.4           |            78             |
|        Peak BF16 TFLOPS (non-Tensor)         |        NA         |            NA            |            39             |
|        Peak FP32 TFLOPS (non-Tensor)         |       10.6        |           15.7           |           19.5            |
|        Peak FP64 TFLOPS (non-Tensor)         |        5.3        |           7.8            |            9.7            |
|               Peak INT32 TOPS                |        NA         |           15.7           |           19.5            |
|                Texture Units                 |        224        |           320            |            432            |
|               Memory Interface               |   4096-bit HBM2   |      4096-bit HBM2       |       5120-bit HBM2       |
|                 Memory Size                  |       16 GB       |      32 GB / 16 GB       |           40 GB           |
|               Memory Data Rate               |    703 MHz DDR    |      877.5 MHz DDR       |       1215 MHz DDR        |
|               Memory Bandwidth               |    720 GB/sec     |        900 GB/sec        |        1555 GB/sec        |
|                L2 Cache Size                 |      4096 KB      |         6144 KB          |         40960 KB          |
|            Shared Memory Size/SM             |       64 KB       | Configurable up to 96 KB | Configurable up to 164 KB |
|            Register File Size/SM             |      256 KB       |          256 KB          |          256 KB           |
|            Register File Size/GPU            |     14336 KB      |         20480 KB         |         27648 KB          |
|                     TDP                      |     300 Watts     |        300 Watts         |         400 Watts         |
|                 Transistors                  |   15.3 billion    |       21.1 billion       |       54.2 billion        |
|                 GPU Die Size                 |      610 mm²      |         815 mm²          |          826 mm²          |
|          TSMC Manufacturing Process          |   16 nm FinFET+   |        12 nm FFN         |          7 nm N7          |

tensor core: 8\*4\*8, A(8\*8),B(8\*4)