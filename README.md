# MetaEmbedding_torch

Codes for our SIGIR-2019 paper: 

**[Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings](https://dl.acm.org/citation.cfm?id=3331268)**

This repo includes an example for training Meta-Embedding upon a deepFM model on the binarized MovieLens-1M dataset. The dataset is preprocessed and splitted already.

Requirements: Python 3 and PyTorch. 

**[Tensorflow Code (Paper Author Version)](https://github.com/Feiyang/MetaEmbedding)**


# Performance (Tensorflow v.s. PyTorch)

|| Tensorflow | PyTorch |
|--|--|--|
|Warm Model|   0.650700 | 0.610819|
|Cold Model(Test_a)| 0.665154| 0.610174|
|Cold Model(Test_b)| 0.671722| 0.614743|
|Cold Model(Test_c)| 0.673590| 0.611912|



# Bibtex

```
@inproceedings{pan2019warm,
 author = {Pan, Feiyang and Li, Shuokai and Ao, Xiang and Tang, Pingzhong and He, Qing},
 title = {Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings},
 booktitle = {Proceedings of the 42Nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR'19},
 year = {2019},
 isbn = {978-1-4503-6172-9},
 location = {Paris, France},
 pages = {695--704},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3331184.3331268},
 doi = {10.1145/3331184.3331268},
 acmid = {3331268},
 publisher = {ACM},
 address = {New York, NY, USA},
} 
```






test a - meta model AUC : 0.602745
LOSS : 0.706194
test b - meta model AUC : 0.600163
LOSS : 0.715177
test c - meta model AUC : 0.595077
LOSS : 0.707629
2. tensorflow version
based model AUC : 0.651291
LOSS : 1.219576
test a - meta model AUC : 0.664774
LOSS : 1.164091
test b - meta model AUC : 0.670858
LOSS : 1.106340
test c - meta model AUC :0.672514
LOSS : 1.073448
———
I think it is weird because torch model LOSS is lower, but AUC is low