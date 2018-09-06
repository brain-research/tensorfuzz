# Finding Broken Loss Functions in Public Code

This example directory contains code to fuzz a slight modification to the 
well known [DCGAN-tensorflow repository](https://github.com/carpedm20/DCGAN-tensorflow).

To find the issue (which is a loss function that can yield a high loss but zero gradients)
execute the following:

```
python examples/dcgan/dcgan_fuzzer.py  --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=64 --alsologtostderr --strategy=ann --ann_threshold=0.1
```
