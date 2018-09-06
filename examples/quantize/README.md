# Finding Disagreements Between fp16 and fp32 models.

This example directory contains code to check for differences related to quantizing models.
The quantized_model.py file trains a model using 32 bit floating point variables and then
casts those variables to use 16 bits.
To train this model, execute something like this:

```
python examples/quantize/quantized_model.py --checkpoint_dir='/tmp/quantized_checkpoints_2' --training_steps=10000
```

To fuzz the trained model, execute something like this:

```
python examples/quantize/quantized_fuzzer.py --checkpoint_dir=/tmp/quantized_checkpoints_2 --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=100 --alsologtostderr --output_path=/cns/ok-d/home/augustusodena/fuzzer/plots/quantized_image.png --ann_threshold=1.0 --perturbation_constraint=1.0 --strategy=ann
```
