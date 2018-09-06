# Finding Numerical Errors in Trained Image Classifiers

First you need to train a model that you suspect may have numerical issues:

```
python examples/nans/nan_model.py --checkpoint_dir=/tmp/nanfuzzer --data_dir=/tmp/mnist --training_steps=35000 --init_scale=0.25
```

Then you can fuzz this model by pointing the fuzzer at its checkpoints.

```
python examples/nans/nan_fuzzer.py --checkpoint_dir=/tmp/nanfuzzer --total_inputs_to_fuzz=1000000 --mutations_per_corpus_item=100 --alsologtostderr --ann_threshold=0.5
```
