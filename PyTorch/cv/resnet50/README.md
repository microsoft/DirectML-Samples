# Resnet50 Model <!-- omit in toc -->

Sample scripts for training the Resnet50 model using PyTorch on DirectML.

These scripts were forked from https://github.com/pytorch/benchmark. The original code is Copyright (c) 2019, pytorch, and is used here under the terms of the BSD 3-Clause License. See [LICENSE](https://github.com/pytorch/benchmark/blob/main/LICENSE) for more information.

The original paper can be found at: https://arxiv.org/abs/1602.07360

- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training](#training)
- [Testing](#testing)
- [Predict](#predict)
- [Tracing](#tracing)
- [External Links](#external-links)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\cv\resnet50\requirements.txt 
```

## Prepare Data

After installing the PyTorch on DirectML package (see [GPU accelerated ML training](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)), open a console to the `root` directory and run the setup script to download and convert data:

```
python pytorch\cv\data\dataset.py
```

Running `setup.py` should take at least a minute or so, since it downloads the CIFAR-10 dataset. The output of running it should look similar to the following:

```
>python pytorch\cv\data\dataset.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz
Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz
170499072it [00:32, 5250164.09it/s]
Extracting E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python
```

## Training

A helper script exists to train Resnet50 with default data, batch size, and so on:

```
python pytorch\cv\resnet50\train.py
```

The first few lines of output should look similar to the following (exact numbers may change):
```
>python pytorch\cv\resnet50\train.py
Loading the training dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 0.2560007572174072s.
Epoch 1
-------------------------------
loss: 2.309573  [    0/50000]
loss: 2.324099  [ 3200/50000]
loss: 2.297763  [ 6400/50000]
loss: 2.292575  [ 9600/50000]
loss: 2.251738  [12800/50000]
loss: 2.183397  [16000/50000]
loss: 2.130508  [19200/50000]
loss: 2.000042  [22400/50000]
loss: 2.183213  [25600/50000]
loss: 2.250935  [28800/50000]
loss: 1.730087  [32000/50000]
loss: 1.999480  [35200/50000]
loss: 1.865684  [38400/50000]
loss: 2.058377  [41600/50000]
loss: 2.059475  [44800/50000]
loss: 2.279521  [48000/50000]
current highest_accuracy:  0.2856000065803528
Test Error:
 Accuracy: 28.6%, Avg loss: 1.862064
```

By default, the script will run for 50 epochs with a batch size of 32 and print the accuracy after every 100 batches. The training script can be run multiple times and saves progress after each epoch (by default). The accuracy should increase over time.

> When discrete memory or shared GPU memory is insufficient consider running the same scripts with a smaller batch size (use the --batch_size argument). For example:

```
python pytorch\cv\resnet50\train.py --batch_size 8
```

You can inspect `train.py` (and the real script, `pytorch\cv\classification\train_classification.py`) to see the command line it is invoking or adjust some of the parameters.

You can save the model for testing by passing in the --save_model flag. This will cause checkpoints to be saved to the `pytorch\cv\checkpoints\<device>\<model>\checkpoint.pth` path.

```
python pytorch\cv\resnet50\train.py --save_model
```


## Testing

Once the model is trained and saved we can now test the model using the following steps. The test script will use the latest trained model from the checkpoints folder.

```
python pytorch\cv\resnet50\test.py
```

You should see the result such as this:

```
>python pytorch\cv\resnet50\test.py
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 0.6159994602203369s.
current highest_accuracy:  0.10559999942779541
Test Error:
 Accuracy: 10.0%, Avg loss: 2.321213
```
## Predict

Once the model is trained and saved we can now run the prediction using the following steps. The predict script will use that latest trained model from the checkpoints folder.

```
python pytorch\cv\resnet50\predict.py --image E:\a.jpeg
```

You should see the result such as this:

```
E:\work\dml>python pytorch\cv\resnet50\predict.py --image E:\a.jpeg
hammerhead 0.35642221570014954
stingray 0.34619468450546265
electric ray 0.09593362361192703
cock 0.07319413870573044
great white shark 0.06555310636758804
```

## Tracing

It may be useful to get a trace during training or evaluation.

```
python pytorch\cv\resnet50\test.py --trace True
python pytorch\cv\resnet50\train.py --trace True
```

With default settings, you'll see output like the following:

```
>python pytorch\cv\resnet50\train.py --trace Tue
Loading the training dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([1, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([1]),
                dtype=torch.int64
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([1, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([1]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 0.594947338104248s.
Epoch 1
-------------------------------
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                           model_inference        34.65%     823.161ms        66.84%        1.588s        1.588s          -4 b         -20 b             1
                        ThnnConv2DBackward         0.05%       1.119ms        21.18%     503.098ms       9.492ms           0 b           0 b            53
                aten::thnn_conv2d_backward        21.04%     499.936ms        21.13%     501.979ms       9.471ms           0 b           0 b            53
                   Optimizer.step#SGD.step         0.24%       5.683ms        10.84%     257.530ms     257.530ms          -4 b         -20 b             1
                          aten::batch_norm         0.09%       2.118ms         8.96%     212.849ms       4.016ms           0 b           0 b            53
              aten::_batch_norm_impl_index         0.08%       1.846ms         8.87%     210.731ms       3.976ms           0 b           0 b            53
                   aten::native_batch_norm         3.82%      90.859ms         8.73%     207.468ms       3.914ms           0 b           0 b            53
                                 aten::add         6.64%     157.698ms         7.77%     184.523ms     862.258us           0 b           0 b           214
                               aten::empty         5.60%     133.136ms         5.60%     133.136ms     166.005us          60 b          60 b           802
                              aten::conv2d         0.08%       1.843ms         5.59%     132.890ms       2.507ms           0 b           0 b            53
                         aten::convolution         0.07%       1.559ms         5.52%     131.047ms       2.473ms           0 b           0 b            53
                        aten::_convolution         0.22%       5.117ms         5.45%     129.488ms       2.443ms           0 b           0 b            53
                aten::_convolution_nogroup         0.08%       1.810ms         5.24%     124.371ms       2.347ms           0 b           0 b            53
                         aten::thnn_conv2d         0.07%       1.760ms         5.16%     122.562ms       2.312ms           0 b           0 b            53
                 aten::thnn_conv2d_forward         4.92%     116.862ms         5.08%     120.802ms       2.279ms           0 b           0 b            53
                   NativeBatchNormBackward         0.05%       1.202ms         4.86%     115.441ms       2.178ms           0 b           0 b            53
          aten::native_batch_norm_backward         3.06%      72.769ms         4.81%     114.239ms       2.155ms           0 b           0 b            53
                       aten::empty_strided         4.68%     111.158ms         4.68%     111.158ms     295.634us           0 b           0 b           376
                               aten::clone         0.67%      15.835ms         3.07%      73.035ms     453.637us           0 b           0 b           161
                          aten::empty_like         0.12%       2.741ms         3.00%      71.267ms     334.588us           0 b           0 b           213
                                aten::add_         2.92%      69.436ms         2.92%      69.436ms     392.292us           0 b           0 b           177
    struct torch::autograd::AccumulateGrad         0.12%       2.960ms         2.62%      62.349ms     387.258us           0 b           0 b           161
                   aten::new_empty_strided         0.06%       1.337ms         2.10%      49.896ms     309.912us           0 b           0 b           161
                             AddmmBackward         0.00%      56.400us         1.84%      43.649ms      43.649ms           0 b           0 b             1
                                  aten::mm         1.79%      42.570ms         1.83%      43.489ms      21.745ms           0 b           0 b             2
                             ReluBackward1         0.02%     394.800us         1.73%      40.983ms     836.398us           0 b           0 b            49
                  aten::threshold_backward         1.71%      40.589ms         1.71%      40.589ms     828.341us           0 b           0 b            49
                               aten::copy_         1.68%      39.820ms         1.68%      39.820ms      82.787us           0 b           0 b           481
                                  aten::to         0.08%       1.928ms         1.13%      26.825ms     506.126us           0 b           0 b            53
                         aten::log_softmax         0.00%      42.400us         0.82%      19.532ms      19.532ms           0 b           0 b             1
                        aten::_log_softmax         0.82%      19.489ms         0.82%      19.489ms      19.489ms           0 b           0 b             1
         Optimizer.zero_grad#SGD.zero_grad         0.52%      12.294ms         0.80%      19.066ms      19.066ms          -4 b         -20 b             1
                             aten::reshape         0.54%      12.869ms         0.78%      18.629ms      49.811us           0 b           0 b           374
                            aten::nll_loss         0.03%     645.100us         0.56%      13.385ms      13.385ms           0 b           0 b             1
                    aten::nll_loss_forward         0.53%      12.600ms         0.54%      12.740ms      12.740ms           0 b           0 b             1
                               aten::relu_         0.36%       8.556ms         0.36%       8.556ms     174.618us           0 b           0 b            49
                              aten::linear         0.00%      49.400us         0.31%       7.462ms       7.462ms           0 b           0 b             1
                          aten::max_pool2d         0.01%     324.600us         0.31%       7.409ms       7.409ms           0 b           0 b             1
                               aten::addmm         0.29%       6.982ms         0.31%       7.312ms       7.312ms           0 b           0 b             1
             aten::max_pool2d_with_indices         0.30%       7.085ms         0.30%       7.085ms       7.085ms           0 b           0 b             1
                               aten::zero_         0.29%       6.806ms         0.29%       6.806ms      41.498us           0 b           0 b           164
              MaxPool2DWithIndicesBackward         0.00%      30.200us         0.28%       6.579ms       6.579ms           0 b           0 b             1
    aten::max_pool2d_with_indices_backward         0.28%       6.548ms         0.28%       6.548ms       6.548ms           0 b           0 b             1
                                aten::view         0.24%       5.794ms         0.24%       5.794ms      15.451us           0 b           0 b           375
                              aten::detach         0.13%       3.044ms         0.24%       5.736ms      35.630us           0 b           0 b           161
                        LogSoftmaxBackward         0.03%     601.300us         0.24%       5.601ms       5.601ms           0 b           0 b             1
                 AdaptiveAvgPool2DBackward         0.00%      13.700us         0.23%       5.370ms       5.370ms           0 b           0 b             1
       aten::_adaptive_avg_pool2d_backward         0.23%       5.357ms         0.23%       5.357ms       5.357ms           0 b           0 b             1
          aten::_log_softmax_backward_data         0.21%       5.000ms         0.21%       5.000ms       5.000ms           0 b           0 b             1
                           aten::ones_like         0.00%      27.700us         0.20%       4.692ms       4.692ms           0 b           0 b             1
                               aten::fill_         0.18%       4.363ms         0.18%       4.363ms       4.363ms           0 b           0 b             1
                           NllLossBackward         0.04%     917.500us         0.15%       3.485ms       3.485ms           0 b           0 b             1
                                    detach         0.11%       2.692ms         0.11%       2.692ms      16.721us           0 b           0 b           161
                   aten::nll_loss_backward         0.11%       2.556ms         0.11%       2.567ms       2.567ms           0 b           0 b             1
                          aten::as_strided         0.05%       1.290ms         0.05%       1.290ms       3.402us           0 b           0 b           379
                           aten::transpose         0.02%     579.900us         0.04%     898.000us       5.476us           0 b           0 b           164
                               aten::zeros         0.02%     575.100us         0.04%     865.200us     288.400us          12 b           0 b             3
                                 TBackward         0.02%     376.000us         0.02%     398.500us     398.500us           0 b           0 b             1
                        aten::broadcast_to         0.01%     281.000us         0.01%     329.400us     329.400us           0 b           0 b             1
                 aten::adaptive_avg_pool2d         0.00%      30.400us         0.01%     321.500us     321.500us           0 b           0 b             1
                aten::_adaptive_avg_pool2d         0.01%     291.100us         0.01%     291.100us     291.100us           0 b           0 b             1
                                   aten::t         0.00%     111.200us         0.01%     204.400us      40.880us           0 b           0 b             5
                             aten::squeeze         0.00%      61.200us         0.00%      95.700us      47.850us           0 b           0 b             2
                             aten::flatten         0.00%      26.800us         0.00%      61.000us      61.000us           0 b           0 b             1
                              AddBackward0         0.00%      56.200us         0.00%      56.200us       3.513us           0 b           0 b            16
                              aten::expand         0.00%      29.300us         0.00%      48.400us      48.400us           0 b           0 b             1
                                aten::conj         0.00%      21.900us         0.00%      21.900us      10.950us           0 b           0 b             2
                              ViewBackward         0.00%       8.200us         0.00%      20.400us      20.400us           0 b           0 b             1
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.376s

Done! with highest_accuracy:  0
```

## External Links

- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternative implementations:
- [ONNX](https://github.com/onnx/models/tree/master/vision/classification/resnet)
