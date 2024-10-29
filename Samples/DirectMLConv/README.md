# DirectMLConv

This application illustrates how to use D3D12 and DirectML APIs to implement and execute a simple convolution layer.

The sample uses the following tensor configuration for the input and filter 
- input tensor: 1x1x3x3
- filter tensor: 1x1x1x1
- output tensor: 1x1x3x3

example input and output
```sh
filter tensor:
5.6000
input tensor:
-2.0000 -2.0000 -2.0000 -2.0000 -2.0000 -2.0000 -2.0000 -2.0000 -2.0000

output tensor:
-11.200000 -11.200000 -11.200000 -11.200000 -11.200000 -11.200000 -11.200000 -11.200000 -11.200000
```

## How to use?
- Install Visual Studio 2022
- Open the project using Visual Studio 2022
- Build the solution which downloads all dependencies
- Run/debug the project