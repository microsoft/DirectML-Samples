// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "helpers.h"

void RunModel(
    IDMLDevice* dmlDevice, 
    ID3D12CommandQueue* d3dQueue, 
    const std::filesystem::path& modelPath,
    const std::filesystem::path& imagePath)
{
    // DML execution provider prefers these session options.
    Ort::SessionOptions sessionOptions;
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    // By passing in an explicitly created DML device & queue, the DML execution provider sends work
    // to the desired device. If not used, the DML execution provider will create its own device & queue.
    const OrtApi& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(
        sessionOptions, 
        dmlDevice, 
        d3dQueue
    ));

    // Load ONNX model into a session.
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");
    Ort::Session ortSession(env, modelPath.wstring().c_str(), sessionOptions);

    if (ortSession.GetInputCount() != 1 && ortSession.GetOutputCount() != 1)
    {
        throw std::invalid_argument("Model must have exactly one input and one output");
    }

    Ort::AllocatorWithDefaultOptions ortAllocator;

    auto inputName = ortSession.GetInputNameAllocated(0, ortAllocator);
    auto inputTypeInfo = ortSession.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();
    auto inputDataType = inputTensorInfo.GetElementType();
    if (inputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && inputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        throw std::invalid_argument("Model input must be of type float32 or float16");
    }
    if (inputShape.size() < 3)
    {
        throw std::invalid_argument("Model input must be an image with 3 or 4 dimensions");
    }

    const uint32_t inputChannels = inputShape[inputShape.size() - 3];
    const uint32_t inputHeight = inputShape[inputShape.size() - 2];
    const uint32_t inputWidth = inputShape[inputShape.size() - 1];
    const uint32_t inputElementSize = inputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);

    auto outputName = ortSession.GetOutputNameAllocated(0, ortAllocator);
    auto outputTypeInfo = ortSession.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    auto outputShape = outputTensorInfo.GetShape();
    auto outputDataType = outputTensorInfo.GetElementType();
    if (outputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && outputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        throw std::invalid_argument("Model output must be of type float32 or float16");
    }
    if (outputShape.size() < 3)
    {
        throw std::invalid_argument("Model output must be an image with 3 or 4 dimensions");
    }

    const uint32_t outputChannels = outputShape[outputShape.size() - 3];
    const uint32_t outputHeight = outputShape[outputShape.size() - 2];
    const uint32_t outputWidth = outputShape[outputShape.size() - 1];
    const uint32_t outputElementSize = outputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);

    // Load image and transform it into an NCHW tensor with the correct shape and data type.
    std::vector<std::byte> inputBuffer(inputChannels * inputHeight * inputWidth * inputElementSize);
    FillNCHWBufferFromImageFilename(imagePath.wstring(), inputBuffer, inputHeight, inputWidth, inputDataType, ChannelOrder::RGB);

    std::cout << "Saving cropped/scaled image to input.png" << std::endl;
    SaveNCHWBufferToImageFilename(L"input.png", inputBuffer, inputHeight, inputWidth, inputDataType, ChannelOrder::RGB);

    // For simplicity, this sample binds input/output buffers in system memory instead of DirectX resources.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor(
        memoryInfo, 
        inputBuffer.data(), 
        inputBuffer.size(), 
        inputShape.data(), 
        inputShape.size(),
        inputDataType
    );

    auto bindings = Ort::IoBinding::IoBinding(ortSession);
    bindings.BindInput(inputName.get(), inputTensor);
    bindings.BindOutput(outputName.get(), memoryInfo);

    // Run the session to get inference results.
    Ort::RunOptions runOpts;
    ortSession.Run(runOpts, bindings);
    bindings.SynchronizeOutputs();

    std::span<const std::byte> outputBuffer(
        reinterpret_cast<const std::byte*>(bindings.GetOutputValues()[0].GetTensorRawData()),
        outputChannels * outputHeight * outputWidth * outputElementSize
    );

    std::cout << "Saving inference results to output.png" << std::endl;
    SaveNCHWBufferToImageFilename(
        L"output.png", 
        outputBuffer, 
        outputHeight, 
        outputWidth, 
        outputDataType, 
        ChannelOrder::RGB
    );
}

int main(int argc, char** argv)
{
    using Microsoft::WRL::ComPtr;

    try
    {
        // Functions in image_helpers.h use WIC APIs, which require CoInitialize.
        auto coInit = wil::CoInitializeEx_failfast(COINIT_MULTITHREADED);

        // Parse command-line options.
        cxxopts::Options commandLineParams("directml_cv", "DirectML Computer Vision Sample");
        commandLineParams.add_options()
            ("m,model", "Path to ONNX model file", cxxopts::value<std::string>()->default_value("esrgan.onnx"))
            ("i,image", "Path to input image file", cxxopts::value<std::string>()->default_value("zebra.jpg"))
            ("a,adapter", "Adapter name substring filter", cxxopts::value<std::string>()->default_value(""));

        auto commandLineArgs = commandLineParams.parse(argc, argv);

        // First try to run on NPU, then GPU.
        std::vector<std::tuple<std::string, bool>> adapters = {{"NPU", false}, {"GPU", true}};
        for (uint32_t i = 0; i < adapters.size(); i++)
        {
            auto [adapterType, isGraphicsAdapter] = adapters[i];
            std::cout << "Trying on " << adapterType << std::endl;
            try
            {
                // See helpers.h for logic to select a DXCore adapter, create DML device, and create D3D command queue.
                auto [dmlDevice, commandQueue] = CreateDmlDeviceAndCommandQueue(
                    adapterType,
                    isGraphicsAdapter,
                    commandLineArgs["adapter"].as<std::string>()
                );

                RunModel(
                    dmlDevice.Get(),
                    commandQueue.Get(),
                    commandLineArgs["model"].as<std::string>(),
                    commandLineArgs["image"].as<std::string>()
                );
            }
            catch (const std::exception& e)
            {
                std::cerr << adapterType << " Error: " << e.what() << std::endl;
                continue;
            }

            return 0;
        }

    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 1;
}

