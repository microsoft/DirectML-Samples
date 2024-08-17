#pragma once

#define UNICODE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NODRAWTEXT
#define NOGDI
#define NOBITMAP
#define NOMCX
#define NOSERVICE
#define NOHELP

#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <wil/result.h>
#include <wil/resource.h>

#include <d3d12.h>
#include <dxgi1_6.h>
// #include "d3dx12.h"

#include <optional>
#include <span>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

using Microsoft::WRL::ComPtr;

struct ImageTensorData
{
    std::vector<std::byte> buffer;
    std::vector<int64_t> shape;

    const int64_t Channels() const { return shape[0]; }
    const int64_t Height() const { return shape[1]; }
    const int64_t Width() const { return shape[2]; }
    const int64_t Pixels() const { return Height() * Width(); }
};

ImageTensorData LoadTensorDataFromImageFilename(std::wstring_view filename)
{
    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmapDecoder> decoder;
    THROW_IF_FAILED(wicFactory->CreateDecoderFromFilename(
        filename.data(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder
    ));

    UINT frameCount;
    THROW_IF_FAILED(decoder->GetFrameCount(&frameCount));

    ComPtr<IWICBitmapFrameDecode> frame;
    THROW_IF_FAILED(decoder->GetFrame(0, &frame));

    UINT width, height;
    THROW_IF_FAILED(frame->GetSize(&width, &height));

    WICPixelFormatGUID pixelFormat;
    THROW_IF_FAILED(frame->GetPixelFormat(&pixelFormat));

    ComPtr<IWICBitmapSource> bitmapSource = frame;

    constexpr bool modelExpectsRGB = true;
    WICPixelFormatGUID desiredFormat = modelExpectsRGB ? GUID_WICPixelFormat24bppRGB : GUID_WICPixelFormat24bppBGR;
    if (pixelFormat != desiredFormat)
    {
        Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&converter));

        THROW_IF_FAILED(converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0f,
            WICBitmapPaletteTypeCustom
        ));

        Microsoft::WRL::ComPtr<IWICBitmap> bitmap;
        THROW_IF_FAILED(wicFactory->CreateBitmapFromSource(
            converter.Get(), 
            WICBitmapCacheOnLoad, 
            &bitmap
        ));

        bitmapSource = bitmap;
    }

    constexpr uint32_t channels = 3;

    // Read pixel data into HWC buffer with 8 bits per channel in RGB order
    std::vector<std::byte> pixelDataHWC8bpc(height * width * channels * sizeof(std::byte));
    const uint32_t pixelDataHWC8bpcStrideH = width * channels * sizeof(uint8_t);
    WICRect rect = { 0, 0, static_cast<INT>(width), static_cast<INT>(height) };
    THROW_IF_FAILED(bitmapSource->CopyPixels(
        &rect, 
        pixelDataHWC8bpcStrideH, 
        pixelDataHWC8bpc.size(), 
        reinterpret_cast<BYTE*>(pixelDataHWC8bpc.data())
    ));

    // Convert pixel data to CHW buffer with 32 bits per channel in RGB order
    std::vector<std::byte> pixelDataCHW32bpc(channels * height * width * sizeof(float));
    float* pixelDataCHWFloat = reinterpret_cast<float*>(pixelDataCHW32bpc.data());
    for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
    {
        float r = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 0]) / 255.0f;
        float g = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 1]) / 255.0f;
        float b = static_cast<float>(pixelDataHWC8bpc[pixelIndex * channels + 2]) / 255.0f;

        pixelDataCHWFloat[pixelIndex + 0 * height * width] = r;
        pixelDataCHWFloat[pixelIndex + 1 * height * width] = g;
        pixelDataCHWFloat[pixelIndex + 2 * height * width] = b;
    }

    return { pixelDataCHW32bpc, { channels, height, width } };
}

void SaveTensorDataToImageFilename(const ImageTensorData& tensorData, std::wstring_view filename)
{
    // Convert CHW tensor at 32 bits per channel to HWC tensor at 8 bits per channel
    auto src = reinterpret_cast<const float*>(tensorData.buffer.data());
    std::vector<BYTE> dst(tensorData.Pixels() * tensorData.Channels() * sizeof(std::byte));

    for (size_t pixelIndex = 0; pixelIndex < tensorData.Pixels(); pixelIndex++)
    {
        float r = src[pixelIndex + 0 * tensorData.Pixels()];
        float g = src[pixelIndex + 1 * tensorData.Pixels()];
        float b = src[pixelIndex + 2 * tensorData.Pixels()];

        dst[pixelIndex * tensorData.Channels() + 0] = static_cast<BYTE>(r * 255.0f);
        dst[pixelIndex * tensorData.Channels() + 1] = static_cast<BYTE>(g * 255.0f);
        dst[pixelIndex * tensorData.Channels() + 2] = static_cast<BYTE>(b * 255.0f);
    }


    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    // Create a WIC bitmap
    ComPtr<IWICBitmap> bitmap;
    THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
        tensorData.Width(),
        tensorData.Height(),
        GUID_WICPixelFormat24bppRGB,
        tensorData.Width() * tensorData.Channels(),
        dst.size(),
        dst.data(),
        &bitmap
    ));

    ComPtr<IWICStream> stream;
    THROW_IF_FAILED(wicFactory->CreateStream(&stream));

    THROW_IF_FAILED(stream->InitializeFromFilename(filename.data(), GENERIC_WRITE));

    ComPtr<IWICBitmapEncoder> encoder;
    THROW_IF_FAILED(wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));

    THROW_IF_FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> propertyBag;

    THROW_IF_FAILED(encoder->CreateNewFrame(&frame, &propertyBag));

    THROW_IF_FAILED(frame->Initialize(propertyBag.Get()));

    THROW_IF_FAILED(frame->WriteSource(bitmap.Get(), nullptr));

    THROW_IF_FAILED(frame->Commit());

    THROW_IF_FAILED(encoder->Commit());

    
}

std::tuple<ComPtr<IDMLDevice>, ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue()
{
    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12Device)));

    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));

    D3D12_COMMAND_QUEUE_DESC queueDesc = 
    {
        .Type = D3D12_COMMAND_LIST_TYPE_COMPUTE,
        .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };

    ComPtr<ID3D12CommandQueue> commandQueue;
    THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

    return { dmlDevice, commandQueue };
}

Ort::Session CreateOnnxRuntimeSession(IDMLDevice* dmlDevice, ID3D12CommandQueue* commandQueue, std::wstring_view modelPath)
{
    const OrtApi& ortApi = Ort::GetApi();
    // Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));

    Ort::SessionOptions sessionOptions;
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    const OrtDmlApi* ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, dmlDevice, commandQueue));

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "DirectML_CV");

    return Ort::Session(env, modelPath.data(), sessionOptions);
}

int main(int argc, char** argv)
{
    THROW_IF_FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));



    // auto [dmlDevice, commandQueue] = CreateDmlDeviceAndCommandQueue();

    // auto ortSession = CreateOnnxRuntimeSession(dmlDevice.Get(), commandQueue.Get(), LR"(C:\src\ort_sr_demo\xlsr.onnx)");

    // // NCHW

    // // load input image

    // // Ort::Value inputTensor = Ort::Value::CreateTensor<float>({ 1, 3, 128, 128 }, dmlDevice.Get());

    // Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, data, dataSizeInElements,  shape, 4);


    // // ortApi.CreateTensorWithDataAsOrtValue(dmlDevice.Get(), inputTensor.Get(), inputTensor.GetElementCount(), inputTensor.GetElementSize(), inputTensor.GetData(), inputTensor.GetAllocatedDataSize(), inputTensor.GetElementType());

    // // ortSession.Run

    auto tensorData = LoadTensorDataFromImageFilename(LR"(C:\src\ort_sr_demo\zebra.jpg)");
    SaveTensorDataToImageFilename(tensorData, LR"(C:\src\ort_sr_demo\zebra_out.jpg)");

    CoUninitialize();

    return 0;
}