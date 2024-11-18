// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "helpers.h"

std::tuple<Microsoft::WRL::ComPtr<IDXCoreAdapter>, D3D_FEATURE_LEVEL> SelectAdapter(
    std::string adapterType,
    bool useGraphicsAdapter,
    std::string_view adapterNameFilter)
{
    using Microsoft::WRL::ComPtr;

    ComPtr<IDXCoreAdapterFactory> adapterFactory;
    THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(adapterFactory.GetAddressOf())));

    // First try getting all GENERIC_ML devices, which is the broadest set of adapters 
    // and includes both GPUs and NPUs; however, running this sample on an older build of 
    // Windows may not have drivers that report GENERIC_ML.
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_1_0_GENERIC;
    ComPtr<IDXCoreAdapterList> adapterList;
    THROW_IF_FAILED(adapterFactory->CreateAdapterList(
        1,
        &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML,
        adapterList.GetAddressOf()
    ));

    // Fall back to CORE_COMPUTE if GENERIC_ML devices are not available. This is a more restricted
    // set of adapters and may filter out some NPUs.
    if (adapterList->GetAdapterCount() == 0)
    {
        featureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
        THROW_IF_FAILED(adapterFactory->CreateAdapterList(
            1, 
            &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 
            adapterList.GetAddressOf()
        ));
    }

    if (adapterList->GetAdapterCount() == 0)
    {
        throw std::runtime_error("No compatible adapters found.");
    }

    // Sort the adapters by preference, with hardware and high-performance adapters first.
    DXCoreAdapterPreference preferences[] = 
    {
        DXCoreAdapterPreference::Hardware,
        DXCoreAdapterPreference::HighPerformance
    };

    THROW_IF_FAILED(adapterList->Sort(_countof(preferences), preferences));

    std::vector<ComPtr<IDXCoreAdapter>> adapters;
    std::vector<std::string> adapterDescriptions;
    std::optional<uint32_t> firstAdapterMatchingNameFilter;
    bool foundPotentialAdapter = false;

    for (uint32_t i = 0; i < adapterList->GetAdapterCount(); i++)
    {
        ComPtr<IDXCoreAdapter> adapter;
        THROW_IF_FAILED(adapterList->GetAdapter(i, adapter.GetAddressOf()));

        size_t descriptionSize;
        THROW_IF_FAILED(adapter->GetPropertySize(
            DXCoreAdapterProperty::DriverDescription, 
            &descriptionSize
        ));

        std::string adapterDescription(descriptionSize, '\0');
        THROW_IF_FAILED(adapter->GetProperty(
            DXCoreAdapterProperty::DriverDescription, 
            descriptionSize, 
            adapterDescription.data()
        ));

        // Remove trailing null terminator written by DXCore.
        while (!adapterDescription.empty() && adapterDescription.back() == '\0')
        {
            adapterDescription.pop_back();
        }

        adapters.push_back(adapter);
        adapterDescriptions.push_back(adapterDescription);

        if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS) == useGraphicsAdapter)
        {
            foundPotentialAdapter = true;
            if (!firstAdapterMatchingNameFilter &&
                adapterDescription.find(adapterNameFilter) != std::string::npos)
            {
                firstAdapterMatchingNameFilter = i;
                std::cout << "Adapter[" << i << "] (SELECTED):\n";
            }
            else
            {
                std::cout << "Adapter[" << i << "]:\n";

            }
            std::cout << " " << adapterType << "\n";
	}
        else
        {
            std::cout << "Adapter[" << i << "]:\n";
            std::cout << " not " << adapterType << "\n";
        }
	std::cout << " " << adapterDescription << "\n";
    }

    if (!foundPotentialAdapter)
    {
        throw std::runtime_error("No compatible adapters found.");
    }

    if (!firstAdapterMatchingNameFilter)
    {
        throw std::invalid_argument("No adapters match the provided name filter.");
    }

    return { adapters[*firstAdapterMatchingNameFilter], featureLevel };
}

std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue(
    std::string adapterType,
    bool useGraphicsAdapter,
    std::string_view adapterNameFilter)
{
    using Microsoft::WRL::ComPtr;
    
    auto [adapter, featureLevel] = SelectAdapter(adapterType, useGraphicsAdapter, adapterNameFilter);

    ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), featureLevel, IID_PPV_ARGS(&d3d12Device)));

    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));

    D3D_FEATURE_LEVEL featureLevelsRequested[] = 
    {
        D3D_FEATURE_LEVEL_1_0_GENERIC,
        D3D_FEATURE_LEVEL_1_0_CORE,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_12_1
    };

    D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevelSupport = 
    {
        .NumFeatureLevels = _countof(featureLevelsRequested),
        .pFeatureLevelsRequested = featureLevelsRequested
    };

    THROW_IF_FAILED(d3d12Device->CheckFeatureSupport(
        D3D12_FEATURE_FEATURE_LEVELS,
        &featureLevelSupport,
        sizeof(featureLevelSupport)
    ));

    // The feature level returned by SelectAdapter is the MINIMUM feature level required for the adapter.
    // However, some adapters may support higher feature levels. For compatibility reasons, this sample
    // uses a direct queue for graphics-capable adapters that support feature levels > CORE.
    auto queueType = (featureLevelSupport.MaxSupportedFeatureLevel <= D3D_FEATURE_LEVEL_1_0_CORE) ? 
        D3D12_COMMAND_LIST_TYPE_COMPUTE : 
        D3D12_COMMAND_LIST_TYPE_DIRECT;

    D3D12_COMMAND_QUEUE_DESC queueDesc = 
    {
        .Type = queueType,
        .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0
    };

    ComPtr<ID3D12CommandQueue> commandQueue;
    THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

    return { std::move(dmlDevice), std::move(commandQueue) };
}

// Converts a pixel buffer to an NCHW tensor (batch size 1).
// Source: buffer of RGB pixels (HWC) using uint8 components.
// Target: buffer of RGB planes (CHW) using float32/float16 components.
template <typename T> 
void CopyPixelsToTensor(
    std::span<const std::byte> src, 
    std::span<std::byte> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels)
{
    std::span<T> dstT(reinterpret_cast<T*>(dst.data()), dst.size_bytes() / sizeof(T));

    for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
    {
        float r = static_cast<float>(src[pixelIndex * channels + 0]) / 255.0f;
        float g = static_cast<float>(src[pixelIndex * channels + 1]) / 255.0f;
        float b = static_cast<float>(src[pixelIndex * channels + 2]) / 255.0f;

        dstT[pixelIndex + 0 * height * width] = r;
        dstT[pixelIndex + 1 * height * width] = g;
        dstT[pixelIndex + 2 * height * width] = b;
    }
}

// Converts an NCHW tensor buffer (batch size 1) to a pixel buffer.
// Source: buffer of RGB planes (CHW) using float32/float16 components.
// Target: buffer of RGB pixels (HWC) using uint8 components.
template <typename T>
void CopyTensorToPixels(
    std::span<const std::byte> src, 
    std::span<BYTE> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels)
{
    std::span<const T> srcT(reinterpret_cast<const T*>(src.data()), src.size_bytes() / sizeof(T));

    for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
    {
        BYTE r = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 0 * height * width])) * 255.0f);
        BYTE g = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 1 * height * width])) * 255.0f);
        BYTE b = static_cast<BYTE>(std::max(0.0f, std::min(1.0f, (float)srcT[pixelIndex + 2 * height * width])) * 255.0f);

        dst[pixelIndex * channels + 0] = r;
        dst[pixelIndex * channels + 1] = g;
        dst[pixelIndex * channels + 2] = b;
    }
}

void FillNCHWBufferFromImageFilename(
    std::wstring_view filename,
    std::span<std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType,
    ChannelOrder bufferChannelOrder)
{
    using Microsoft::WRL::ComPtr;

    uint32_t bufferChannels = 0;
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;
    switch (bufferChannelOrder)
    {
        case ChannelOrder::RGB:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB;
            break;

        case ChannelOrder::BGR:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR;
            break;

        default:
            throw std::invalid_argument("Unsupported channel order");
    }

    uint32_t expectedBufferSizeInBytes = bufferChannels * bufferHeight * bufferWidth;
    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            expectedBufferSizeInBytes *= sizeof(float);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            expectedBufferSizeInBytes *= sizeof(half_float::half);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    if (tensorBuffer.size_bytes() < expectedBufferSizeInBytes)
    {
        throw std::invalid_argument("Provided buffer is too small");
    }

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmapDecoder> bitmapDecoder;
    THROW_IF_FAILED(wicFactory->CreateDecoderFromFilename(
        filename.data(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &bitmapDecoder
    ));

    ComPtr<IWICBitmapFrameDecode> bitmapFrame;
    THROW_IF_FAILED(bitmapDecoder->GetFrame(0, &bitmapFrame));

    ComPtr<IWICBitmapSource> bitmapSource = bitmapFrame;

    UINT imageWidth, imageHeight;
    THROW_IF_FAILED(bitmapFrame->GetSize(&imageWidth, &imageHeight));

    WICPixelFormatGUID imagePixelFormat;
    THROW_IF_FAILED(bitmapFrame->GetPixelFormat(&imagePixelFormat));

    if (imagePixelFormat != desiredImagePixelFormat)
    {
        Microsoft::WRL::ComPtr<IWICFormatConverter> formatConverter;
        THROW_IF_FAILED(wicFactory->CreateFormatConverter(&formatConverter));

        THROW_IF_FAILED(formatConverter->Initialize(
            bitmapFrame.Get(),
            desiredImagePixelFormat,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0f,
            WICBitmapPaletteTypeCustom
        ));

        bitmapSource = formatConverter;
    }

    if (imageWidth != bufferWidth || imageHeight != bufferHeight)
    {
        if (bufferWidth == bufferHeight)
        {
            // Most ML models take a square input. In this case, crop to the 
            // top-left square of the image to avoid stretching.
            INT minSide = static_cast<INT>(std::min(imageWidth, imageHeight));
            WICRect cropRect = 
            { 
                .X = 0, 
                .Y = 0, 
                .Width = minSide, 
                .Height = minSide 
            };
            
            ComPtr<IWICBitmapClipper> clipper;
            THROW_IF_FAILED(wicFactory->CreateBitmapClipper(&clipper));
            THROW_IF_FAILED(clipper->Initialize(bitmapSource.Get(), &cropRect));
            bitmapSource = clipper;
        }

        // Scale image to match the buffer size.
        ComPtr<IWICBitmapScaler> scaler;
        THROW_IF_FAILED(wicFactory->CreateBitmapScaler(&scaler));
        THROW_IF_FAILED(scaler->Initialize(
            bitmapSource.Get(),
            bufferWidth,
            bufferHeight,
            WICBitmapInterpolationModeHighQualityCubic
        ));

        bitmapSource = scaler;
    }

    // WIC doesn't support copying into "CHW" (planar) buffers except when
    // using Y'CbCr pixel formats. Instead, we copy into an interleaved
    // "HWC" buffer and then convert to "CHW" while simultaneously casting
    // data elements from uint8 to float.

    // Copy to HWC buffer with 8 bits (uint8) per channel element
    std::vector<std::byte> pixelBuffer(bufferHeight * bufferWidth * bufferChannels);
    WICRect pixelBufferRect = { 0, 0, static_cast<INT>(bufferWidth), static_cast<INT>(bufferHeight) };
    const uint32_t pixelBufferStride = bufferWidth * bufferChannels * sizeof(uint8_t);

    THROW_IF_FAILED(bitmapSource->CopyPixels(
        &pixelBufferRect, 
        pixelBufferStride, 
        pixelBuffer.size(), 
        reinterpret_cast<BYTE*>(pixelBuffer.data())
    ));

    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            CopyPixelsToTensor<float>(pixelBuffer, tensorBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            CopyPixelsToTensor<half_float::half>(pixelBuffer, tensorBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

void SaveNCHWBufferToImageFilename(
    std::wstring_view filename,
    std::span<const std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType,
    ChannelOrder bufferChannelOrder)
{
    using Microsoft::WRL::ComPtr;

    uint32_t bufferChannels = 0;
    WICPixelFormatGUID desiredImagePixelFormat = GUID_WICPixelFormatDontCare;
    switch (bufferChannelOrder)
    {
        case ChannelOrder::RGB:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppRGB;
            break;

        case ChannelOrder::BGR:
            bufferChannels = 3;
            desiredImagePixelFormat = GUID_WICPixelFormat24bppBGR;
            break;

        default:
            throw std::invalid_argument("Unsupported channel order");
    }

    uint32_t outputBufferSizeInBytes = bufferChannels * bufferHeight * bufferWidth;
    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            outputBufferSizeInBytes *= sizeof(float);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            outputBufferSizeInBytes *= sizeof(half_float::half);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    std::vector<BYTE> pixelBuffer(outputBufferSizeInBytes);

    switch (bufferDataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            CopyTensorToPixels<float>(tensorBuffer, pixelBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            CopyTensorToPixels<half_float::half>(tensorBuffer, pixelBuffer, bufferHeight, bufferWidth, bufferChannels);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
    }

    ComPtr<IWICImagingFactory> wicFactory;
    THROW_IF_FAILED(CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&wicFactory)
    ));

    ComPtr<IWICBitmap> bitmap;
    THROW_IF_FAILED(wicFactory->CreateBitmapFromMemory(
        bufferWidth,
        bufferHeight,
        desiredImagePixelFormat,
        bufferWidth * bufferChannels,
        pixelBuffer.size(),
        pixelBuffer.data(),
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
