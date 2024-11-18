// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

enum class ChannelOrder
{
    RGB,
    BGR,
};

std::tuple<Microsoft::WRL::ComPtr<IDMLDevice>, Microsoft::WRL::ComPtr<ID3D12CommandQueue>> CreateDmlDeviceAndCommandQueue(
    std::string adapterType,
    bool useGraphicsAdapter,
    std::string_view adapterNameFilter = ""
);

// Converts a pixel buffer to an NCHW tensor (batch size 1).
// Source: buffer of RGB pixels (HWC) using uint8 components.
// Target: buffer of RGB planes (CHW) using float32/float16 components.
template <typename T> 
void CopyPixelsToTensor(
    std::span<const std::byte> src, 
    std::span<std::byte> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels
);

// Converts an NCHW tensor buffer (batch size 1) to a pixel buffer.
// Source: buffer of RGB planes (CHW) using float32/float16 components.
// Target: buffer of RGB pixels (HWC) using uint8 components.
template <typename T>
void CopyTensorToPixels(
    std::span<const std::byte> src, 
    std::span<BYTE> dst,
    uint32_t height,
    uint32_t width,
    uint32_t channels
);

void FillNCHWBufferFromImageFilename(
    std::wstring_view filename,
    std::span<std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB
);

void SaveNCHWBufferToImageFilename(
    std::wstring_view filename,
    std::span<const std::byte> tensorBuffer,
    uint32_t bufferHeight,
    uint32_t bufferWidth,
    ONNXTensorElementDataType bufferDataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ChannelOrder bufferChannelOrder = ChannelOrder::RGB
);
