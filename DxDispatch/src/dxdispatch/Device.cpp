#include "pch.h"
#include "Device.h"

using Microsoft::WRL::ComPtr;

// {0059DA69-B561-43D9-A39B-3355074B1082}
static const GUID PIX_EVAL_CAPTURABLE_WORK_GUID =
{ 0x59da69, 0xb561, 0x43d9, { 0xa3, 0x9b, 0x33, 0x55, 0x7, 0x4b, 0x10, 0x82 } };

// Callback to log D3D12/DirectML debug messages.
#ifdef _GAMING_XBOX
static bool DebugMessageCallback(void* context, void* commandList, DWORD messageId, const CHAR* message)
{
    LogError(message);
    return true;
}
#else
static void __stdcall DebugMessageCallback(D3D12_MESSAGE_CATEGORY cat, D3D12_MESSAGE_SEVERITY sev, D3D12_MESSAGE_ID id, LPCSTR message, void* context)
{
    LogError(message);
}
#endif

Device::Device(
    IAdapter* adapter, 
    bool debugLayersEnabled, 
    D3D12_COMMAND_LIST_TYPE commandListType, 
    uint32_t dispatchRepeat,
    bool uavBarrierAfterDispatch,
    bool aliasingBarrierAfterDispatch,
    std::shared_ptr<PixCaptureHelper> pixCaptureHelper,
    std::shared_ptr<D3d12Module> d3dModule,
    std::shared_ptr<DmlModule> dmlModule
    ) : m_pixCaptureHelper(std::move(pixCaptureHelper)),
        m_d3dModule(std::move(d3dModule)),
        m_dmlModule(std::move(dmlModule)),
        m_dispatchRepeat(dispatchRepeat)
{
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = debugLayersEnabled ? DML_CREATE_DEVICE_FLAG_DEBUG : DML_CREATE_DEVICE_FLAG_NONE;

#ifdef _GAMING_XBOX
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version = D3D12_SDK_VERSION;

    if (debugLayersEnabled)
    {
        params.ProcessDebugFlags = D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED;
    }
    if (m_pixCaptureHelper->GetPixCaptureType() != PixCaptureType::None)
    {
        // Enable the instrumented driver.
        params.ProcessDebugFlags = D3D12XBOX_PROCESS_DEBUG_FLAG_INSTRUMENTED;
    }

    params.GraphicsCommandQueueRingSizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.GraphicsScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.ComputeScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);

    THROW_IF_FAILED(D3D12XboxCreateDevice(adapter, &params, IID_GRAPHICS_PPV_ARGS(m_d3d.ReleaseAndGetAddressOf())));

    if (debugLayersEnabled)
    {
        m_d3d->SetDebugCallbackX(DebugMessageCallback, /*context*/nullptr);
    }
#else // !_GAMING_XBOX
    if (debugLayersEnabled)
    {
        ComPtr<ID3D12Debug3> d3dDebug;
        THROW_IF_FAILED(m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&d3dDebug)));
        d3dDebug->EnableDebugLayer();
        d3dDebug->SetEnableGPUBasedValidation(true);
    }

    THROW_IF_FAILED(m_d3dModule->CreateDevice(
        adapter, 
        D3D_FEATURE_LEVEL_1_0_CORE,
        IID_PPV_ARGS(&m_d3d)));

    if (debugLayersEnabled)
    {
        THROW_IF_FAILED(m_d3d->QueryInterface(m_infoQueue.GetAddressOf()));
        DWORD callbackCookie = 0;
        m_infoQueue->RegisterMessageCallback(
            DebugMessageCallback, 
            D3D12_MESSAGE_CALLBACK_FLAG_NONE, 
            nullptr, 
            &callbackCookie);
    }
#endif // !_GAMING_XBOX

    THROW_IF_FAILED(m_d3d->CreateFence(
        0, 
        D3D12_FENCE_FLAG_NONE, 
        IID_GRAPHICS_PPV_ARGS(m_fence.ReleaseAndGetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = commandListType;
    THROW_IF_FAILED(m_d3d->CreateCommandQueue(
        &queueDesc, 
        IID_GRAPHICS_PPV_ARGS(m_queue.ReleaseAndGetAddressOf())));
    m_commandListType = queueDesc.Type;

    THROW_IF_FAILED(m_dmlModule->CreateDevice1(
        m_d3d.Get(), 
        dmlCreateDeviceFlags, 
        DML_FEATURE_LEVEL_5_0, 
        IID_PPV_ARGS(&m_dml)));

    THROW_IF_FAILED(m_d3d->CreateCommandAllocator(
        m_commandListType,
        IID_GRAPHICS_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3d->CreateCommandList(
        0,
        m_commandListType,
        m_commandAllocator.Get(),
        nullptr,
        IID_GRAPHICS_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_dml->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));

    D3D12_QUERY_HEAP_DESC queryHeapDesc;
    queryHeapDesc.Count = timestampCapacity;
    queryHeapDesc.NodeMask = 0;
    queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

    THROW_IF_FAILED(m_d3d->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_timestampHeap)));

    m_pixCaptureHelper->Initialize(m_queue.Get());

    if (uavBarrierAfterDispatch)
    {
        m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    }
    if (aliasingBarrierAfterDispatch)
    {
        m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));
    }
}

Device::~Device()
{
}

ComPtr<ID3D12Resource> Device::CreateDefaultBuffer(
    uint64_t sizeInBytes, 
    D3D12_RESOURCE_FLAGS resourceFlags, 
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps, 
        heapFlags, 
        &resourceDesc, 
        D3D12_RESOURCE_STATE_COMMON, 
        nullptr, 
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateBuffer(
    uint64_t sizeInBytes,
    D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
    D3D12_MEMORY_POOL memoryPoolPreference,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    if (cpuPageProperty == D3D12_CPU_PAGE_PROPERTY_UNKNOWN && memoryPoolPreference == D3D12_MEMORY_POOL_UNKNOWN)
    {
        return CreateDefaultBuffer(sizeInBytes, resourceFlags, alignment, heapFlags);
    }

    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(cpuPageProperty, memoryPoolPreference);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateUploadBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

ComPtr<ID3D12Resource> Device::CreateReadbackBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags)
{
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

    ComPtr<ID3D12Resource> resource;
    THROW_IF_FAILED(m_d3d->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

    return resource;
}

void Device::WaitForGpuWorkToComplete()
{
    uint64_t nextFenceValue = m_fence->GetCompletedValue() + 1;
    THROW_IF_FAILED(m_queue->Signal(m_fence.Get(), nextFenceValue));
    THROW_IF_FAILED(m_fence->SetEventOnCompletion(nextFenceValue, nullptr));
}

void Device::RecordInitialize(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable)
{
    m_commandRecorder->RecordDispatch(m_commandList.Get(), dispatchable, bindingTable);
}

void Device::RecordDispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* bindingTable)
{
    RecordTimestamp();

    for (uint32_t i = 0; i < m_dispatchRepeat; i++)
    {
        m_commandRecorder->RecordDispatch(m_commandList.Get(), dispatchable, bindingTable);
        if (!m_postDispatchBarriers.empty())
            m_commandList->ResourceBarrier(m_postDispatchBarriers.size(), m_postDispatchBarriers.data());
    }

    RecordTimestamp();
}

void Device::RecordDispatch(const char* name, uint32_t threadGroupX, uint32_t threadGroupY, uint32_t threadGroupZ)
{
    PIXBeginEvent(m_commandList.Get(), PIX_COLOR(255, 255, 0), "HLSL: '%s'", name);
    RecordTimestamp();
    
    for (uint32_t i = 0; i < m_dispatchRepeat; i++)
    {
        m_commandList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
        if (!m_postDispatchBarriers.empty())
            m_commandList->ResourceBarrier(m_postDispatchBarriers.size(), m_postDispatchBarriers.data());
    }

    RecordTimestamp();
    PIXEndEvent(m_commandList.Get());
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::CreateBuffer(
    uint64_t totalSize, 
    gsl::span<const std::byte> data, 
    std::wstring_view name,
    D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
    D3D12_MEMORY_POOL memoryPoolPreference
)
{
    if (data.size() > totalSize)
    {
        throw std::invalid_argument("Attempting to upload more data than the size of the buffer");
    }

    auto buffer = CreateBuffer(totalSize, cpuPageProperty, memoryPoolPreference);

    if (!name.empty())
    {
        buffer->SetName(name.data());
    }

    if (data.empty())
    {
        // No need to create an upload resource if the source data is empty.
        return buffer;
    }

    auto uploadBuffer = CreateUploadBuffer(totalSize);
    uploadBuffer->SetName(L"Device::Upload");
    {
        void* uploadBufferData = nullptr;
        THROW_IF_FAILED(uploadBuffer->Map(0, nullptr, &uploadBufferData));
        memcpy(uploadBufferData, data.data(), data.size());
        uploadBuffer->Unmap(0, nullptr);
    }

    {
        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                buffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
        };
        m_commandList->ResourceBarrier(_countof(barriers), barriers);
    }

    m_commandList->CopyResource(buffer.Get(), uploadBuffer.Get());

    {
        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                buffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        m_commandList->ResourceBarrier(_countof(barriers), barriers);
    }

    m_temporaryResources.push_back(std::move(uploadBuffer));

    return buffer;
}

std::vector<std::byte> Device::Download(Microsoft::WRL::ComPtr<ID3D12Resource> buffer)
{
    auto readbackBuffer = CreateReadbackBuffer(buffer->GetDesc().Width);
    readbackBuffer->SetName(L"Device::Download");

    {
        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                buffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
        };
        m_commandList->ResourceBarrier(_countof(barriers), barriers);
    }

    m_commandList->CopyResource(readbackBuffer.Get(), buffer.Get());

    {
        D3D12_RESOURCE_BARRIER barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                buffer.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        m_commandList->ResourceBarrier(_countof(barriers), barriers);
    }

    ExecuteCommandListAndWait();

    std::vector<std::byte> outputBuffer(buffer->GetDesc().Width);
    {
        size_t dataSize = buffer->GetDesc().Width;
        CD3DX12_RANGE readRange(0, gsl::narrow<size_t>(dataSize));
        void* readbackBufferData = nullptr;
        THROW_IF_FAILED(readbackBuffer->Map(0, &readRange, &readbackBufferData));
        memcpy(outputBuffer.data(), readbackBufferData, dataSize);
        readbackBuffer->Unmap(0, nullptr);
    }

    m_temporaryResources.push_back(std::move(readbackBuffer));

    return outputBuffer;
}

void Device::ExecuteCommandList()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}

void Device::ExecuteCommandListAndWait()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
    WaitForGpuWorkToComplete();
    THROW_IF_FAILED(m_d3d->GetDeviceRemovedReason());
    THROW_IF_FAILED(m_commandAllocator->Reset());
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

    m_temporaryResources.clear();
}

void Device::RecordTimestamp()
{
    m_commandList->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_timestampHeadIndex);
    m_timestampHeadIndex = (m_timestampHeadIndex + 1) % Device::timestampCapacity;
    if (m_timestampCount < timestampCapacity)
    {
        m_timestampCount++;
    }
}

std::vector<uint64_t> Device::ResolveTimestamps()
{
    assert(m_timestampCount <= timestampCapacity);

    auto timestampReadbackBuffer = CreateReadbackBuffer(sizeof(uint64_t) * m_timestampCount);

    m_commandList->ResolveQueryData(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, m_timestampCount, timestampReadbackBuffer.Get(), 0);
    ExecuteCommandListAndWait();

    void* pData = nullptr;
    D3D12_RANGE readRange = { 0, sizeof(uint64_t) * m_timestampCount };
    timestampReadbackBuffer->Map(0, &readRange, &pData);

    std::vector<uint64_t> timestamps;
    const uint64_t* pTimestamps = reinterpret_cast<uint64_t*>(pData);
    timestamps.insert(timestamps.end(), pTimestamps, pTimestamps + m_timestampCount);

    m_timestampHeadIndex = 0;
    m_timestampCount = 0;

    return timestamps;
}

std::vector<double> Device::ResolveTimingSamples()
{
    std::vector<uint64_t> timestamps = ResolveTimestamps();

    uint64_t frequency;
    THROW_IF_FAILED(m_queue->GetTimestampFrequency(&frequency));

    std::vector<double> samples(timestamps.size() / 2);

    for (uint32_t i = 0; i < samples.size(); ++i) 
    {
        uint64_t timestampDelta = (timestamps[2 * i + 1] - timestamps[2 * i]) * 1000;
        samples[i] = double(timestampDelta) / frequency / m_dispatchRepeat;
    }

    return samples;
}

#ifndef DXCOMPILER_NONE
void Device::EnsureDxcInterfaces()
{
#if defined(_GAMING_XBOX) || defined(_AMD64_)
    if (!m_dxcCompiler)
    {
        // Lazily create DXC compiler and helpers.
        THROW_IF_FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils)));
        THROW_IF_FAILED(m_dxcUtils->CreateDefaultIncludeHandler(&m_dxcIncludeHandler));
        THROW_IF_FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler)));
    }
#endif
}

IDxcUtils* Device::GetDxcUtils()
{
    EnsureDxcInterfaces();
    return m_dxcUtils.Get();
}

IDxcIncludeHandler* Device::GetDxcIncludeHandler()
{
    EnsureDxcInterfaces();
    return m_dxcIncludeHandler.Get();
}

IDxcCompiler3* Device::GetDxcCompiler()
{
    EnsureDxcInterfaces();
    return m_dxcCompiler.Get();
}
#endif // !DXCOMPILER_NONE

void Device::ClearShaderCaches()
{
    struct
    {
        D3D12_SHADER_CACHE_KIND_FLAGS kind;
        const char* name;
    } caches[] =
    {
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER" },
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS" },
        { D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED, "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED" },
        { D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED, "D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED" },
    };

    for (auto cache : caches)
    {
        auto hr = m_d3d->ShaderCacheControl(cache.kind, D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR);
        if (FAILED(hr))
        {
            LogInfo(fmt::format("Clearing {} failed. Do you have developer mode enabled?", cache.name));
        }
        else
        {
            LogInfo(fmt::format("Clearing {} succeeded.", cache.name));
        }
    }
}

/*static*/ uint32_t Device::GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_INT8:
        case DML_TENSOR_DATA_TYPE_UINT8:
            return 1;

        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_INT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
            return 2;

        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_INT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
            return 4;
    
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_INT64:
        case DML_TENSOR_DATA_TYPE_UINT64:
            return 8;

        default:
            throw std::invalid_argument("Unknown data type");
    }
}

/*static*/ DXGI_FORMAT Device::GetDxgiFormatFromDmlTensorDataType(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: return DXGI_FORMAT_R16_FLOAT;
    case DML_TENSOR_DATA_TYPE_FLOAT32: return DXGI_FORMAT_R32_FLOAT;
    // case DML_TENSOR_DATA_TYPE_FLOAT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_UINT8: return DXGI_FORMAT_R8_UINT;
    case DML_TENSOR_DATA_TYPE_UINT16: return DXGI_FORMAT_R16_UINT;
    case DML_TENSOR_DATA_TYPE_UINT32: return DXGI_FORMAT_R32_UINT;
    // case DML_TENSOR_DATA_TYPE_UINT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_INT8: return DXGI_FORMAT_R8_SINT;
    case DML_TENSOR_DATA_TYPE_INT16: return DXGI_FORMAT_R16_SINT;
    case DML_TENSOR_DATA_TYPE_INT32: return DXGI_FORMAT_R32_SINT;
    // case DML_TENSOR_DATA_TYPE_INT64: no DXGI type exists
    default: throw std::invalid_argument(fmt::format("No DXGI_FORMAT exists for given data type."));
    }
}