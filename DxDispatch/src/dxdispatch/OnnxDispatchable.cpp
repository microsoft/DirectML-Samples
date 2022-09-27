#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"
#include "OnnxParsers.h"

using Microsoft::WRL::ComPtr;

template<typename T>
using deleting_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

static Ort::Value CreateTensorFromResource(
    const OrtDmlApi* ortDmlApi,
    Ort::MemoryInfo const& memoryInformation,
    ID3D12Resource* d3dResource,
    gsl::span<const int64_t> tensorDimensions,
    ONNXTensorElementDataType elementDataType,
    void** dmlEpResourceWrapper)
{
    *dmlEpResourceWrapper = nullptr;

    void* dmlAllocatorResource;
    Ort::ThrowOnError(ortDmlApi->CreateGPUAllocationFromD3DResource(d3dResource, &dmlAllocatorResource));
    auto deleter = [&](void*) {ortDmlApi->FreeGPUAllocation(dmlAllocatorResource); };
    deleting_unique_ptr<void> dmlAllocatorResourceCleanup(dmlAllocatorResource, deleter);

    size_t tensorByteSize = static_cast<size_t>(d3dResource->GetDesc().Width);
    Ort::Value newValue(
        Ort::Value::CreateTensor(
            memoryInformation,
            dmlAllocatorResource,
            tensorByteSize,
            tensorDimensions.data(),
            tensorDimensions.size(),
            elementDataType
        )
    );
    *dmlEpResourceWrapper = dmlAllocatorResource;
    dmlAllocatorResourceCleanup.release();

    return newValue;
}

static ID3D12Resource* GetResourceFromModelBinding(
    const std::string& tensorName, 
    const Dispatchable::Bindings& bindings)
{
    auto binding = bindings.find(tensorName);
    if (binding == bindings.end())
    {
        throw std::runtime_error(fmt::format("Could not find binding for tensor '{}'", tensorName));
    }
    auto& bindingSources = binding->second;

    if (bindingSources.size() != 1)
    {
        throw std::invalid_argument("ONNX dispatchables' tensors must map to a single binding source.");
    }

    auto& bindingSource = bindingSources[0];

    if (bindingSource.counterResource != nullptr)
    {
        throw std::invalid_argument("ONNX dispatchables do not support counter resources in bindings.");
    }
    
    if (bindingSource.elementOffset != 0)
    {
        throw std::invalid_argument("ONNX dispatchables do not support binding offsets.");
    }

    return bindingSource.resource;
}

OnnxDispatchable::OnnxDispatchable(
    std::shared_ptr<Device> device, 
    const Model::OnnxDispatchableDesc& desc,
    const CommandLineArgs& args
    ) : m_device(device), m_desc(desc), m_args(args)
{
}

void OnnxDispatchable::Initialize()
{
    const OrtApi& ortApi = Ort::GetApi();
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));

    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DxDispatch"); // Note ORT_LOGGING_LEVEL_VERBOSE is useful too.
    
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Note ORT_ENABLE_BASIC is useful for debugging.
 
    for (auto& freeDimOverride : m_args.GetOnnxFreeDimensionNameOverrides())
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverrideByName(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    for (auto& freeDimOverride : m_args.GetOnnxFreeDimensionDenotationOverrides())
    {
        Ort::ThrowOnError(ortApi.AddFreeDimensionOverride(sessionOptions, freeDimOverride.first.c_str(), freeDimOverride.second));
    }

    const OrtDmlApi* ortDmlApi;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, m_device->DML(), m_device->GetCommandQueue()));

    m_session = Ort::Session(ortEnvironment, m_desc.sourcePath.wstring().c_str(), sessionOptions);
    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);
}

void OnnxDispatchable::Bind(const Bindings& bindings)
{
    m_ioBindings->ClearBoundInputs();
    m_ioBindings->ClearBoundOutputs();
    m_inputTensors.clear();
    m_tensorWrappers.clear();
    m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo memoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::Allocator deviceAllocator(*m_session, memoryInformation);

    auto inputCount = m_session->GetInputCount();
    m_inputTensors.resize(inputCount);

    // Bind input tensors.
    for (size_t tensorIndex = 0; tensorIndex < inputCount; ++tensorIndex)
    {
        std::string tensorName = OnnxParsers::GetTensorName(tensorIndex, *m_session, /*isInputTensor*/true);
        Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(tensorIndex);
        if (typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR)
        {
            throw std::runtime_error(fmt::format("Unknown binding type for '{}'", tensorName));
        }

        Ort::Unowned<Ort::TensorTypeAndShapeInfo> shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType tensorDataType = shapeInfo.GetElementType();
        if (!OnnxParsers::IsSupportedOnnxTensorElementDataType(tensorDataType))
        {
            throw std::runtime_error("Unsupported tensor data type");
        }

        // Convert free dimensions (-1) to their minimum positive size (1).
        std::vector<int64_t> tensorShape = shapeInfo.GetShape();
        for (auto& dim : tensorShape)
        {
            dim = std::abs(dim);
        }

        auto resource = GetResourceFromModelBinding(tensorName, bindings);

        std::optional<Ort::Value>& tensor = m_inputTensors[tensorIndex];

        // Create an ORT tensor from the existing D3D resource.
        Microsoft::WRL::ComPtr<IUnknown> resourceWrapper;
        tensor = CreateTensorFromResource(
            m_ortDmlApi, 
            memoryInformation, 
            resource,
            tensorShape,
            tensorDataType,
            &resourceWrapper);

        m_tensorWrappers.push_back(std::move(resourceWrapper));

        // Bind the tensor.
        m_ioBindings->BindInput(tensorName.c_str(), *tensor);
    }

    // Bind outputs by name only; let the execution provider allocate the output resource.
    for (size_t tensorIndex = 0; tensorIndex < m_session->GetOutputCount(); ++tensorIndex)
    {
        std::string tensorName = OnnxParsers::GetTensorName(tensorIndex, *m_session, /*isInputTensor*/false);
        m_ioBindings->BindOutput(tensorName.c_str(), *m_memoryInfo);
    }
}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args)
{
    Ort::RunOptions runOptions;
    m_session->Run(runOptions, *m_ioBindings);
    m_ioBindings->SynchronizeOutputs();

    // TODO: need to copy to model output resource if one is bound.
}