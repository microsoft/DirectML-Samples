#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "OnnxDispatchable.h"

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


static std::string GetTensorName(size_t index, Ort::Session const& session, bool isInput)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = isInput ? session.GetInputNameAllocated(index, allocator) : session.GetOutputNameAllocated(index, allocator);
    std::string returnName(name.get());
    return returnName;
}

struct DataTypeInfo
{
    ONNXTensorElementDataType onnxDataType;
    DML_TENSOR_DATA_TYPE dmlDataType;
    uint32_t sizeInBytes;
};

static DataTypeInfo GetDataTypeInfo(ONNXTensorElementDataType dataType)
{
    DataTypeInfo info = {};
    info.onnxDataType = dataType;
    info.dmlDataType = DML_TENSOR_DATA_TYPE_UNKNOWN;

    switch (dataType)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT8;
        info.sizeInBytes = 1;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT8;
        info.sizeInBytes = 1;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        info.sizeInBytes = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        info.sizeInBytes = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_UINT64;
        info.sizeInBytes = 8;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_INT64;
        info.sizeInBytes = 8;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        info.dmlDataType = DML_TENSOR_DATA_TYPE_FLOAT64;
        info.sizeInBytes = 8;
        break;
    }

    return info;
}

const char* GetOnnxTensorTypeString(ONNXTensorElementDataType dataType)
{
    switch (dataType)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "FLOAT";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "STRING";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "BOOL";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "DOUBLE";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "UINT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "COMPLEX64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "COMPLEX128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "BFLOAT16";
        default: return "UNDEFINED";
    }
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

    OrtLoggingLevel loggingLevel = m_args.GetOnnxLoggingLevel() ? 
        static_cast<OrtLoggingLevel>(*m_args.GetOnnxLoggingLevel()) : 
        static_cast<OrtLoggingLevel>(m_desc.loggingLevel);

    m_environment = Ort::Env(loggingLevel, "DxDispatch");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();

    GraphOptimizationLevel graphOptimizationLevel = m_args.GetOnnxGraphOptimizationLevel() ? 
        static_cast<GraphOptimizationLevel>(*m_args.GetOnnxGraphOptimizationLevel()) :
        static_cast<GraphOptimizationLevel>(m_desc.graphOptimizationLevel);

    sessionOptions.SetGraphOptimizationLevel(graphOptimizationLevel);
 
    using DimOverridesList = std::initializer_list<gsl::span<const std::pair<std::string, uint32_t>>>;

    // Dimension name overrides (command-line overrides take priority over JSON values)
    for (auto& overrides : DimOverridesList{ m_desc.freeDimNameOverrides, m_args.GetOnnxFreeDimensionNameOverrides() })
    {
        for (auto& override : overrides)
        {
            Ort::ThrowOnError(ortApi.AddFreeDimensionOverrideByName(sessionOptions, override.first.c_str(), override.second));
        }
    }

    // Denotation overrides (command-line overrides take priority over JSON values)
    for (auto& overrides : DimOverridesList{ m_desc.freeDimDenotationOverrides, m_args.GetOnnxFreeDimensionDenotationOverrides() })
    {
        for (auto& override : overrides)
        {
            Ort::ThrowOnError(ortApi.AddFreeDimensionOverride(sessionOptions, override.first.c_str(), override.second));
        }
    }

    using ConfigEntriesList = std::initializer_list<gsl::span<const std::pair<std::string, std::string>>>;

    // SessionOptions config entries (command-line entries take priority over JSON values)
    for (auto& configEntries : ConfigEntriesList{ m_desc.sessionOptionsConfigEntries, m_args.GetOnnxSessionOptionConfigEntries() })
    {
        for (auto& configEntry : configEntries)
        {
            Ort::ThrowOnError(ortApi.AddSessionConfigEntry(sessionOptions, configEntry.first.c_str(), configEntry.second.c_str()));
        }
    }

    const OrtDmlApi* ortDmlApi;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));
    Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, m_device->DML(), m_device->GetCommandQueue()));

    m_session = Ort::Session(*m_environment, m_desc.sourcePath.wstring().c_str(), sessionOptions);
    m_ioBindings = Ort::IoBinding::IoBinding(*m_session);
}

void OnnxDispatchable::Bind(const Bindings& jsonBindings, uint32_t iteration)
{
    // Early exit for all iterations after the first. Bindings are cached in m_ioBindings.
    if (iteration > 0)
    {
        return;
    }

    // Binding behavior is complex. The motivation behind these rules:
    // 1. Be flexible in running models without explicit JSON bindings (most likely profiling; generate either CPU or DX resources to unblock execution).
    // 2. Be strict when using explicit JSON bindings (fail if the binding doesn't make sense).
    // While it may be possible to ignore an invalid binding in JSON to unblock execution, this is most likely not what the user wants.

    m_ioBindings->ClearBoundInputs();
    m_ioBindings->ClearBoundOutputs();
    m_mergedBindings.clear();

    Ort::MemoryInfo cpuMemoryInformation = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo dmlMemoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

    for (int bindingPass = 0; bindingPass < 2; ++bindingPass)
    {
        const bool isInputTensor = (bindingPass == 0);
        const size_t tensorCount = isInputTensor ? m_session->GetInputCount() : m_session->GetOutputCount();

        for (size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex)
        {
            TensorBinding binding = {};
            auto tensorName = GetTensorName(tensorIndex, *m_session, isInputTensor);
            binding.name = tensorName;
            binding.isInput = isInputTensor;

            Ort::TypeInfo typeInfo = isInputTensor ? m_session->GetInputTypeInfo(tensorIndex) : m_session->GetOutputTypeInfo(tensorIndex);

            bool isDmlSupportedType = false;

            if (typeInfo.GetONNXType() == ONNXType::ONNX_TYPE_TENSOR)
            {
                auto shapeInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto dataTypeInfo = GetDataTypeInfo(shapeInfo.GetElementType());
                isDmlSupportedType = dataTypeInfo.dmlDataType != DML_TENSOR_DATA_TYPE_UNKNOWN;

                // Get shape stored in ONNX model.
                binding.shape = shapeInfo.GetShape();
                binding.dataType = shapeInfo.GetElementType();
                
                // Check if the tensor shape is static or dynamic, which determines if resources can be preallocated.
                bool tensorShapeHasFreeDimensions = false;
                for (int64_t& dimSize : binding.shape)
                {
                    // Dimensions that aren't statically known/inferrable are "free dimensions" with size -1. 
                    if (dimSize == -1)
                    {
                        // Try fixing any free dimensions that appear on *inputs* to size 1, which may make the graph valid
                        // for execution (e.g. dim represents batch size). This trick cannot be done for outputs, since their 
                        // free dimensions may correspond to symbolic dimensions that are only known at runtime. Tensors with 
                        // free dimensions have "dynamic shapes" and cannot be preallocated; their total size is unknown.
                        if (isInputTensor)
                        {
                            dimSize = 1;
                        }
                        else
                        {
                            tensorShapeHasFreeDimensions = true;
                        }
                    }
                }

                // Override tensorShape with the one specified in JSON, if any.
                auto jsonBinding = jsonBindings.find(tensorName);
                ID3D12Resource* jsonResource = nullptr;
                if (jsonBinding != jsonBindings.end())
                {
                    auto& bindingShape = jsonBinding->second[0].shape;
                    if (!bindingShape.empty())
                    {
                        binding.shape = bindingShape;
                        tensorShapeHasFreeDimensions = false;
                    }

                    // The JSON binding may also include the name of a JSON resource.
                    binding.resource = jsonBinding->second[0].resource;
                }

                // Override tensorShape with the one specified on the command line, if any.
                auto commandLineBindingShape = m_args.GetOnnxBindingShapes().find(tensorName);
                if (commandLineBindingShape != m_args.GetOnnxBindingShapes().end())
                {
                    auto& bindingShape = commandLineBindingShape->second;
                    if (!bindingShape.empty())
                    {
                        binding.shape = bindingShape;
                        tensorShapeHasFreeDimensions = false;
                    }
                }

                // Attempt to preallocate/wrap resources where possible.
                if (binding.resource)
                {
                    // If a DX resource was explicitly bound in the JSON model, then it has already been allocated.
                    // Simply wrap the existing DX resource as an OrtValue.
                    if (!tensorShapeHasFreeDimensions)
                    {
                        if (isDmlSupportedType)
                        {
                            binding.ortValue = CreateTensorFromResource(
                                m_ortDmlApi,
                                dmlMemoryInformation,
                                binding.resource.Get(),
                                binding.shape,
                                dataTypeInfo.onnxDataType,
                                &binding.wrapper
                            );
                            binding.resourceType = "explicit (DirectX)";
                        }
                        else
                        {
                            throw std::invalid_argument(fmt::format(
                                "Binding resource '{}' to tensor '{}' is invalid because the ONNX model tensor's data type is not supported by DML.",
                                jsonBinding->second[0].resourceDesc->name, 
                                tensorName
                            ));
                        }
                    }
                    else
                    {
                        throw std::invalid_argument(fmt::format("Binding resource '{}' to tensor '{}' is invalid because the tensor shape is not static.", 
                            jsonBinding->second[0].resourceDesc->name, 
                            tensorName
                        ));
                    }
                }
                else
                {
                    // Attempt to lazily create resources/bindings for tensors not bound in the JSON model.
                    // Only tensors with static shapes can be preallocated.
                    if (!tensorShapeHasFreeDimensions)
                    {
                        if (isDmlSupportedType)
                        {
                            // Convert int64_t tensorShape to uint32_t for DML
                            std::vector<uint32_t> tensorShapeUint32;
                            for (int64_t dimSize : binding.shape)
                            {
                                tensorShapeUint32.push_back(static_cast<uint32_t>(std::abs(dimSize)));
                            }

                            // Scalars have empty shapes. DML stores these as [1].
                            if (tensorShapeUint32.empty())
                            {
                                tensorShapeUint32.push_back(1);
                            }

                            UINT64 resourceSize = DMLCalcBufferTensorSize(
                                dataTypeInfo.dmlDataType,
                                tensorShapeUint32.size(),
                                tensorShapeUint32.data(),
                                nullptr);

                            binding.resource = m_device->CreateBuffer(
                                resourceSize,
                                m_args.GetCustomHeaps() ? D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE : D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                m_args.GetCustomHeaps() ? D3D12_MEMORY_POOL_L0 : D3D12_MEMORY_POOL_UNKNOWN);

                            binding.ortValue = CreateTensorFromResource(
                                m_ortDmlApi,
                                dmlMemoryInformation,
                                binding.resource.Get(),
                                binding.shape,
                                dataTypeInfo.onnxDataType,
                                &binding.wrapper
                            );

                            binding.resourceType = "implicit (DirectX)";
                        }
                        else
                        {
                            // Preallocate as a CPU resource.
                            binding.ortValue = Ort::Value::CreateTensor(
                                static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions()), 
                                binding.shape.data(),
                                binding.shape.size(), 
                                dataTypeInfo.onnxDataType
                            );

                            binding.resourceType = "implicit (CPU)";
                        }
                    }
                }
            }

            // Finally, set the ORT IO binding for the tensor.
            if (isInputTensor)
            {
                if (binding.ortValue)
                {
                    m_ioBindings->BindInput(tensorName.c_str(), *binding.ortValue);
                }
                else
                {
                    // Only non-tensor inputs should remain unbound.
                    assert(typeInfo.GetONNXType() != ONNXType::ONNX_TYPE_TENSOR);
                }
            }
            else
            {
                assert(!isInputTensor);

                if (binding.ortValue)
                {
                    m_ioBindings->BindOutput(tensorName.c_str(), *binding.ortValue);
                }
                else
                {
                    // Let the execution provider allocate the output.
                    m_ioBindings->BindOutput(tensorName.c_str(), isDmlSupportedType ? dmlMemoryInformation : cpuMemoryInformation);
                    binding.resourceType = isDmlSupportedType ? "deferred (DirectX)" : "deferred (CPU)";
                }
            }

            m_mergedBindings.emplace_back(std::move(binding));
        }
    }

    if (m_args.PrintVerboseOnnxBindingInfo())
    {
        for (auto& binding : m_mergedBindings)
        {
            LogInfo(fmt::format("{} Tensor '{}':", (binding.isInput ? "Input" : "Output"), binding.name));
            LogInfo(fmt::format("  Resource  = {}", binding.resourceType));
            LogInfo(fmt::format("  Data Type = {}", GetOnnxTensorTypeString(binding.dataType)));
            std::string shapeString = "[";
            for (size_t i = 0; i < binding.shape.size(); i++)
            {
                shapeString += std::to_string(binding.shape[i]);
                if (i < binding.shape.size() - 1)
                {
                    shapeString += ",";
                }
            }
            shapeString += "]";
            LogInfo(fmt::format("  Shape     = {}", shapeString));
            LogInfo("");
        }
    }
}

void OnnxDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration)
{
    PIXBeginEvent(m_device->GetCommandList(), PIX_COLOR(255, 255, 0), "ONNX: '%s'", args.dispatchableName.c_str());
    m_device->RecordTimestamp();
    m_device->ExecuteCommandList();

    Ort::RunOptions runOptions;
    for (uint32_t i = 0; i < m_args.DispatchRepeat(); i++)
        m_session->Run(runOptions, *m_ioBindings);

    m_device->RecordTimestamp();
    PIXEndEvent(m_device->GetCommandList());
    m_device->ExecuteCommandList();
}

void OnnxDispatchable::Wait()
{
    m_ioBindings->SynchronizeOutputs();
}