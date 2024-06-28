#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlSerializedGraphDispatchable.h"
#include "../DirectMLHelpers/DmlGraphHelper.h"

using Microsoft::WRL::ComPtr;

DmlSerializedGraphDispatchable::DmlSerializedGraphDispatchable(
    std::string_view name, 
    std::shared_ptr<Device> device, 
    const Model::DmlSerializedGraphDispatchableDesc& desc,
    const Dispatchable::Bindings& initBindings) :
          m_name(name), m_device(device), m_desc(desc), m_initBindings(initBindings)
{
}

void DmlSerializedGraphDispatchable::BuildGraph() 
{
    // Convert to Public Graph Description
    StackAllocator<1024> allocator;
    DML_GRAPH_DESC dmlGraphDesc = {};
    std::vector<Microsoft::WRL::ComPtr<IDMLOperator>> dmlOperators;
    std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes;
    std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges;
    std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges;
    std::vector<std::vector<std::uint8_t>> constDataVectors;

    // Convert the graph description
    ConvertGraphDesc<1024>(
        m_desc.desc,
        m_desc.desc.InputCount,
        m_desc.desc.OutputCount,
        m_device->DML(),
        allocator,
        nullptr, 
        nullptr, 
        dmlGraphDesc,
        dmlOperators,
        dmlGraphNodes,
        dmlInputEdges,
        dmlOutputEdges,
        dmlIntermediateEdges);

    //Compile the graph
    THROW_IF_FAILED(m_device->DML()->CompileGraph(
        &dmlGraphDesc,
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(&m_graphCompiled)));
}

struct BindingData
{
    std::vector<DML_BUFFER_BINDING> bufferBindings;
    std::vector<DML_BINDING_DESC> bindingDescs;
};

void FillBindingData(
    const std::vector<Model::DmlSerializedGraphDispatchableDesc::BindPoint>& bindPoints,
    const Dispatchable::Bindings* initializeBindings,
    const Dispatchable::Bindings* executeBindings,
    BindingData& bindingData,
    bool bindingForInitialization = false)
{
    const Dispatchable::Bindings& bindings = bindingForInitialization ? *initializeBindings : *executeBindings;

    uint32_t totalResourceCount = 0;
    for (size_t i = 0; i < bindPoints.size(); i++) { totalResourceCount += bindPoints[i].resourceCount; }

    bindingData.bufferBindings.resize(totalResourceCount);
    bindingData.bindingDescs.resize(totalResourceCount);

    size_t bufferIndex = 0;

    for (size_t i = 0; i < bindPoints.size(); i++)
    {
        auto bindPointName = bindPoints[i].name;
        auto bindingIterator = bindings.find(bindPointName);

        if (bindingIterator == bindings.end())
        {
            if (bindPoints[i].required && !bindingForInitialization)
            {
                if (!initializeBindings || initializeBindings->find(bindPointName) == initializeBindings->end())
                {
                    throw std::invalid_argument(fmt::format("Nothing bound for required tensor '{}'.", bindPointName));
                }
            }

            for (size_t j = 0; j < bindPoints[i].resourceCount; j++)
            {
                bindingData.bufferBindings[bufferIndex].Buffer = nullptr;
                bindingData.bufferBindings[bufferIndex].Offset = 0;
                bindingData.bufferBindings[bufferIndex].SizeInBytes = 0;
                bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_NONE;
                bindingData.bindingDescs[bufferIndex].Desc = nullptr;
                bufferIndex++;
            }
        }
        else
        {
            auto& sources = bindingIterator->second;

            if (bindPoints[i].resourceCount != sources.size())
            {
                throw std::invalid_argument(fmt::format(
                    "Bind point '{}' requires {} resources, but {} were bound.",
                    bindPointName,
                    bindPoints[i].resourceCount,
                    sources.size()));
            }

            for (auto& source : sources)
            {
                assert(source.resource != nullptr);
                assert(source.resourceDesc != nullptr);

                if (!std::holds_alternative<Model::BufferDesc>(source.resourceDesc->value))
                {
                    throw std::invalid_argument("DML operators only support buffer bindings");
                }

                auto& bufferDesc = std::get<Model::BufferDesc>(source.resourceDesc->value);

                bindingData.bufferBindings[bufferIndex].Buffer = source.resource;
                bindingData.bufferBindings[bufferIndex].Offset = source.elementOffset * source.elementSizeInBytes;
                bindingData.bufferBindings[bufferIndex].SizeInBytes = bufferDesc.sizeInBytes - bindingData.bufferBindings[bufferIndex].Offset;
                bindingData.bindingDescs[bufferIndex].Type = DML_BINDING_TYPE_BUFFER;
                bindingData.bindingDescs[bufferIndex].Desc = &bindingData.bufferBindings[bufferIndex];
                bufferIndex++;
            }
        }
    }
}

void DmlSerializedGraphDispatchable::Initialize()
{
    BuildGraph();
    m_graphCompiled->SetName(std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(m_name).data());

    ComPtr<IDMLOperatorInitializer> initializer;
    IDMLCompiledOperator* ops[] = { m_graphCompiled.Get() };
    THROW_IF_FAILED(m_device->DML()->CreateOperatorInitializer(
        _countof(ops),
        ops,
        IID_PPV_ARGS(&initializer)));

    auto min = initializer->GetBindingProperties().RequiredDescriptorCount;

    // Create a descriptor heap with at least one descriptor. Even if the op doesn't require any descriptors the
    // binding table expects valid descriptor handles.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = std::max(1u, initializer->GetBindingProperties().RequiredDescriptorCount);
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(descriptorHeap.ReleaseAndGetAddressOf())));

    ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap.Get() };
    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = initializer.Get();
    bindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = initializer->GetBindingProperties().RequiredDescriptorCount;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Inputs flagged OWNED_BY_DML must be bound during initialization (and only initialization).
    if (!m_initBindings.empty())
    {
        // Initializers can initialize multiple inputs simultaneously, so each compiled op's inputs must
        // be bound using a separate buffer array binding.
        BindingData inputBindingData = {};
        FillBindingData(m_desc.bindPoints.inputs, &m_initBindings, nullptr, inputBindingData, true);

        DML_BUFFER_ARRAY_BINDING bufferArrayBindings = {};
        if (inputBindingData.bufferBindings.size() > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument(fmt::format("Initialization Input BindingCount '{}' is too large.", inputBindingData.bufferBindings.size()));
        }
        bufferArrayBindings.BindingCount = static_cast<uint32_t>(inputBindingData.bufferBindings.size());
        bufferArrayBindings.Bindings = inputBindingData.bufferBindings.data();

        DML_BINDING_DESC bindingDesc = {};
        bindingDesc.Desc = &bufferArrayBindings;
        bindingDesc.Type = DML_BINDING_TYPE_BUFFER_ARRAY;

        bindingTable->BindInputs(1, &bindingDesc);
    }

    // A temporary resource may be required to initialize the operators.
    auto tempBufferSize = initializer->GetBindingProperties().TemporaryResourceSize;
    if (tempBufferSize > 0)
    {
        ComPtr<ID3D12Resource> tempBuffer = m_device->CreatePreferredDeviceMemoryBuffer(tempBufferSize);
        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer));
    }

    // Each compiled op's persistent resource is bound as an output of the initializer.
    auto persistentBufferSize = m_graphCompiled->GetBindingProperties().PersistentResourceSize;
    if (persistentBufferSize > 0)
    {
        m_persistentBuffer = m_device->CreatePreferredDeviceMemoryBuffer(persistentBufferSize);
        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindOutputs(1, &bindingDesc);
    }

    m_device->KeepAliveUntilNextCommandListDispatch(std::move(descriptorHeap));
    m_device->RecordInitialize(initializer.Get(), bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}

void DmlSerializedGraphDispatchable::Bind(const Bindings& bindings, uint32_t iteration)
{
    auto bindingProps = m_graphCompiled->GetBindingProperties();

    BindingData inputBindingData = {};
    FillBindingData(m_desc.bindPoints.inputs, &m_initBindings, &bindings, inputBindingData);

    BindingData outputBindingData = {};
    FillBindingData(m_desc.bindPoints.outputs, &m_initBindings, &bindings, outputBindingData);

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = bindingProps.RequiredDescriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_device->D3D()->CreateDescriptorHeap(
        &descriptorHeapDesc,
        IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.ReleaseAndGetAddressOf())));

    ID3D12DescriptorHeap* descriptorHeaps[] = { m_descriptorHeap.Get() };
    m_device->GetCommandList()->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_graphCompiled.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    THROW_IF_FAILED(m_device->DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(m_bindingTable.ReleaseAndGetAddressOf())));

    if (inputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(fmt::format("BindInputs count  '{}' is too large.", inputBindingData.bindingDescs.size()));
    }
    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingData.bindingDescs.size()), inputBindingData.bindingDescs.data());

    ComPtr<ID3D12Resource> tempBuffer;
    auto tempBufferSize = bindingProps.TemporaryResourceSize;
    if (tempBufferSize > 0)
    {
        tempBuffer = m_device->CreatePreferredDeviceMemoryBuffer(tempBufferSize);

        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
        m_device->KeepAliveUntilNextCommandListDispatch(std::move(tempBuffer));
    }

    auto persistentBufferSize = bindingProps.PersistentResourceSize;
    if (persistentBufferSize > 0)
    {
        DML_BUFFER_BINDING bufferBinding = { m_persistentBuffer.Get(), 0, persistentBufferSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }
    if (outputBindingData.bindingDescs.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(fmt::format("BindOutputs count  '{}' is too large.", outputBindingData.bindingDescs.size()));
    }
    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingData.bindingDescs.size()), outputBindingData.bindingDescs.data());

    // DML may remove the device if invalid bindings are specified.
    THROW_IF_FAILED(m_device->DML()->GetDeviceRemovedReason());
}

void DmlSerializedGraphDispatchable::Dispatch(const Model::DispatchCommand& args, uint32_t iteration, DeferredBindings& deferredBindings)
{
    m_device->RecordDispatch(m_graphCompiled.Get(), m_bindingTable.Get());
    m_device->ExecuteCommandListAndWait();
}