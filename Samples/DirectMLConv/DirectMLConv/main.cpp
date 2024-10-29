#include "pch.h"

#pragma warning(disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

using Microsoft::WRL::ComPtr;

void InitializeDirect3D12(
    ComPtr<ID3D12Device>& d3D12Device,
    ComPtr<ID3D12CommandQueue>& commandQueue,
    ComPtr<ID3D12CommandAllocator>& commandAllocator,
    ComPtr<ID3D12GraphicsCommandList>& commandList)
{
    HRESULT hr{};

    ComPtr<IDXGIFactory4> dxgiFactory;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf()));

    if (hr != S_OK)
        std::cout << "failed to create dxgi factory";

    ComPtr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};

    do
    {
        dxgiAdapter = nullptr;
        THROW_IF_FAILED(dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.ReleaseAndGetAddressOf()));
        ++adapterIndex;

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED) continue;
        // THROW_IF_FAILED(hr);

        if (hr != S_OK)
            std::cout << "failed to init adapter";

    } while (hr != S_OK);

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_GRAPHICS_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_GRAPHICS_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator.Get(),
        nullptr,
        IID_GRAPHICS_PPV_ARGS(commandList.ReleaseAndGetAddressOf())));
}

void CloseExecuteResetWait(
    ComPtr<ID3D12Device> d3D12Device,
    ComPtr<ID3D12CommandQueue> commandQueue,
    ComPtr<ID3D12CommandAllocator> commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> commandList)
{
    THROW_IF_FAILED(commandList->Close());

    ID3D12CommandList* commandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    ComPtr<ID3D12Fence> d3D12Fence;
    THROW_IF_FAILED(d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_GRAPHICS_PPV_ARGS(d3D12Fence.GetAddressOf())));

    wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
    THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

    THROW_IF_FAILED(commandQueue->Signal(d3D12Fence.Get(), 1));
    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);

    THROW_IF_FAILED(commandAllocator->Reset());
    THROW_IF_FAILED(commandList->Reset(commandAllocator.Get(), nullptr));
}

// stride for NCHW layout
void SetStrides(const UINT sizes[4], UINT stridesOut[4]) {
    stridesOut[0] = sizes[1] * sizes[2] * sizes[3];
    stridesOut[1] = sizes[2] * sizes[3];
    stridesOut[2] = sizes[3];
    stridesOut[3] = 1;
}

int main() {

    // initialize D3D12 related resources
    ComPtr<ID3D12Device> d3D12Device;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;

    // Set up Direct3D 12.
    InitializeDirect3D12(d3D12Device, commandQueue, commandAllocator, commandList);


    // Create the DirectML device.
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(
        d3D12Device.Get(),
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(dmlDevice.GetAddressOf())));


    // ************ set input tensor related params ************

    constexpr UINT inputSizes[4] = { 1,1,3,3 };
    constexpr UINT inputTensorElementCount = inputSizes[0] * inputSizes[1] * inputSizes[2] * inputSizes[3];

    UINT inputStrides[4];
    SetStrides(inputSizes, inputStrides);

    DML_BUFFER_TENSOR_DESC inputBufferTensorDesc = {};
    inputBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    inputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    inputBufferTensorDesc.DimensionCount = ARRAYSIZE(inputSizes);
    inputBufferTensorDesc.Sizes = inputSizes;
    inputBufferTensorDesc.Strides = inputStrides;
    inputBufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        inputBufferTensorDesc.DataType,
        inputBufferTensorDesc.DimensionCount,
        inputBufferTensorDesc.Sizes,
        inputBufferTensorDesc.Strides);

    DML_TENSOR_DESC inputTensorDesc{};
    inputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    inputTensorDesc.Desc = &inputBufferTensorDesc;


    // ************ set input tensor related params ************

    constexpr UINT filterSizes[4] = { 1,1,1,1 };
    constexpr UINT filterTensorElementCount = filterSizes[0] * filterSizes[1] * filterSizes[2] * filterSizes[3];

    UINT filterStrides[4];
    SetStrides(filterSizes, filterStrides);

    DML_BUFFER_TENSOR_DESC filterBufferTensorDesc = {};
    filterBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    filterBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    filterBufferTensorDesc.DimensionCount = ARRAYSIZE(filterSizes);
    filterBufferTensorDesc.Sizes = filterSizes;
    filterBufferTensorDesc.Strides = filterStrides;
    filterBufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        filterBufferTensorDesc.DataType,
        filterBufferTensorDesc.DimensionCount,
        filterBufferTensorDesc.Sizes,
        filterBufferTensorDesc.Strides);

    DML_TENSOR_DESC filterTensorDesc{};
    filterTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    filterTensorDesc.Desc = &filterBufferTensorDesc;


    // ************ set output tensor related params ************

    UINT outputSizes[4];
    outputSizes[0] = inputSizes[0];
    outputSizes[1] = filterSizes[0];
    outputSizes[2] = inputSizes[2];
    outputSizes[3] = inputSizes[3];

    UINT outputTensorElementCount = outputSizes[0] * outputSizes[1] * outputSizes[2] * outputSizes[3];

    UINT outputStrides[4];
    SetStrides(outputSizes, outputStrides);

    DML_BUFFER_TENSOR_DESC outputBufferTensorDesc = {};
    outputBufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    outputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    outputBufferTensorDesc.DimensionCount = ARRAYSIZE(outputSizes);
    outputBufferTensorDesc.Sizes = outputSizes;
    outputBufferTensorDesc.Strides = outputStrides;
    outputBufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        outputBufferTensorDesc.DataType,
        outputBufferTensorDesc.DimensionCount,
        outputBufferTensorDesc.Sizes,
        outputBufferTensorDesc.Strides);

    DML_TENSOR_DESC outputTensorDesc{};
    outputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    outputTensorDesc.Desc = &outputBufferTensorDesc;


    // ************ define convolution operator ************ 

    // The output size of a convolution operation is given by:
    //  height = (inputHeight - filterHeight + 2*paddingHeight) / filterStride + 1
    //  width  = (inputWidth  - filterWidth  + 2*paddingWidth ) / filterStride + 1
    //
    // We want to preserve the height and width, so assuming stride is 1, we get:
    //  paddingHeight = (filterHeight - 1) / 2
    //  paddingWidth  = (filterWidth  - 1) / 2
    // If padding is fractional, we pad unevenly with ceil/floor.
    UINT paddingHeightTop = static_cast<UINT>(ceil((filterSizes[2] - 1) / 2.0f));
    UINT paddingHeightBottom = static_cast<UINT>(floor((filterSizes[2] - 1) / 2.0f));
    UINT paddingWidthLeft = static_cast<UINT>(ceil((filterSizes[3] - 1) / 2.0f));
    UINT paddingWidthRight = static_cast<UINT>(floor((filterSizes[3] - 1) / 2.0f));

    UINT strides[] = { 1, 1 };
    UINT dilations[] = { 1, 1 };
    UINT startPadding[] = { paddingHeightTop, paddingWidthLeft };
    UINT endPadding[] = { paddingHeightBottom, paddingWidthRight };
    UINT outputPadding[] = { 0, 0 };


    // create conv DML operator descriptions

    DML_CONVOLUTION_OPERATOR_DESC dmlConvOperatorDesc{};
    dmlConvOperatorDesc.BiasTensor = nullptr;
    dmlConvOperatorDesc.InputTensor = &inputTensorDesc;
    dmlConvOperatorDesc.FilterTensor = &filterTensorDesc;
    dmlConvOperatorDesc.OutputTensor = &outputTensorDesc; // Input and output tensors have same size/type.
    dmlConvOperatorDesc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    dmlConvOperatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    dmlConvOperatorDesc.DimensionCount = 2;
    dmlConvOperatorDesc.GroupCount = 1;

    dmlConvOperatorDesc.StartPadding = startPadding;
    dmlConvOperatorDesc.EndPadding = endPadding;
    dmlConvOperatorDesc.OutputPadding = outputPadding;
    dmlConvOperatorDesc.Strides = strides;
    dmlConvOperatorDesc.Dilations = dilations;

    DML_OPERATOR_DESC dmlOperatorDesc{};
    dmlOperatorDesc.Type = DML_OPERATOR_CONVOLUTION;
    dmlOperatorDesc.Desc = &dmlConvOperatorDesc;


    // Create and compile the conv DML operator

    ComPtr<IDMLOperator> dmlOperator;
    THROW_IF_FAILED(dmlDevice->CreateOperator(
        &dmlOperatorDesc,
        IID_PPV_ARGS(dmlOperator.GetAddressOf())));

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    THROW_IF_FAILED(dmlDevice->CompileOperator(
        dmlOperator.Get(),
        DML_EXECUTION_FLAG_NONE,
        IID_PPV_ARGS(dmlCompiledOperator.GetAddressOf())));


    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator* dmlCompiledOperators[] = { dmlCompiledOperator.Get() };
    THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(
        ARRAYSIZE(dmlCompiledOperators),
        dmlCompiledOperators,
        IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf())));


    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(d3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc,
        IID_GRAPHICS_PPV_ARGS(descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap.Get() };
    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);


    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    THROW_IF_FAILED(dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(dmlBindingTable.GetAddressOf())));


    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    UINT64 temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;


    // Bind and initialize the operator on the GPU.

    ComPtr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0)
    {
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    THROW_IF_FAILED(dmlDevice->CreateCommandRecorder(
        IID_PPV_ARGS(dmlCommandRecorder.GetAddressOf())));

    // Record execution of the operator initializer.
    dmlCommandRecorder->RecordDispatch(
        commandList.Get(),
        dmlOperatorInitializer.Get(),
        dmlBindingTable.Get());


    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // 
    // Bind and execute the operator on the GPU.
    // 
    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);


    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).

    dmlBindingTableDesc.Dispatchable = dmlCompiledOperator.Get();
    THROW_IF_FAILED(dmlBindingTable->Reset(&dmlBindingTableDesc));

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindPersistentResource(&bindingDesc);
    }


    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    // *************** filter tensor ***************

    // 24 elements * 4 == 96 bytes.
    UINT64 filterTensorBufferSize{ filterBufferTensorDesc.TotalTensorSizeInBytes };

    ComPtr<ID3D12Resource> filterUploadBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(filterTensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(filterUploadBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> filterInputBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(filterTensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(filterInputBuffer.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(4);
    std::array<FLOAT, filterTensorElementCount> filterTensorElementArray;
    {
        std::wcout << L"filter tensor: \n";
        for (auto& element : filterTensorElementArray)
        {
            element = 5.6f;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA filterTensorSubresourceData{};
        filterTensorSubresourceData.pData = filterTensorElementArray.data();
        filterTensorSubresourceData.RowPitch = static_cast<LONG_PTR>(filterTensorBufferSize);
        filterTensorSubresourceData.SlicePitch = filterTensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            commandList.Get(),
            filterInputBuffer.Get(),
            filterUploadBuffer.Get(),
            0,
            0,
            1,
            &filterTensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                filterInputBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING filterInputBufferBinding{ filterInputBuffer.Get(), 0, filterTensorBufferSize };
    DML_BINDING_DESC filterInputBindingDesc{ DML_BINDING_TYPE_BUFFER, &filterInputBufferBinding };
    // dmlBindingTable->BindInputs(1, &filterInputBindingDesc);


    // *************** input tensor ***************
    // 24 elements * 4 == 96 bytes.
    UINT64 tensorBufferSize{ inputBufferTensorDesc.TotalTensorSizeInBytes };

    ComPtr<ID3D12Resource> uploadBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(uploadBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> inputBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(inputBuffer.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(4);
    std::array<FLOAT, inputTensorElementCount> inputTensorElementArray;
    {
        std::wcout << L"input tensor: \n";
        for (auto& element : inputTensorElementArray)
        {
            element = -2.0f;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputTensorElementArray.data();
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(tensorBufferSize);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            commandList.Get(),
            inputBuffer.Get(),
            uploadBuffer.Get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING inputBufferBinding{ inputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding };

    DML_BINDING_DESC bindings[3];
    bindings[0] = inputBindingDesc;
    bindings[1] = filterInputBindingDesc;
    bindings[2].Type = DML_BINDING_TYPE_NONE;
    bindings[2].Desc = nullptr;
    dmlBindingTable->BindInputs(3, bindings);
    // dmlBindingTable->BindInputs(1, &inputBindingDesc);


    // *************** output tensor ***************

    ComPtr<ID3D12Resource> outputBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(outputBuffer.GetAddressOf())));

    DML_BUFFER_BINDING outputBufferBinding{ outputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC outputBindingDesc{ DML_BINDING_TYPE_BUFFER, &outputBufferBinding };
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);

    // Record execution of the compiled operator.
    dmlCommandRecorder->RecordDispatch(commandList.Get(), dmlCompiledOperator.Get(), dmlBindingTable.Get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.

    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_GRAPHICS_PPV_ARGS(readbackBuffer.GetAddressOf())));

    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        )
    );

    commandList->CopyResource(readbackBuffer.Get(), outputBuffer.Get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(tensorBufferSize) };
    FLOAT* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));

    std::wstring outputString = L"\noutput tensor: \n";
    for (size_t tensorElementIndex{ 0 }; tensorElementIndex < outputTensorElementCount; ++tensorElementIndex, ++outputBufferData)
    {
        outputString += std::to_wstring(*outputBufferData) + L' ';
    }

    std::wcout << outputString << std::endl;
    OutputDebugStringW(outputString.c_str());

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);

}