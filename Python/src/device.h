//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <gpgmm_d3d12.h>

namespace pydml
{
    class Device
    {
    public:
        explicit Device(bool useGpu = true, bool useDebugLayer = false, DXGI_GPU_PREFERENCE gpuPreference = DXGI_GPU_PREFERENCE_UNSPECIFIED);

        std::vector<pydml::TensorData*> Compute(
            IDMLCompiledOperator* op,
            std::vector<pydml::Binding*>& inputs,
            std::vector<dml::Expression*>& outputs
            );

        inline bool UseGpu() const
        {
            return m_useGpu;
        }

        inline IDMLDevice* GetDevice() const
        {
            return m_dmlDevice.Get();
        }

    protected:
        void InitializeOperator(
            IDMLCompiledOperator* op,
            std::vector<pydml::Binding*>& inputs
            );

        std::vector<pydml::TensorData*> DispatchOperator(
            IDMLCompiledOperator* op,
            std::vector<pydml::Binding*>& inputs,
            std::vector<dml::Expression*>& outputs
            );

        void RecordOutputReadBack(uint64_t outputsResourceSize);

        std::vector<pydml::TensorData*> DownloadFromReadBackHeap(
            uint64_t outputsResourceSize, 
            std::vector<dml::Expression*>& outputs,
            std::vector<DmlBufferBinding>& outputBindings
            );

        void EnsureUploadHeapSize(uint64_t requestedSizeInBytes);
        void EnsureReadBackHeapSize(uint64_t requestedSizeInBytes);
        void EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors);

        void ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers);

        void ExecuteCommandListAndWait();

        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocator> m_resourceAllocator;
        
        // Residency management is used to handle oversubscribing of video memory. 
        // The lifetime of |m_residencyManager| will be fully owned by |m_resourceAllocator|.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResidencyManager> m_residencyManager;

        Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_operatorInitializer;
        Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
        
        // GPU descriptor heaps require explicit residency management since they must
        // stay in a GPU visible memory.
        class SVDescriptorHeap : public gpgmm::d3d12::Heap {
        public:
            SVDescriptorHeap(ComPtr<ID3D12DescriptorHeap> heap, uint64_t size) 
            : gpgmm::d3d12::Heap(heap, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, size), m_Heap(heap) {
            }

            Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_Heap;
        };

        // GPU- and CPU-visible descriptor heaps used for ClearUnorderedAccessView
        std::unique_ptr<SVDescriptorHeap> m_clearUavDescriptorHeapGpu;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapCpu;

        // Lazily-initialized resources for operator initialization/execution
        std::unique_ptr<SVDescriptorHeap> m_descriptorHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_uploadHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_readbackHeap;

        // DEFAULT heap buffers to hold input tensors, output tensors, and temporary and persistent resources. The input
        // and output resources are suballocated for operators that have multiple inputs or outputs.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_inputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_outputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_temporaryResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_persistentResource;

        gpgmm::d3d12::ResidencySet m_residencySet;

        bool m_useCpuCustomHeapResources = false;
        bool m_useGpu = true;
        DXGI_GPU_PREFERENCE m_gpuPreference;
    };
}