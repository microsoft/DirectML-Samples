#include "pch.h"
#include "Adapter.h"
#include "Device.h"
#include "BucketAllocator.h"
#include "Model.h"
#include "Dispatchable.h"
#include "DmlDispatchable.h"
#ifndef DXCOMPILER_NONE
#include "HlslDispatchable.h"
#endif
#ifndef ONNXRUNTIME_NONE
#include "OnnxDispatchable.h"
#endif
#include "StdSupport.h"
#include "NpyReaderWriter.h"
#include "CommandLineArgs.h"
#include "Executor.h"
#include <half.hpp>

using Microsoft::WRL::ComPtr;

struct Timer
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    Timer() { start = std::chrono::steady_clock::now(); }
    Timer& Start() { start = std::chrono::steady_clock::now(); return *this; }
    Timer& End() { end = std::chrono::steady_clock::now(); return *this; }
    double DurationInMilliseconds() { return std::chrono::duration<double>(end - start).count() * 1000; }
};

struct Timings
{
    std::vector<double> rawSamples;

    struct Stats
    {
        size_t count;
        double sum;
        double average;
        double median;
        double min;
        double max;
    };

    struct SampleStats
    {
        Stats cold;
        Stats hot;
    };

    Stats ComputeStats(gsl::span<const double> sampleSpan) const
    {
        Stats stats = {};

        if (!sampleSpan.empty())
        {
            std::vector<double> samples(sampleSpan.size());
            std::copy(sampleSpan.begin(), sampleSpan.end(), samples.begin());
            std::sort(samples.begin(), samples.end());

            stats.count = sampleSpan.size();
            stats.sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            stats.average = stats.sum / samples.size();
            stats.median = samples[samples.size() / 2];
            stats.min = samples[0];
            stats.max = samples[samples.size() - 1];
        }

        return stats;
    }

    SampleStats ComputeStats(size_t maxWarmupSampleCount) const
    {
        SampleStats stats = {};
        if (rawSamples.empty())
        {
            return stats;
        }

        // The first samples may be from "warmup" runs that skew the results because of cold caches.
        // We call the first few samples "cold" and the later samples "hot". We always want at least 
        // 1 hot sample. Example:
        //
        // Raw Samples | maxWarmup | cold | hot
        // ------------|-----------|------|----
        //           0 |         2 |    0 |   0
        //           1 |         2 |    0 |   1
        //           2 |         2 |    1 |   1
        //           3 |         2 |    2 |   1
        //           4 |         2 |    2 |   2
        //           5 |         2 |    2 |   3

        size_t coldSampleCount = std::min(std::max<size_t>(rawSamples.size(), 1) - 1, maxWarmupSampleCount);
        size_t hotSampleCount = rawSamples.size() - coldSampleCount;
        assert(coldSampleCount + hotSampleCount == rawSamples.size());

        stats.cold = ComputeStats(gsl::make_span<const double>(rawSamples.data(), coldSampleCount));
        stats.hot = ComputeStats(gsl::make_span<const double>(rawSamples.data() + coldSampleCount, hotSampleCount));

        return stats;
    }
};

Executor::Executor(Model& model, std::shared_ptr<Device> device, const CommandLineArgs& args) : m_model(model), m_device(device), m_commandLineArgs(args)
{
    // Initialize buffer resources.
    {
        PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255, 255, 0), "Initialize resources");
        for (auto& desc : model.GetResourceDescs())
        {
            // Only buffers are supported right now.
            assert(std::holds_alternative<Model::BufferDesc>(desc.value));
            auto& bufferDesc = std::get<Model::BufferDesc>(desc.value);
            auto wName = std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(desc.name);

            m_resources[desc.name] = std::move(device->CreateBuffer(
                bufferDesc.sizeInBytes, 
                bufferDesc.initialValues, 
                wName,
                m_commandLineArgs.GetCustomHeaps() ? D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE : D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                m_commandLineArgs.GetCustomHeaps() ? D3D12_MEMORY_POOL_L0 : D3D12_MEMORY_POOL_UNKNOWN
            ));
        }
    }
    device->ExecuteCommandListAndWait();

    // Create dispatchables.
    for (auto& desc : model.GetDispatchableDescs())
    {
        try
        {
            if (std::holds_alternative<Model::HlslDispatchableDesc>(desc.value))
            {
#ifdef DXCOMPILER_NONE
                throw std::invalid_argument("HLSL dispatchables require DXCompiler");
#else
                m_dispatchables[desc.name] = std::make_unique<HlslDispatchable>(device, std::get<Model::HlslDispatchableDesc>(desc.value), args);
#endif
            }
            else if (std::holds_alternative<Model::OnnxDispatchableDesc>(desc.value))
            {
#ifdef ONNXRUNTIME_NONE
                throw std::invalid_argument("ONNX dispatchables require ONNX Runtime");
#else
                m_dispatchables[desc.name] = std::make_unique<OnnxDispatchable>(device, std::get<Model::OnnxDispatchableDesc>(desc.value), args);
#endif
            }
            else
            {
                auto& dmlDispatchableDesc = std::get<Model::DmlDispatchableDesc>(desc.value);

                Dispatchable::Bindings initBindings;
                try
                {
                    initBindings = ResolveBindings(dmlDispatchableDesc.initBindings);
                }
                catch (const std::exception& e)
                {
                    LogError(fmt::format("Failed to resolve bindings: {}", e.what()));
                    return;
                }

                m_dispatchables[desc.name] = std::make_unique<DmlDispatchable>(desc.name, device, dmlDispatchableDesc, initBindings);
            }
        }
        catch(const std::exception& e)
        {
            throw std::invalid_argument(fmt::format("ERROR creating dispatchable '{}': {}", desc.name, e.what()));
        }
    }

    // Compile/initialize dispatchables.
    {
        Timer timer;

        PIXBeginEvent(m_device->GetCommandQueue(), PIX_COLOR(255, 255, 0), "Initialize dispatchables");
        for (auto& dispatchable : m_dispatchables)
        {
            try
            {
                timer.Start();
                PIXBeginEvent(PIX_COLOR(128,255,0), L"Init");
                dispatchable.second->Initialize();
                PIXEndEvent();
                timer.End();

                if (m_commandLineArgs.GetTimingVerbosity() >= TimingVerbosity::Extended)
                {
                    LogInfo(fmt::format("Initialize '{}': {:.4f} ms", dispatchable.first, timer.DurationInMilliseconds()));
                }
            }
            catch (const std::exception& e)
            {
                throw std::invalid_argument(fmt::format("ERROR while initializing '{}': {}", dispatchable.first, e.what()));
            }
        }
        PIXEndEvent(m_device->GetCommandQueue());
    }
}

void Executor::Run()
{
    for (auto& command : m_model.GetCommands())
    {
        std::visit(*this, command);
    }
}

void Executor::operator()(const Model::DispatchCommand& command)
{
    auto& dispatchable = m_dispatchables[command.dispatchableName];

    Timings cpuTimings;
    Timings gpuTimings;

    Dispatchable::Bindings bindings;
    try
    {
        bindings = ResolveBindings(command.bindings);
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to resolve bindings: {}", e.what()));
        return;
    }

    // Dispatch
    uint32_t iterationsCompleted = 0;
    bool timedOut = false;
    PIXBeginEvent(PIX_COLOR(128, 255, 0), L"Dispatch Loop");
    try
    {
        Timer loopTimer, iterationTimer, bindTimer, dispatchTimer;

        for (; !timedOut && iterationsCompleted < m_commandLineArgs.DispatchIterations(); iterationsCompleted++)
        {
            iterationTimer.Start();

            // Bind
            PIXBeginEvent(PIX_COLOR(128, 255, 0), L"Bind");
            try
            {
                dispatchable->Bind(bindings, iterationsCompleted);
            }
            catch (const std::exception& e)
            {
                LogError(fmt::format("ERROR while binding resources: {}\n", e.what()));
                return;
            }
            PIXEndEvent();

            // Dispatch
            dispatchTimer.Start();
            dispatchable->Dispatch(command, iterationsCompleted);
            dispatchable->Wait();
            cpuTimings.rawSamples.push_back(dispatchTimer.End().DurationInMilliseconds() / m_commandLineArgs.DispatchRepeat());

            // The dispatch interval defaults to 0 (dispatch as fast as possible). However, the user may increase it
            // to potentially introduce a sleep between each iteration.
            double timeToSleep = std::max(0.0, m_commandLineArgs.MinimumDispatchIntervalInMilliseconds() - iterationTimer.End().DurationInMilliseconds());

            if (m_commandLineArgs.TimeToRunInMilliseconds() &&
                loopTimer.End().DurationInMilliseconds() + timeToSleep > m_commandLineArgs.TimeToRunInMilliseconds().value())
            {
                timedOut = true;
            }
            else
            {
                // Not particularly precise (may be off by some milliseconds). Consider using OS APIs in the future.
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<size_t>(timeToSleep)));
            }
        }
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to execute dispatchable: {}", e.what()));
        return;
    }
    PIXEndEvent();

    auto cpuStats = cpuTimings.ComputeStats(m_commandLineArgs.MaxWarmupSamples());

    // GPU timings are capped at a fixed size ring buffer. The first samples may have been 
    // overwritten, in which case the warmup samples are dropped.
    gpuTimings.rawSamples = m_device->ResolveTimingSamples();
    assert (cpuTimings.rawSamples.size() >= gpuTimings.rawSamples.size());
    uint32_t gpuSamplesOverwritten = cpuTimings.rawSamples.size() - gpuTimings.rawSamples.size();
    auto gpuStats = gpuTimings.ComputeStats(std::max(m_commandLineArgs.MaxWarmupSamples(), gpuSamplesOverwritten) - gpuSamplesOverwritten);

    if (iterationsCompleted > 0)
    {
        if (m_commandLineArgs.GetTimingVerbosity() == TimingVerbosity::Basic)
        {
            LogInfo(fmt::format("Dispatch '{}': {} iterations, {:.4f} ms median (CPU), {:.6f} ms median (GPU)", 
                command.dispatchableName, 
                iterationsCompleted,
                cpuStats.hot.median,
                gpuStats.hot.median
            ));
        }
        else
        {
            LogInfo(fmt::format("Dispatch '{}': {} iterations", 
                command.dispatchableName, iterationsCompleted
            ));

            LogInfo(fmt::format("CPU Timings (Cold) : {} samples, {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                cpuStats.cold.count, cpuStats.cold.average, cpuStats.cold.min, cpuStats.cold.median, cpuStats.cold.max
            ));

            LogInfo(fmt::format("GPU Timings (Cold) : {} samples, {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                gpuStats.cold.count, gpuStats.cold.average, gpuStats.cold.min, gpuStats.cold.median, gpuStats.cold.max
            ));

            LogInfo(fmt::format("CPU Timings (Hot)  : {} samples, {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                cpuStats.hot.count, cpuStats.hot.average, cpuStats.hot.min, cpuStats.hot.median, cpuStats.hot.max
            ));

            LogInfo(fmt::format("GPU Timings (Hot)  : {} samples, {:.4f} ms average, {:.4f} ms min, {:.4f} ms median, {:.4f} ms max", 
                gpuStats.hot.count, gpuStats.hot.average, gpuStats.hot.min, gpuStats.hot.median, gpuStats.hot.max
            ));

            if (gpuSamplesOverwritten > 0)
            {
                LogInfo(fmt::format("GPU samples buffer has {} samples overwritten.", gpuSamplesOverwritten));
            }
        }

        if (m_commandLineArgs.GetTimingVerbosity() >= TimingVerbosity::All)
        {
            LogInfo("The timings of each iteration: ");

            for (uint32_t i = 0; i < iterationsCompleted; ++i)
            {
                if (i < gpuSamplesOverwritten)
                {
                    // GPU samples are limited to a fixed size, so the initial iterations
                    // may not have timing information (overwritten timestamps).
                    LogInfo(fmt::format("iteration {}: {:.4f} ms (CPU)", 
                        i, cpuTimings.rawSamples[i]
                    ));
                }
                else
                {
                    LogInfo(fmt::format("iteration {}: {:.4f} ms (CPU), {:.4f} ms (GPU)",
                        i, cpuTimings.rawSamples[i], gpuTimings.rawSamples[i - gpuSamplesOverwritten]
                    ));
                }
            }
        }
    }
}

template <typename T>
struct BufferDataView
{
    gsl::span<const std::byte> byteValues;
    const Model::BufferDesc& desc;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BufferDataView<T>& view)
{
    uint32_t elementCount = view.desc.initialValues.size() / Device::GetSizeInBytes(view.desc.initialValuesDataType);
    auto values = reinterpret_cast<const T*>(view.byteValues.data());
    for (uint32_t elementIndex = 0; elementIndex < elementCount; elementIndex++)
    {
        os << values[elementIndex];
        if (elementIndex < elementCount - 1)
        {
            os << ", ";
        }
    }
    return os;
}

std::string ToString(gsl::span<const std::byte> byteValues, const Model::BufferDesc& desc)
{
    std::stringstream ss;
    switch (desc.initialValuesDataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT16: ss << BufferDataView<half_float::half>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_FLOAT32: ss << BufferDataView<float>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_FLOAT64: ss << BufferDataView<double>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT8: ss << BufferDataView<uint8_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT16: ss << BufferDataView<uint16_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT32: ss << BufferDataView<uint32_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_UINT64: ss << BufferDataView<uint64_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT8: ss << BufferDataView<int8_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT16: ss << BufferDataView<int16_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT32: ss << BufferDataView<int32_t>{byteValues, desc}; break;
    case DML_TENSOR_DATA_TYPE_INT64: ss << BufferDataView<int64_t>{byteValues, desc}; break;
    default: throw std::invalid_argument("Unexpected DML_TENSOR_DATA_TYPE");
    }
    return ss.str();
}

void Executor::operator()(const Model::PrintCommand& command)
{
    PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "Print: %s", command.resourceName.c_str());

    try
    {
        auto resource = m_resources[command.resourceName];
        auto outputValues = m_device->Download(resource.Get());
        auto& resourceDesc = m_model.GetResource(command.resourceName);
        auto& bufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);
        LogInfo(fmt::format("Resource '{}': {}", command.resourceName, ToString(outputValues, bufferDesc)));
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to print resource: {}", e.what()));
    }
}

void Executor::operator()(const Model::WriteFileCommand& command)
{
    PIXScopedEvent(m_device->GetCommandList(), PIX_COLOR(255,255,0), "WriteFile: %s", command.resourceName.c_str());

    try
    {
        auto resource = m_resources[command.resourceName];
        auto fileData = m_device->Download(resource.Get());
        auto& resourceDesc = m_model.GetResource(command.resourceName);
        auto& bufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);

        std::ofstream file(command.targetPath.c_str(), std::ifstream::trunc | std::ifstream::binary);
        if (!file.is_open())
        {
            throw std::ios::failure("Could not open file");
        }

        // If NumPy array, serialize data into .npy file.
        if (IsNpyFilenameExtension(command.targetPath))
        {
            // If no dimensions were given, then treat as a 1D array.
            std::vector<uint32_t> dimensions(command.dimensions);
            if (dimensions.empty())
            {
                uint32_t elementCount = bufferDesc.sizeInBytes / Device::GetSizeInBytes(bufferDesc.initialValuesDataType);
                dimensions.push_back(elementCount);
            }

            std::vector<std::byte> npyFileData;
            WriteNpy(fileData, bufferDesc.initialValuesDataType, dimensions, /*out*/ npyFileData);
            std::swap(fileData, npyFileData);
        }

        file.write(reinterpret_cast<const char*>(fileData.data()), fileData.size());
        LogInfo(fmt::format("Resource '{}' written to '{}'", command.resourceName, command.targetPath));
    }
    catch (const std::exception& e)
    {
        LogError(fmt::format("Failed to write resource to file '{}': {}", command.targetPath, e.what()));
    }
}

Dispatchable::Bindings Executor::ResolveBindings(const Model::Bindings& modelBindings)
{
    Dispatchable::Bindings bindings;

    for (auto& modelBinding : modelBindings)
    {
        std::vector<Dispatchable::BindingSource> sourceResources;

        for (auto& modelSource : modelBinding.second)
        {
            // Validated when the model is constructed.
            assert(m_resources.find(modelSource.name) != m_resources.end());
            auto& resourceDesc = m_model.GetResource(modelSource.name);

            Dispatchable::BindingSource source = {};
            source.elementSizeInBytes = modelSource.elementSizeInBytes;
            source.elementCount = modelSource.elementCount;
            source.elementOffset = modelSource.elementOffset;
            source.format = modelSource.format;
            source.resource = m_resources[modelSource.name].Get();
            source.resourceDesc = &resourceDesc;
            source.shape = modelSource.shape;

            if (std::holds_alternative<Model::BufferDesc>(resourceDesc.value))
            {
                auto& modelBufferDesc = std::get<Model::BufferDesc>(resourceDesc.value);
                if (source.elementSizeInBytes == 0 && modelBufferDesc.initialValuesDataType != DML_TENSOR_DATA_TYPE_UNKNOWN)
                {
                    // If the binding doesn't specify, assume the data type size used to initialize the buffer.
                    source.elementSizeInBytes = Device::GetSizeInBytes(modelBufferDesc.initialValuesDataType);
                }

                if (source.elementCount == 0 && source.elementSizeInBytes != 0)
                {
                    // If the binding doesn't specify, assume the number of elements used to initialize the buffer.
                    source.elementCount = modelBufferDesc.initialValues.size() / source.elementSizeInBytes;
                }
            }

            if (modelSource.counterName)
            {
                // Validated when the model is constructed.
                assert(m_resources.find(*modelSource.counterName) != m_resources.end());
                source.counterResource = m_resources[*modelSource.counterName].Get();
                source.counterOffsetBytes = modelSource.counterOffsetBytes;
            }

            sourceResources.push_back(source);
        }

        bindings[modelBinding.first] = sourceResources;
    }

    return bindings;
}