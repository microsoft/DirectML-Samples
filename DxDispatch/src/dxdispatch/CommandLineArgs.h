#pragma once

#include "PixCaptureHelper.h"

enum TimingVerbosity
{
    Basic,
    Extended,
    All
};

class CommandLineArgs
{
public:
    CommandLineArgs() = default;
    CommandLineArgs(int argc, char** argv);

    bool ShowAdapters() const { return m_showAdapters; }
    bool ShowDependencies() const { return m_showDependencies; }
    bool PrintHelp() const { return m_printHelp; }
    bool PrintHlslDisassembly() const { return m_printHlslDisassembly; }
    bool DebugLayersEnabled() const { return m_debugLayersEnabled; }
    TimingVerbosity GetTimingVerbosity() const { return m_timingVerbosity; }
    bool ForceDisablePrecompiledShadersOnXbox() const { return m_forceDisablePrecompiledShadersOnXbox; }
    bool ClearShaderCaches() const { return m_clearShaderCaches; }
    bool DisableAgilitySDK() const { return m_disableAgilitySDK; }
    const std::string& AdapterSubstring() const { return m_adapterSubstring; }
    const std::filesystem::path& ModelPath() const { return m_modelPath; }
    const std::string& HelpText() const { return m_helpText; }
    uint32_t DispatchIterations() const { return m_dispatchIterations; }
    uint32_t DispatchRepeat() const { return m_dispatchRepeat; }
    std::optional<uint32_t> TimeToRunInMilliseconds() const { return m_timeToRunInMilliseconds; }
    uint32_t MinimumDispatchIntervalInMilliseconds() const { return m_minDispatchIntervalInMilliseconds; }
    uint32_t MaxWarmupSamples() const { return m_maxWarmupSamples; }
    D3D12_COMMAND_LIST_TYPE CommandListType() const { return m_commandListType; }
    PixCaptureType GetPixCaptureType() const { return m_pixCaptureType; }
    const std::string& PixCaptureName() const { return m_pixCaptureName; }

    bool GetUavBarrierAfterDispatch() const { return m_uavBarrierAfterDispatch; }
    bool GetAliasingBarrierAfterDispatch() const { return m_aliasingBarrierAfterDispatch; }

    bool GetCustomHeaps() const { return m_customHeaps; }

    // ONNX
    gsl::span<const std::pair<std::string, uint32_t>> GetOnnxFreeDimensionNameOverrides() const { return m_onnxFreeDimensionNameOverrides; }
    gsl::span<const std::pair<std::string, uint32_t>> GetOnnxFreeDimensionDenotationOverrides() const { return m_onnxFreeDimensionDenotationOverrides; }
    gsl::span<const std::pair<std::string, std::string>> GetOnnxSessionOptionConfigEntries() const { return m_onnxSessionOptionConfigEntries; }
    const std::unordered_map<std::string, std::vector<int64_t>>& GetOnnxBindingShapes() const { return m_onnxBindShapes; }
    std::optional<uint32_t> GetOnnxGraphOptimizationLevel() const { return m_onnxGraphOptimizationLevel; }
    std::optional<uint32_t> GetOnnxLoggingLevel() const { return m_onnxLoggingLevel; }
    bool PrintVerboseOnnxBindingInfo() const { return m_onnxPrintVerboseBindingInfo; }

private:
    bool m_showAdapters = false;
    bool m_showDependencies = false;
    bool m_printHelp = false;
    bool m_printHlslDisassembly = false;
    bool m_debugLayersEnabled = false;
    TimingVerbosity m_timingVerbosity = TimingVerbosity::Basic;
    bool m_forceDisablePrecompiledShadersOnXbox = true;
    bool m_clearShaderCaches = false;
    bool m_disableAgilitySDK = false;
    bool m_uavBarrierAfterDispatch = true;
    bool m_aliasingBarrierAfterDispatch = false;
    bool m_customHeaps = false;
    std::string m_adapterSubstring = "";
    std::filesystem::path m_modelPath;
    std::string m_pixCaptureName = "dxdispatch";
    std::string m_helpText;
    uint32_t m_dispatchIterations = 1;
    uint32_t m_dispatchRepeat = 1;
    std::optional<uint32_t> m_timeToRunInMilliseconds = {};
    uint32_t m_minDispatchIntervalInMilliseconds = 0;
    uint32_t m_maxWarmupSamples = 1;

    // Tools like PIX generally work better when work is recorded into a graphics queue, so it's set as the default here.
    D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    PixCaptureType m_pixCaptureType = PixCaptureType::Manual;

    // [onnx models] Overrides for free dimensions by name. 
    // The first value in the pair is the name, and the second value is the dimension size.
    std::vector<std::pair<std::string, uint32_t>> m_onnxFreeDimensionNameOverrides;

    // [onnx models] Overrides for free dimensions by denotation. 
    // The first value in the pair is the denotation, and the second value is the dimension size.
    std::vector<std::pair<std::string, uint32_t>> m_onnxFreeDimensionDenotationOverrides;

    // [onnx models] Key/value pairs passed to SessionOptions config entries
    // https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/dml/dml_session_options_config_keys.h
    std::vector<std::pair<std::string, std::string>> m_onnxSessionOptionConfigEntries;

    // Optional binding shapes for dynamically shaped tensors.
    // Maps ONNX tensor names to a specific shape that should be used when binding the tensor to a resource/OrtValue.
    std::unordered_map<std::string, std::vector<int64_t>> m_onnxBindShapes;

    std::optional<uint32_t> m_onnxGraphOptimizationLevel;
    std::optional<uint32_t> m_onnxLoggingLevel;
    bool m_onnxPrintVerboseBindingInfo = false;
};