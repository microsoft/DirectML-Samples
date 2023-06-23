#include "pch.h"
#include "CommandLineArgs.h"
// cxxopts will parse comma-separated arguments as vectors by default (e.g. --foo 1,2,3,4 can be
// parsed as a vector<int>. This interferes with binding shape syntax, so we override the delimiter
// and parse manually.
#define CXXOPTS_VECTOR_DELIMITER '\0'
#include <cxxopts.hpp>
#include "config.h"

CommandLineArgs::CommandLineArgs(int argc, char** argv)
{
        auto banner = fmt::format(R"({} version {}
  DirectML     : {}
  D3D12        : {}
  DXCompiler   : {}
  PIX          : {}
  ONNX Runtime : {}
)",
        c_projectName, 
        c_projectVersion,
        c_directmlConfig,
        c_d3d12Config,
        c_dxcompilerConfig,
        c_pixConfig,
        c_ortConfig);
    
    cxxopts::Options options(c_projectName, banner);
    options.add_options()
        (
            "m,model", 
            "Path to JSON/ONNX model file", 
            cxxopts::value<decltype(m_modelPath)>()
        )
        (
            "h,help", 
            "Print command-line usage help", 
            cxxopts::value<bool>()
        )
        (
            "S,show_dependencies",
            "Show version info for dependencies including DirectX components",
            cxxopts::value<bool>()
        )
        ;

    // TIMING OPTIONS
    options.add_options("Timing")
        (
            "i,dispatch_iterations", 
            "The number of iterations in bind/dispatch/wait loop", 
            cxxopts::value<uint32_t>()->default_value("1")
        )
        (
            "r,dispatch_repeat", 
            "The number of times dispatch is invoked within each loop iteration (for microbenchmarking)", 
            cxxopts::value<uint32_t>()->default_value("1")
        )
        (
            "t,milliseconds_to_run",
            "Specifies the total time to run the test for. Overrides dispatch_iterations",
            cxxopts::value<uint32_t>()
        )
        (
            "I,dispatch_interval",
            "The minimum time in milliseconds between dispatches (a large interval may introduce sleeps between dispatches)",
            cxxopts::value<uint32_t>()->default_value("0")
        )
        (
            "w,warmup_samples",
            "Max number of warmup samples to discard from timing statistics",
            cxxopts::value<uint32_t>()
        )
        (
            "v,timing_verbosity",
            "Timing verbosity level. 0 = show hot timings, 1 = init/cold/hot timings, 2 = show all timing info",
            cxxopts::value<uint32_t>()->default_value("0")
        )
        ;

    // DIRECTX OPTIONS
    options.add_options("DirectX")
        (
            "d,debug", 
            "Enable D3D and DML debug layers", 
            cxxopts::value<bool>()
        )
        (
            "a,adapter", 
            "Substring to match a desired DirectX adapter", 
            cxxopts::value<std::string>()->default_value(m_adapterSubstring)
        )
        (
            "s,show_adapters", 
            "Show all available DirectX adapters", 
            cxxopts::value<bool>()
        )
        (
            "q,queue_type", 
            "Type of command queue/list to use ('compute' or 'direct')", 
            cxxopts::value<std::string>()->default_value("direct")
        )
        (
            "clear_shader_caches", 
            "Clears D3D shader caches before running commands", 
            cxxopts::value<bool>()
        )
        (
            "disable_agility_sdk", 
            "Force use of the system version of D3D12Core.dll", 
            cxxopts::value<bool>()
        )
        (
            "print_hlsl_disassembly", 
            "Prints disassembled shader bytecode (HLSL dispatchables only)", 
            cxxopts::value<bool>()
        )
        (
            "post_dispatch_barriers",
            "Sets barrier types issued after every dispatch is recorded into a command list: none, uav, or uav+aliasing",
            cxxopts::value<std::string>()->default_value("uav")
        )
        (
            "u, custom_heaps",
            "Binds input and output resources are allocated from custom heaps. Write-combined caching and system memory (L0) are used when this is specified.",
            cxxopts::value<bool>()
        )

        // DxDispatch generates root signatures that are guaranteed to match HLSL source, which eliminates
        // having to write it inline in the HLSL file. DXC for Xbox precompiles shaders for Xbox (by default), 
        // but precompilation requires the root signature to be in the HLSL source itself; to allow use of the
        // DxDispatch-generated root signature we have to disable precompilation by defining `__XBOX_DISABLE_PRECOMPILE`
        // for each DXC invocation. The `xbox_allow_precompile` switch allows users to opt out of this behavior
        // and instead manually write root signatures in the HLSL source.
        (
            "xbox_allow_precompile",
            "Disables automatically defining __XBOX_DISABLE_PRECOMPILE when compiling shaders for Xbox",
            cxxopts::value<bool>()
        )
        (
            "c,pix_capture_type",
            "Type of PIX captures to take: gpu, timing, or manual.",
            cxxopts::value<std::string>()->default_value("manual")
        )
        (
            "o,pix_capture_name",
            "Name used for PIX capture files.",
            cxxopts::value<std::string>()->default_value("dxdispatch")
        )
        ;

    // ONNX OPTIONS
    options.add_options("ONNX")
        (
            "f,onnx_free_dim_name_override",
            "List of free dimension overrides by name. Can be repeated. Example: -f foo:3 -f bar:5",
            cxxopts::value<std::vector<std::string>>()->default_value({})
        )
        (
            "F,onnx_free_dim_denotation_override",
            "List of free dimension overrides by denotation. Can be repeated. Example: -F DATA_BATCH:3 -F DATA_CHANNEL:5",
            cxxopts::value<std::vector<std::string>>()->default_value({})
        )
        (
            "e,onnx_session_config_entry",
            "List of SessionOption config keys and values. Can be repeated. Example: -e foo:0 -e bar:1",
            cxxopts::value<std::vector<std::string>>()->default_value({})
        )
        (
            "b,binding_shape",
            "Explicit shapes for ONNX model tensors (-b <tensor_name>:<shape>, where <shape> is a comma-separated list of "
            "dimension sizes without whitespace). Can be repeated. Example: -b input1:2,2 -b input2:3,2",
            cxxopts::value<std::vector<std::string>>()->default_value({})
        )
        (
            "l,onnx_graph_optimization_level",
            "Sets the ONNX Runtime graph optimization level. 0 = Disabled; 1 = Basic; 2 = Extended; 99 = All",
            cxxopts::value<uint32_t>()->default_value("99")
        )
        (
            "L,onnx_logging_level",
            "Sets the ONNX Runtime logging level. 0 = Verbose; 1 = Info; 2 = Warning; 3 = Error, 4 = Fatal",
            cxxopts::value<uint32_t>()->default_value("2")
        )
        (
            "p,print_onnx_bindings",
            "Prints verbose ONNX model binding information.",
            cxxopts::value<bool>()
        )
        ;

    options.positional_help("<PATH_TO_MODEL>");

    options.parse_positional({"model"});
    auto result = options.parse(argc, argv);

    m_printHelp = result.arguments().empty() || (result.count("help") && result["help"].as<decltype(m_printHelp)>());

    if (result.count("debug")) 
    { 
        m_debugLayersEnabled = result["debug"].as<bool>(); 
    }
    
    if (result.count("dispatch_iterations"))
    {
        m_dispatchIterations = result["dispatch_iterations"].as<uint32_t>();
    }
    
    if (result.count("dispatch_repeat"))
    {
        m_dispatchRepeat = result["dispatch_repeat"].as<uint32_t>();
    }

    if (result.count("milliseconds_to_run"))
    {
        m_timeToRunInMilliseconds.emplace(result["milliseconds_to_run"].as<uint32_t>());
        m_dispatchIterations = std::numeric_limits<uint32_t>::max();   // override the "iterations" setting
    }

    if (result.count("dispatch_interval"))
    {
        m_minDispatchIntervalInMilliseconds =result["dispatch_interval"].as<uint32_t>();
    }

    if (result.count("warmup_samples"))
    {
        m_maxWarmupSamples =result["warmup_samples"].as<uint32_t>();
    }

    if (result.count("model")) 
    { 
        m_modelPath = result["model"].as<decltype(m_modelPath)>(); 
    }

    if (result.count("adapter")) 
    { 
        m_adapterSubstring = result["adapter"].as<decltype(m_adapterSubstring)>(); 
    }

    if (result.count("show_adapters")) 
    { 
        m_showAdapters = result["show_adapters"].as<bool>(); 
    }

    if (result.count("timing_verbosity"))
    {
        m_timingVerbosity = static_cast<TimingVerbosity>(result["timing_verbosity"].as<uint32_t>());
    }

    if (result.count("show_dependencies"))
    {
        m_showDependencies = result["show_dependencies"].as<bool>();
    }

    if (result.count("clear_shader_caches"))
    {
        m_clearShaderCaches = result["clear_shader_caches"].as<bool>();
    }

    if (result.count("disable_agility_sdk"))
    {
        m_disableAgilitySDK = result["disable_agility_sdk"].as<bool>();
    }

    if (result.count("print_hlsl_disassembly"))
    {
        m_printHlslDisassembly = result["print_hlsl_disassembly"].as<bool>();
    }

    if (result.count("post_dispatch_barriers"))
    {
        auto value = result["post_dispatch_barriers"].as<std::string>();

        if (value == "none")
        {
            m_uavBarrierAfterDispatch = false;
            m_aliasingBarrierAfterDispatch = false;
        }
        else if (value == "uav")
        {
            m_uavBarrierAfterDispatch = true;
            m_aliasingBarrierAfterDispatch = false;
        }
        else if (value == "uav+aliasing")
        {
            m_uavBarrierAfterDispatch = true;
            m_aliasingBarrierAfterDispatch = true;
        }
    }

    if (result.count("custom_heaps"))
    {
        m_customHeaps = result["custom_heaps"].as<bool>();
    }

    auto queueTypeStr = result["queue_type"].as<std::string>();
    if (queueTypeStr == "direct")
    {
        m_commandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    }
    else if (queueTypeStr == "compute")
    {
        m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    }
    else
    {
        throw std::invalid_argument("Unexpected value for queue_type. Must be 'compute' or 'direct'");
    }

    if (result.count("xbox_allow_precompile") && result["xbox_allow_precompile"].as<bool>())
    {
        m_forceDisablePrecompiledShadersOnXbox = false;
    }

    if (result.count("pix_capture_type")) 
    { 
        auto pixCaptureTypeStr = result["pix_capture_type"].as<std::string>();
        if (pixCaptureTypeStr == "gpu")
        {
            m_pixCaptureType = PixCaptureType::ProgrammaticGpu;
        }
        else if (pixCaptureTypeStr == "timing")
        {
            m_pixCaptureType = PixCaptureType::ProgrammaticTiming;
#ifndef _GAMING_XBOX
            throw std::invalid_argument("Programmatic timing captures are not supported");
#endif
        }
        else if (pixCaptureTypeStr == "manual")
        {
            m_pixCaptureType = PixCaptureType::Manual;
        }
        else
        {
            throw std::invalid_argument("Unexpected value for pix_capture_type. Must be 'gpu', 'timing', or 'manual'");
        }
    }

    if (result.count("pix_capture_name"))
    {
        m_pixCaptureName = result["pix_capture_name"].as<std::string>();
    }

    auto ParseFreeDimensionOverrides = [&](const char* parameterName, std::vector<std::pair<std::string, uint32_t>>& overrides)
    {
        if (result.count(parameterName))
        {
            auto freeDimOverrides = result[parameterName].as<std::vector<std::string>>(); 
            for (auto& value : freeDimOverrides)
            {
                auto splitPos = value.find(":");
                if (splitPos == std::string::npos)
                {
                    throw std::invalid_argument("Expected ':' separating name/denotation and its value");
                }
                auto overrideName = value.substr(0, splitPos);
                auto overrideValue = value.substr(splitPos + 1);
                overrides.emplace_back(overrideName, std::stoul(overrideValue));
            }
        }
    };

    ParseFreeDimensionOverrides("onnx_free_dim_name_override", m_onnxFreeDimensionNameOverrides);
    ParseFreeDimensionOverrides("onnx_free_dim_denotation_override", m_onnxFreeDimensionDenotationOverrides);

    auto ParseConfigOptionEntries = [&](const char* parameterName, std::vector<std::pair<std::string, std::string>>& overrides)
    {
        if (result.count(parameterName))
        {
            auto freeDimOverrides = result[parameterName].as<std::vector<std::string>>(); 
            for (auto& value : freeDimOverrides)
            {
                auto splitPos = value.find(":");
                if (splitPos == std::string::npos)
                {
                    throw std::invalid_argument("Expected ':' separating name/denotation and its value");
                }
                auto overrideName = value.substr(0, splitPos);
                auto overrideValue = value.substr(splitPos + 1);
                overrides.emplace_back(overrideName, overrideValue);
            }
        }
    };

    ParseConfigOptionEntries("onnx_session_config_entry", m_onnxSessionOptionConfigEntries);

    if (result.count("onnx_graph_optimization_level")) 
    { 
        m_onnxGraphOptimizationLevel = result["onnx_graph_optimization_level"].as<uint32_t>(); 
    }

    if (result.count("onnx_logging_level")) 
    { 
        m_onnxLoggingLevel = result["onnx_logging_level"].as<uint32_t>(); 
    }

    // Parse binding shapes
    if (result.count("binding_shape"))
    {
        auto bindingShapes = result["binding_shape"].as<std::vector<std::string>>(); 
        for (auto& bindingShapeArg : bindingShapes)
        {
            auto splitPos = bindingShapeArg.rfind(":");
            if (splitPos == std::string::npos)
            {
                throw std::invalid_argument("Expected ':' separating tensor name and its shape");
            }
            auto tensorName = bindingShapeArg.substr(0, splitPos);
            auto tensorShapeStr = bindingShapeArg.substr(splitPos + 1);

            // Tokenize shape string (e.g "1,2,15,8") by commas -> [1,2,15,8]
            std::vector<int64_t> shape;

            size_t startPos = 0;
            while (startPos != std::string::npos)
            {
                size_t endPos = tensorShapeStr.find(",", startPos + 1);
                auto substr = tensorShapeStr.substr(startPos, endPos - startPos);
                shape.push_back(std::stoll(substr));
                startPos = endPos == std::string::npos ? std::string::npos : endPos + 1;
            }

            m_onnxBindShapes[tensorName] = std::move(shape);
        }
    }

    if (result.count("print_onnx_bindings")) 
    { 
        m_onnxPrintVerboseBindingInfo = result["print_onnx_bindings"].as<bool>(); 
    }

    m_helpText = options.help();
}