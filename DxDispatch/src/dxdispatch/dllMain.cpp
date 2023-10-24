#include "pch.h"
#include "dxDispatchWrapper.h"

volatile std::atomic<ULONG> g_ModuleCount = 0;

ULONG AddDllRef()
{
    return ++g_ModuleCount;
}

ULONG ReleaseDllRef()
{
    return --g_ModuleCount;
}

BOOL CanUnload()
{
    return (g_ModuleCount == 0 ? TRUE : FALSE);
}

class StringToArgs
{
public:
    StringToArgs(LPCSTR commandline)
    {
        //Add additional parameter for parser
        std::vector<char> arg({'t','e','.','e','x','e'});
        bool isQuote = false;
        auto AddWord = [&](){
            if (arg.size() > 0)
            {
                auto argRawString = (char*)calloc((arg.size() + 1), sizeof(char));
                memcpy(argRawString, arg.data(), arg.size());
                m_args.push_back(argRawString);
                arg.clear();
            }
        };
        AddWord();
        for (size_t i = 0, c = strlen(commandline); i <= c; i++)
        {
            if (i == c)
            {
                AddWord();
            }
            else if (isQuote)
            {
                if (commandline[i] == '\"')
                {
                    AddWord();
                    isQuote = false;
                }
                else
                {
                    arg.push_back(commandline[i]);
                }
            }
            else
            {
                if (commandline[i] == ' ' || commandline[i] == '\t')
                {
                    AddWord();
                }
                else if (commandline[i] == '\"')
                {
                    isQuote = true;
                }
                else
                {
                    arg.push_back(commandline[i]);
                }
            }
        }
    }
    ~StringToArgs()
    {
        for(auto &arg : m_args)
        {
            if (arg)
            {
                free(arg);
                arg = nullptr;
            }
        }
        m_args.clear();
    }
    size_t size()
    {
        return m_args.size();
    }
    char** data()
    {
        return m_args.data();
    }

private:
    std::vector<char*> m_args;
};


 STDAPI CreateDxDispatchFromString(
   _In_           PCSTR args,                         // DxDispatch commandArgs
   _In_           PCSTR jsonConfig,                   // DxDisPatch Json contents
   _In_opt_       IUnknown *adapter,                 // will use DxDispatch logic to pick device if nullptr
   _In_opt_       IDxDispatchLogger *customLogger,   // will log to console if not overwritten
   _COM_Outptr_   IDxDispatch **dxDispatch)
{

    StringToArgs stringToArgs(args);
    return DxDispatch::CreateDxDispatchFromJsonString(
        (int)stringToArgs.size(), 
        stringToArgs.data(), 
        jsonConfig,
        adapter,
        customLogger,
        dxDispatch);
}

STDAPI CreateDxDispatchFromArgs(
   _In_           int argc,
   _In_           char** argv,
   _In_           PCSTR jsonConfig,                   // DxDisPatch Json contents
   _In_opt_       IUnknown *adapter,                  // will use DxDispatch logic to pick device if nullptr
   _In_opt_       IDxDispatchLogger *customLogger,    // will log to console if not overwritten
   _COM_Outptr_   IDxDispatch **dxDispatch)
{
    return DxDispatch::CreateDxDispatchFromJsonString(
        argc,
        argv, 
        jsonConfig, 
        adapter,
        customLogger,
        dxDispatch);
}


extern "C" BOOL WINAPI DllMain(HINSTANCE hInstance, ULONG reason, CONTEXT* context)
{
    HRESULT hr = S_OK;
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hInstance);
        break;

    case DLL_PROCESS_DETACH:
        break;
    };
    return SUCCEEDED(hr);
}