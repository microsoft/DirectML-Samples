// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define NOMINMAX

#include <wil/result.h>
#include <wil/resource.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <optional>

#ifdef _GAMING_XBOX_SCARLETT
#include <d3dx12_xs.h>
#include <d3d12_xs.h>
#else
#include "d3dx12.h" // The D3D12 Helper Library that you downloaded.
#include <dxgi1_4.h>
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
#endif

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
#include "DirectMLX.h"

#include <d3d12sdklayers.h>