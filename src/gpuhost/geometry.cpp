// SPDX-License-Identifier: GPL-2.0-only
#include "gpuhost/geometry.hpp"

namespace nss::gpuhost {

RasterGrid make_raster_grid(int width, int height, int block, int step) {
    RasterGrid grid{0, 0, step, width - block, height - block};
    if (width < block || height < block || step < 1) return grid;
    grid.nx = (width - block + step - 1) / step + 1;
    grid.ny = (height - block + step - 1) / step + 1;
    return grid;
}

}  // namespace nss::gpuhost
