#include <oqmc/pmjbn.h>

namespace oqmc {
    auto test() {
        // 2. Initialise the sampler cache.
        auto cache = new char[oqmc::PmjBnSampler::cacheSize];
        oqmc::PmjBnSampler::initialiseCache(cache);
        int resolution = 256;
        int nSamples = 64;

        // 3. Loop over all pixels in the image.
        for(int x = 0; x < resolution; ++x)
        {
            for(int y = 0; y < resolution; ++y)
            {
                // 4. Loop over all the sample indices.
                for(int index = 0; index < nSamples; ++index)
                {
                    // 5. Create a sampler object for the pixel domain.
                    const auto domain = oqmc::PmjBnSampler(x, y, 0, index, cache);

                    // 6. Draw a sample point from the domain.
                    float sample[2];
                    domain.drawSample<2>(sample);

                    // 7. Offset the point into the pixel.
                    const auto xOffset = x + sample[0];
                    const auto yOffset = y + sample[1];

                    // 8. Add value to the pixel if within disk.
                    if(xOffset * xOffset + yOffset * yOffset < resolution * resolution)
                    {
                        // image[x * resolution + y] += 1.0f / nSamples;
                    }
                }
            }
        }

        // 9. Deallocate the sampler cache.
        delete[] cache;
    }
}