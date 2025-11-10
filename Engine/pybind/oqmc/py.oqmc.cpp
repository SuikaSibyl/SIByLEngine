#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/wstring.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>
#include <oqmc/pmjbn.h>
#include <oqmc/pmj.h>
namespace nb = nanobind;
using namespace nb::literals;

// A handy alias for a 4D, C-contiguous float NumPy array shaped [H,W,spp,5].
using PatchSample3 = nb::ndarray<float, nb::shape<-1, -1, -1, 3>, nb::device::cpu>;
using PatchSample4 = nb::ndarray<float, nb::shape<-1, -1, -1, 4>, nb::device::cpu>;


auto generate_pmj_samples_dim3(
    int x0, int x1, int y0, int y1, int nSamples, int frame
) -> nb::object {
    // allocate via NumPy from C++ (robust & simple)
    nb::object np = nb::module_::import_("numpy");
    nb::object arr = np.attr("empty")(nb::make_tuple(x1 - x0, y1 - y0, nSamples, 3), "float32");
    // cast the Python object to our typed ndarray view
    PatchSample3 out = nb::cast<PatchSample3>(arr);

	enum DomainKey { Next, };

    auto cache = new char[oqmc::PmjBnSampler::cacheSize];
    oqmc::PmjBnSampler::initialiseCache(cache);
    // Loop over all pixels in the image.
    for(int x = x0; x < x1; ++x) {
        for(int y = y0; y < y1; ++y) {
            // Loop over all the sample indices.
            for(int index = 0; index < nSamples; ++index) {
                // Create a sampler object for the pixel domain.
                const auto domain = oqmc::PmjBnSampler(x, y, 0, index, cache);
                // Derive 'secondDomain' from 'domain' parameter.
                const auto secondDomain = domain.newDomain(DomainKey::Next);
                // Draw a sample point from the domain.
                float sample[3];
                domain.drawSample<3>(sample);
                out(x - x0, y - y0, index, 0) = sample[0];
                out(x - x0, y - y0, index, 1) = sample[1];
                out(x - x0, y - y0, index, 2) = sample[2];
            }
        }
    }

    // 9. Deallocate the sampler cache.
    delete[] cache;
    return arr; // return the NumPy array
}

auto generate_pmjbn_samples_dim2x2(
    int x0, int x1, int y0, int y1, int nSamples, int frame
) -> nb::object {
    // allocate via NumPy from C++ (robust & simple)
    nb::object np = nb::module_::import_("numpy");
    nb::object arr = np.attr("empty")(nb::make_tuple(x1 - x0, y1 - y0, nSamples, 4), "float32");
    // cast the Python object to our typed ndarray view
    PatchSample4 out = nb::cast<PatchSample4>(arr);

	enum DomainKey { Next, };

    auto cache = new char[oqmc::PmjBnSampler::cacheSize];
    oqmc::PmjBnSampler::initialiseCache(cache);
    // Loop over all pixels in the image.
    for(int x = x0; x < x1; ++x) {
        for(int y = y0; y < y1; ++y) {
            // Loop over all the sample indices.
            for(int index = 0; index < nSamples; ++index) {
                // Create a sampler object for the pixel domain.
                const auto domain = oqmc::PmjBnSampler(x, y, 0, index, cache);
                // Derive 'secondDomain' from 'domain' parameter.
                const auto secondDomain = domain.newDomain(DomainKey::Next);
                // Draw a sample point from the domain.
                float sample[2];
                domain.drawSample<2>(sample);
                out(x - x0, y - y0, index, 0) = sample[0];
                out(x - x0, y - y0, index, 1) = sample[1];
                secondDomain.drawSample<2>(sample);
                out(x - x0, y - y0, index, 2) = sample[0];
                out(x - x0, y - y0, index, 3) = sample[1];
            }
        }
    }

    // 9. Deallocate the sampler cache.
    delete[] cache;
    return arr; // return the NumPy array
}

auto generate_pmj_samples_dim2x2(
    int x0, int x1, int y0, int y1, int nSamples, int frame
) -> nb::object {
    // allocate via NumPy from C++ (robust & simple)
    nb::object np = nb::module_::import_("numpy");
    nb::object arr = np.attr("empty")(nb::make_tuple(x1 - x0, y1 - y0, nSamples, 4), "float32");
    // cast the Python object to our typed ndarray view
    PatchSample4 out = nb::cast<PatchSample4>(arr);

	enum DomainKey { Next, };

    auto cache = new char[oqmc::PmjSampler::cacheSize];
    oqmc::PmjSampler::initialiseCache(cache);
    // Loop over all pixels in the image.
    for(int x = x0; x < x1; ++x) {
        for(int y = y0; y < y1; ++y) {
            // Loop over all the sample indices.
            for(int index = 0; index < nSamples; ++index) {
                // Create a sampler object for the pixel domain.
                const auto domain = oqmc::PmjSampler(x, y, 0, index, cache);
                // Derive 'secondDomain' from 'domain' parameter.
                const auto secondDomain = domain.newDomain(DomainKey::Next);
                // Draw a sample point from the domain.
                float sample[2];
                domain.drawSample<2>(sample);
                out(x - x0, y - y0, index, 0) = sample[0];
                out(x - x0, y - y0, index, 1) = sample[1];
                secondDomain.drawSample<2>(sample);
                out(x - x0, y - y0, index, 2) = sample[0];
                out(x - x0, y - y0, index, 3) = sample[1];
            }
        }
    }

    // 9. Deallocate the sampler cache.
    delete[] cache;
    return arr; // return the NumPy array
}

NB_MODULE(pyoqmc, m) {
    // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    // ┃ log                                                                       ┃
    // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    m.def("generate_pmj_samples_dim3", &generate_pmj_samples_dim3);
    m.def("generate_pmjbn_samples_dim2x2", &generate_pmjbn_samples_dim2x2);
    m.def("generate_pmj_samples_dim2x2", &generate_pmj_samples_dim2x2);
}