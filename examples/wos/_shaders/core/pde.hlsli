#ifndef _WOS_PDE_DEFINITION_HLSLI_
#define _WOS_PDE_DEFINITION_HLSLI_

struct source_sample<let DimIn:int, let DimOut:int> {
    vector<float, DimIn> position;
    float pdf_sample;
    vector<float, DimOut> pdf_channels;
};

struct boundary_sample<let DimIn:int, let DimOut:int> {
    vector<float, DimIn>  position;
    vector<float, DimIn>  normal;
    vector<float, DimOut> values;
    float pdf;
};

/**
 * IBoundary defines the geometry for a boundary condition.
 */
interface IBoundary<let DimIn : int, let DimOut : int> {
    // returns the distance from the point x to the boundary
    static float distance(vector<float, DimIn> x);
    // return the closest point on the boundary to the point x
    static vector<float, DimIn> closest_point(vector<float, DimIn> x);
    // return the sample of the boundary
    boundary_sample<DimIn, DimOut> sample_boundary(uint seed);
}

interface IPde<let DimIn:int, let DimOut:int> {
    // returns the dirichlet boundary value at the input point
    vector<float, DimOut> dirichlet(vector<float, DimIn> x);
    // returns the source value at the input point
    vector<float, DimOut> source(vector<float, DimIn> x);
    // returns the absorption coefficient value at the input point
    vector<float, DimOut> absorption(vector<float, DimIn> x);
    // returns the diffusion coefficient value at the input point
    float diffusion(vector<float, DimIn> x);
    // returns the absorption coefficient value for the transformed problem at the input point
    float transformed_absorption(vector<float, DimIn> x);
    // returns the bound of the domain
    float get_bound();
    // return the sample of the source
    source_sample<DimIn, DimOut> sample_source(uint seed);
    // return the sample of the boundary
    boundary_sample<DimIn, DimOut> sample_boundary(uint seed);
    // return the pdf of sampling a point from the source
    float pdf_source(vector<float, DimIn> p);
    // return if the PDE is interior only
    bool interior_only();
}

interface IPDEImpl<let DimIn:int, let DimOut:int> {
    // returns the diffusion coefficient value at the input point
    static vector<float, DimOut> dirichlet_impl(vector<float, DimIn> x);
    // returns the source value at the input point
    static vector<float, DimOut> source_impl(vector<float, DimIn> x);
    // returns the diffusion coefficient value at the input point
    static float diffusion_impl(
        vector<float, DimIn> x,
        bool isGradient = false, bool isLaplacian = false,
        out vector<float, DimIn> gradientStub, out float laplacianStub);
    // returns the absorption coefficient value at the input point
    static vector<float, DimOut> absorption_impl(vector<float, DimIn> x);
    // returns the bound of the domain
    static float get_bound_impl();
    // return the sample of the source
    static source_sample<DimIn, DimOut> sample_source_impl(uint seed);
    // return a sample of the boundary
    static boundary_sample<DimIn, DimOut> sample_boundary_impl(uint seed);
    // return the pdf of sampling a point from the source
    static float pdf_source_impl(vector<float, DimIn> p);
    // return if the PDE is interior only
    static bool interior_only_impl();
}

struct PDE_WRAPPER<let DimIn:int, let DimOut:int, 
 PDEImpl : IPDEImpl<DimIn, DimOut>> : IPde<DimIn, DimOut>
{
    typealias input_t = vector<float, DimIn>;
    typealias return_t = vector<float, DimOut>;

    // returns the dirichlet boundary value at the input point
    return_t dirichlet(input_t x) { return PDEImpl::dirichlet_impl(x); }
    // returns the source value at the input point
    return_t source(input_t x) { return PDEImpl::source_impl(x); }
    // returns the absorption coefficient value at the input point
    return_t absorption(input_t x) { return PDEImpl::absorption_impl(x); }
    // returns the bound of the domain
    float get_bound() { return PDEImpl::get_bound_impl(); }
    // returns the absorption coefficient value for the transformed problem at the input point
    float transformed_absorption(input_t x) {
        input_t diffusionGradient;
        float diffusionLaplacian;
        float diffusionCoeff = PDEImpl::diffusion_impl(x, true, true, diffusionGradient, diffusionLaplacian);
        float absorptionCoeff = PDEImpl::absorption_impl(x)[0]; // fix make vector-valued
        float absorptionTerm = absorptionCoeff / diffusionCoeff;
        float diffusionGradientTerm = 0.25f * dot(diffusionGradient, diffusionGradient) / (diffusionCoeff * diffusionCoeff);
        float diffusionLaplacianTerm = 0.5f * diffusionLaplacian / diffusionCoeff;
        return absorptionTerm + diffusionLaplacianTerm - diffusionGradientTerm;
    }
    // returns the diffusion coefficient value at the input point
    float diffusion(input_t x) {
        input_t diffusionGradient;
        float diffusionLaplacian;
        return PDEImpl::diffusion_impl(x, false, false, diffusionGradient, diffusionLaplacian);
    }
    // return the sample of the source
    source_sample<DimIn, DimOut> sample_source(uint seed) {
        return PDEImpl::sample_source_impl(seed);
    }
    // return the sample of the boundary
    boundary_sample<DimIn, DimOut> sample_boundary(uint seed) {
        return PDEImpl::sample_boundary_impl(seed);
    }
    // return the pdf of sampling a point from the source
    float pdf_source(vector<float, DimIn> p) {
        return PDEImpl::pdf_source_impl(p);
    }
    // return if the PDE is interior only
    bool interior_only() { return PDEImpl::interior_only_impl(); }
}

float3 sdf_colormap(float d) {
    float3 col = (d > 0.0) 
        ? float3(0.9, 0.6, 0.3) 
        : float3(0.65, 0.85, 1.0);
    col *= 1.0 - exp(-6.0 * abs(d));
    col *= 0.8 + 0.2 * cos(150.0 * d);
    col = lerp(col, float3(1.0, 1.0, 1.0), 
        1.0 - smoothstep(0.0, 0.01, abs(d)));
    return col;
}

#endif // _WOS_PDE_DEFINITION_HLSLI_