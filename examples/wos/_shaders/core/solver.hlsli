#ifndef _WOS_PDE_SOLVER_HLSLI_
#define _WOS_PDE_SOLVER_HLSLI_

#include "pde.hlsli"
#include "distributions.hlsli"

struct WalkState<let DimIn : int, let DimOut : int> {
    typealias input_t = vector<float, DimIn>;
    typealias return_t = vector<float, DimOut>;

    input_t current_pt;
    return_t total_source_contribution;
    float potential;
    int walkLength;
    bool isSplitState;

    __init(input_t currentPt_, float potential_,
           int walkLength_, bool isSplitState_, return_t initVal) {
        current_pt = currentPt_;
        total_source_contribution = initVal;
        potential = potential_;
        walkLength = walkLength_;
        isSplitState = isSplitState_;
    }
}

struct SampleStatistics<let DimIn : int, let DimOut : int, let BounceRCV : int> {
    typealias input_t = vector<float, DimIn>;
    typealias return_t = vector<float, DimOut>;

    return_t solutionSum;
    // return_t solutionMean, solutionM2;
    // return_t gradientMean[DimIn], gradientM2[DimIn];
    // return_t firstSourceContributionSum;
    int nSolutionContributions, nGradientContributions;
    int totalWalkLength, totalSplits;

    Array<return_t, BounceRCV> contrib_per_bounce;
    Array<return_t, BounceRCV> cv_per_bounce;
    return_t extra_contrib;
    
    __init(return_t initVal) { reset(initVal); }

    // resets the statistics
    [mutating]
    void reset(return_t initVal) {
        // solutionMean = initVal;
        // solutionM2 = initVal;
        // for (int i = 0; i < DimIn; i++) {
        //     gradientMean[i] = initVal;
        //     gradientM2[i] = initVal;
        // }
        // firstSourceContributionSum = initVal;
        solutionSum = initVal;
        nSolutionContributions = 0;
        nGradientContributions = 0;
        totalWalkLength = 0;
        totalSplits = 0;

        for (int bounce = 0; bounce < BounceRCV; ++bounce) {
            contrib_per_bounce[bounce] = initVal;
            cv_per_bounce[bounce] = initVal;
        }
        extra_contrib = initVal;
    }

    return_t getSolution() {
        if (nSolutionContributions == 0) { return return_t(0); }
        return solutionSum / nSolutionContributions;
    }

    return_t getSolution_rcv() {
        if (nSolutionContributions == 0) { return return_t(0); }
        return_t solution = extra_contrib / nSolutionContributions;
        // accumulate contributions from each bounce
        for (int i = 0; i < BounceRCV; ++i) {
            return_t contrib = contrib_per_bounce[i] / nSolutionContributions;
            return_t cv = select(cv_per_bounce[i] == 0, return_t(1), cv_per_bounce[i] / nSolutionContributions);
            solution += contrib / cv;
        }
        return solution;
    }

    // adds solution contribution to running sum
    [mutating]
    void addSolutionContribution(return_t contribution) {
        nSolutionContributions += 1;
        solutionSum += contribution;
    }

    [mutating]
    void addSolutionPerBounce(int bounce, return_t contribution, return_t cv) {
        if (bounce < BounceRCV) {
            contrib_per_bounce[bounce] += contribution;
            cv_per_bounce[bounce] += cv;
        }
        else {
            extra_contrib += contribution;
        }
    }

    // // adds gradient contribution to running sum
    // [mutating]
    // void addGradientContribution(
    //     return_t boundaryContribution,
    //     return_t sourceContribution) {
    //     nGradientContributions += 1;
    //     for (int i = 0; i < DimIn; i++) {
    //         update(boundaryContribution[i] + sourceContribution[i],
    //                gradientMean[i], gradientM2[i], nGradientContributions);
    //     }
    // }

    // // add first source contribution
    // [mutating] void addFirstSourceContribution(return_t contribution) {
    //     firstSourceContributionSum += contribution; }
    // // adds walk length to running sum
    // [mutating] void addWalkLength(int length) { totalWalkLength += length; }
    // // adds walk length to running sum
    // [mutating] void addSplits(int nSplits) { totalSplits += nSplits; }
    // returns the solution estimate

    // // returns the estimate of the solution variance
    // return_t getSolutionVariance() {
    //     int N = max(1, nSolutionContributions - 1);
    //     return solutionM2 / N;
    // }
    // // returns the gradient estimate
    // return_t[DimIn] getGradient() { return gradientMean; }

    // // returns the estimate of the gradient variance
    // return_t[DimIn] getGradientVariance() {
    //     int N = max(1, nGradientContributions - 1);
    //     return_t[DimIn] variance;
    //     for (int i = 0; i < DimIn; i++) {
    //         variance[i] = gradientM2[i] / N;
    //     }
    //     return variance;
    // }

    // // returns first source contribution
    // return_t getFirstSourceContribution() {
    //     int N = max(1, nSolutionContributions);
    //     return firstSourceContributionSum / N;
    // }

    // // returns the number of solution contributions
    // int getSolutionContributionsCount() { return nSolutionContributions; }
    // // returns the number of gradient contributions
    // int getGradientContributionsCount() { return nGradientContributions; }
    // // returns the mean walk length
    // float getMeanWalkLength() {
    //     int N = max(1, nSolutionContributions);
    //     return (float)totalWalkLength / N;
    // }
    // // returns the total number of splits
    // int getTotalSplits() { return totalSplits; }

    // // updates statistics
    // void update(
    //     return_t contribution,
    //     inout return_t mean,
    //     inout return_t M2, int N) {
    //     return_t delta = contribution - mean;
    //     mean += delta / N;
    //     return_t delta2 = contribution - mean;
    //     M2 += delta * delta2;
    // }
};

struct WoS
<let DimIn : int,
           let DimOut : int,
           let BounceRCV : int,
           PDE_t : IPde<DimIn, DimOut>,
           Green_t : IGreensFnBall<DimIn>,
           Boundary_t : IBoundary<DimIn, DimOut>>
{   
    typealias input_t = vector<float, DimIn>;
    typealias return_t = vector<float, DimOut>;
    typealias walk_state_t = WalkState<DimIn, DimOut>;

    static void solve(
        PDE_t pde,
        Boundary_t boundary,
        inout Green_t greensFn,
        float epsilonShell,
        int maxWalkLength,
        input_t pt, int nWalks,
        return_t initVal,
        inout SampleStatistics<DimIn, DimOut, BounceRCV> statistics,
        inout RandomSamplerState rng
    ) {
        float initDist = boundary.distance(pt);
        // run no walks if we are outside the domain
        if (initDist < 0) { nWalks = 0; }
        // run a single random walk if the distance to the boundary 
        // is smaller than the epsilon shell
        else if (initDist <= epsilonShell) { nWalks = 1; }
        
        // perform random walks
        for (int w = 0; w < nWalks; w++) {
            // initialize the walk state
            walk_state_t state = walk_state_t(pt, 1.0f, 0, false, initVal);
            return_t totalContribution = initVal;
            bool anySuccess = false;
            int splitsPerformed = -1;
            
            // splitsPerformed++;
            bool success = walkConstantCoeffs(pde, boundary, greensFn, epsilonShell, maxWalkLength, initVal, state, statistics, rng);
            totalContribution += state.total_source_contribution;

            if (success) {
                // get the boundary contribution
                return_t boundaryContribution = pde.dirichlet(state.current_pt);
                // update the total walk contribution
                totalContribution += state.potential * boundaryContribution;
                statistics.extra_contrib += state.potential * boundaryContribution;
            }
            
            statistics.addSolutionContribution(totalContribution);
        }
    }
    
    // performs a single random walk starting at the input point for a constant coeff problem.
    // returns false if the walk does not terminate
    static bool walkConstantCoeffs(
        PDE_t pde,
        Boundary_t boundary,
        inout Green_t greensFn,
        float epsilonShell,
        int maxWalkLength,
        return_t initVal,
        inout WalkState<DimIn, DimOut> state,
        // std::queue<WalkState<T, DIM>> &queue
        inout SampleStatistics<DimIn, DimOut, BounceRCV> statistics,
        inout RandomSamplerState rng
    ) {
        // recursively perform a random walk till it reaches the boundary
        float dist = boundary.distance(state.current_pt);
        float diffusion = pde.diffusion(state.current_pt);
        
        while (dist > epsilonShell) {
            // update the sphere center and radius
            greensFn.update_ball(state.current_pt, dist);

            // { // compute the source contribution inside the sphere
            //     float pdf; greensFn.sample_volume(pdf, rng);
            //     float greensFnNorm = greensFn.norm();
            //     return_t sourceContribution = greensFnNorm * pde.source(greensFn.get_sample_inside_ball()) / diffusion;
            //     float w_green = pdf / (pde.pdf_source(greensFn.get_sample_inside_ball()) + pdf);

            //     if (state.walkLength > 0) {
            //         state.total_source_contribution += sourceContribution * state.potential;    
                    
            //     }
            // }
            { // sample the source contribution
                source_sample<DimIn, DimOut> sample_s = pde.sample_source(GetNextRandomUint(rng));
                float greensFnValue = greensFn.evaluate(distance(sample_s.position, state.current_pt));
                return_t sourceContribution = greensFnValue * pde.source(sample_s.position) / (sample_s.pdf_sample * diffusion);
                float w_nee = sample_s.pdf_sample / (greensFn.pdf_sample_volume(sample_s.position) + sample_s.pdf_sample);
                
                if (true) {
                    state.total_source_contribution += sourceContribution * state.potential;

                    statistics.addSolutionPerBounce(
                        state.walkLength,
                        sourceContribution,
                        sample_s.pdf_channels / sample_s.pdf_sample);
                }
            }

            // sample a point uniformly on the sphere,
            // and update the current position of the walk
            float pdf; state.current_pt = greensFn.sample_surface(pdf, rng);
            // update the potential
            state.potential *= greensFn.poisson_kernel() / pdf;

            // // apply the weight window
            // bool terminate = applyWeightWindow(window, initVal, state, queue);
            // if (terminate) return true;

            // compute the distance of the currentPt to the boundary
            dist = boundary.distance(state.current_pt);

            // update the walk length and break if the max walk length is exceeded
            state.walkLength++;
            if (state.walkLength > maxWalkLength) {
                // Maximum walk length exceeded
                break;
            }
        }

        return dist <= epsilonShell;
    }

    // performs a single random walk starting at the input point for a variable coeff problem;
    // returns false if the walk does not terminate
    static bool walkDeltaTracking(
        PDE_t pde,
        Boundary_t boundary,
        inout Green_t greensFn,
        float epsilonShell,
        int maxWalkLength,
        return_t initVal,
        inout WalkState<DimIn, DimOut> state,
        // std::queue<WalkState<T, DIM>> &queue
        inout RandomSamplerState rng
    ) {
        // recursively perform a random walk till it reaches the boundary
        float dist = boundary.distance(state.current_pt);
        float diffusion = pde.diffusion(state.current_pt);

        while (dist > epsilonShell) {
            // update the sphere center and radius
            greensFn.update_ball(state.current_pt, dist);

            // compute the source contribution inside the sphere
            float pdf; greensFn.sample_volume(pdf, rng);
            float greensFnNorm = greensFn.norm();

            input_t sample_inside = greensFn.get_sample_inside_ball();
            float diffusionWeight = 1.0f / sqrt(diffusion * pde.diffusion(sample_inside));
            return_t sourceContribution = greensFnNorm * diffusionWeight * pde.source(sample_inside);
            state.total_source_contribution += state.potential * sourceContribution;
            
            // decide whether to sample the volume or boundary integral
            bool sampleVolumeIntegral = GetNextRandom(rng) < pde.get_bound() * greensFnNorm;
            if (sampleVolumeIntegral) {
                // sample a point inside the sphere using the centered greens function,
                // and update the current position of the walk
                float pdf;
                state.current_pt = greensFn.sample_volume(pdf, rng);
                // update the potential with the collision probability
                state.potential *= (1.0f - pde.transformed_absorption(state.current_pt) / pde.get_bound());
            } else {
                // sample a point uniformly on the sphere, and update the current
                // position of the walk
                float pdf;
                state.current_pt = greensFn.sample_surface(pdf, rng);
            }

            // update the potential with the diffusion weight
            float diffusionCurrentPt = pde.diffusion(state.current_pt);
            state.potential *= sqrt(diffusionCurrentPt / diffusion);
            diffusion = diffusionCurrentPt;
            
            // // apply the weight window
            // bool terminate = applyWeightWindow(window, initVal, state, queue);
            // if (terminate) return true;

            // compute the distance of the currentPt to the boundary
            dist = boundary.distance(state.current_pt);

            // update the walk length and break if the max walk length is exceeded
            state.walkLength++;
            if (state.walkLength > maxWalkLength) {
                // Maximum walk length exceeded
                break;
            }
        }
        
        return dist <= epsilonShell;
    }

    // // terminates, splits or continues a walk based on its potential value
    // bool applyWeightWindow(WeightWindow window, T initVal, WalkState<T, DIM> &state,
    //                        std::queue<WalkState<T, DIM>> &queue) const {
    //     if (window.upperBound > window.lowerBound) {
    //         if (state.potential > window.upperBound) {
    //             // split the walk
    //             float potentialLeft = state.potential - window.upperBound;
    //             state.potential = window.upperBound;
    //             WalkState<T, DIM> splitState(state.currentPt, state.potential, 0, true, initVal);

    //             while (potentialLeft > window.upperBound) {
    //                 potentialLeft -= window.upperBound;
    //                 queue.emplace(splitState);
    //             }

    //             if (sampleUniform() < potentialLeft / window.upperBound) {
    //                 queue.emplace(splitState);
    //             }
    //         } else if (state.potential < window.lowerBound) {
    //             // kill the walk using russian roulette
    //             float survivalProb = state.potential / window.lowerBound;
    //             if (survivalProb < sampleUniform()) {
    //                 state.potential = 0.0f;
    //                 return true;
    //             }

    //             state.potential = window.lowerBound;
    //         }
    //     }

    //     return false;
    // }
};

#endif // _WOS_PDE_SOLVER_HLSLI_