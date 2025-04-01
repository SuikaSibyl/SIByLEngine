#ifndef _SRENDERER_COMMON_COLORMAPS_HEADER_
#define _SRENDERER_COMMON_COLORMAPS_HEADER_

/**
 * The following implementation is adopted from mattz'implmentation in Shadertoy.
 * @url: https://www.shadertoy.com/view/WlfXRN
 * And you can also find the preview of the colormaps in the above link.
 *
 * The original license information is as follows:
 * | fitting polynomials to matplotlib colormaps
 * |
 * | License CC0 (public domain)
 * |   https://creativecommons.org/share-your-work/public-domain/cc0/
 * |
 * | feel free to use these in your own work!
 * |
 * | similar to https://www.shadertoy.com/view/XtGGzG but with a couple small differences:
 * |
 * |  - use degree 6 instead of degree 5 polynomials
 * |  - use nested horner representation for polynomials
 * |  - polynomials were fitted to minimize maximum error (as opposed to least squares)
 * |
 * | data fitted from https://github.com/BIDS/colormap/blob/master/colormaps.py
 * | (which is licensed CC0)
 */

float3 viridis(float t) {
    const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float3 plasma(float t) {
    const float3 c0 = float3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    const float3 c1 = float3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    const float3 c2 = float3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
    const float3 c3 = float3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    const float3 c4 = float3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    const float3 c5 = float3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    const float3 c6 = float3(-3.658713842777788, -22.93153465461149, 18.19190778539828);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float3 magma(float t) {
    const float3 c0 = float3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const float3 c1 = float3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
    const float3 c2 = float3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
    const float3 c3 = float3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
    const float3 c4 = float3(52.17613981234068, -27.94360607168351, 12.94416944238394);
    const float3 c5 = float3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
    const float3 c6 = float3(18.65570506591883, -11.48977351997711, -5.601961508734096);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float3 inferno(float t) {
    const float3 c0 = float3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const float3 c1 = float3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const float3 c2 = float3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const float3 c3 = float3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const float3 c4 = float3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const float3 c5 = float3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const float3 c6 = float3(25.13112622477341, -12.24266895238567, -23.07032500287172);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

float colormap_red(float x) {
	if (x < 0.09790863520700754) {
		return 5.14512820512820E+02 * x + 1.64641025641026E+02;
	} else if (x < 0.2001887081633112) {
		return 2.83195402298854E+02 * x + 1.87288998357964E+02;
	} else if (x < 0.3190117539655621) {
		return 9.27301587301214E+01 * x + 2.25417989417999E+02;
	} else if (x < 0.500517389125164) {
		return 255.0;
	} else if (x < 0.6068377196788788) {
		return -3.04674876847379E+02 * x + 4.07495073891681E+02;
	} else if (x < 0.9017468988895416) {
		return (1.55336390191951E+02 * x - 7.56394659038288E+02) * x + 6.24412733169483E02;
	} else {
		return -1.88350769230735E+02 * x + 2.38492307692292E+02;
	}
}

float colormap_green(float x) {
    if (x < 0.09638568758964539) {
		return 4.81427692307692E+02 * x + 4.61538461538488E-01;
	} else if (x < 0.4987066686153412) {
		return ((((3.25545903568267E+04 * x - 4.24067109461319E+04) * x + 1.83751375886345E+04) * x - 3.19145329617892E+03) * x + 8.08315127034676E+02) * x - 1.44611527812961E+01;
 	} else if (x < 0.6047312345537269) {
 		return -1.18449917898218E+02 * x + 3.14234811165860E+02;
 	} else if (x < 0.7067635953426361) {
 		return -2.70822112753102E+02 * x + 4.06379036672115E+02;
 	} else {
 		return (-4.62308723214883E+02 * x + 2.42936159122279E+02) * x + 2.74203431802418E+02;
	}
}

float colormap_blue(float x) {
	if (x < 0.09982818011951204) {
		return 1.64123076923076E+01 * x + 3.72646153846154E+01;
	} else if (x < 0.2958717460833126) {
		return 2.87014675052409E+02 * x + 1.02508735150248E+01;
	} else if (x < 0.4900527540014758) {
		return 4.65475113122167E+02 * x - 4.25505279034673E+01;
	} else if (x < 0.6017014681258838) {
		return 5.61032967032998E+02 * x - 8.93789173789407E+01;
	} else if (x < 0.7015737100463595) {
		return -1.51655677655728E+02 * x + 3.39446886446912E+02;
	} else if (x < 0.8237156500567735) {
		return -2.43405347593559E+02 * x + 4.03816042780725E+02;
	} else {
		return -3.00296889157305E+02 * x + 4.50678495922638E+02;
	}
}

float3 CBRdYiBu(float x) {
    x = 1.f - x;
	float r = clamp(colormap_red(x) / 255.0, 0.0, 1.0);
	float g = clamp(colormap_green(x) / 255.0, 0.0, 1.0);
	float b = clamp(colormap_blue(x) / 255.0, 0.0, 1.0);
	return float3(r, g, b);
}

#endif // _SRENDERER_COMMON_COLORMAPS_HEADER_