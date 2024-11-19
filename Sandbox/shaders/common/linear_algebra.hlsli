#ifndef _SRENDERER_LINEAR_ALGEBRA_HEADER_
#define _SRENDERER_LINEAR_ALGEBRA_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix) {
    const float determinant = matrix._11 * matrix._22 - matrix._12 * matrix._21;
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix[1][1] * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix[0][1] * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix[1][0] * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix[0][0] * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to calculate the inverse of a 2x2 matrix
float2x2 Inverse2x2(in_ref(float2x2) matrix, float determinant) {
    const float reciprocalDeterminant = 1.0 / determinant;
    float2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix._22 * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix._12 * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix._21 * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to calculate the inverse of a 2x2 matrix
double2x2 Inverse2x2(in_ref(double2x2) matrix, double determinant) {
    const double reciprocalDeterminant = 1.0 / determinant;
    double2x2 inverseMatrix;
    inverseMatrix[0][0] = matrix._22 * reciprocalDeterminant;
    inverseMatrix[0][1] = -matrix._12 * reciprocalDeterminant;
    inverseMatrix[1][0] = -matrix._21 * reciprocalDeterminant;
    inverseMatrix[1][1] = matrix._11 * reciprocalDeterminant;
    return inverseMatrix;
}

// Function to perform Cholesky decomposition on a 2x2 matrix
float2x2 CholeskyDecomposition2x2(in_ref(float2x2) matrix) {
    float2x2 result;
    result[0][0] = sqrt(matrix._11);
    result[0][1] = matrix._12 / result._11;
    result[1][0] = 0.f;
    result[1][1] = sqrt(matrix._22 - result._12 * result._12);
    return result;
}

// Function to perform Cholesky decomposition on a 2x2 matrix
double2x2 CholeskyDecomposition2x2(in_ref(double2x2) matrix) {
    double2x2 result;
    result[0][0] = sqrt(matrix._11);
    result[0][1] = matrix._12 / result._11;
    result[1][0] = 0.;
    result[1][1] = sqrt(matrix._22 - result._12 * result._12);
    return result;
}

float QuadraticForm(in_ref(float2x2) matrix, in_ref(float2) vector) {
    return dot(vector, mul(matrix, vector));
}

// Function to calculate the inverse of a 3x3 matrix.
[Differentiable]
float3x3 Inverse3x3(in_ref(float3x3) m) {
    const float oneOverDet = 1.f / determinant(m);
    float3x3 result;
    result[0][0] = +(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * oneOverDet;
    result[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * oneOverDet;
    result[0][2] = +(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * oneOverDet;
    result[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * oneOverDet;
    result[1][1] = +(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * oneOverDet;
    result[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * oneOverDet;
    result[2][0] = +(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * oneOverDet;
    result[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * oneOverDet;
    result[2][2] = +(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * oneOverDet;
    return result;
}

[Differentiable]
float3x3 identity3x3() {
    return float3x3(1, 0, 0,
                    0, 1, 0,
                    0, 0, 1);
}

[Differentiable]
float3x3 outer_product(float3 v0, float3 v1) {
    return float3x3(
        v0 * v1.x,
        v0 * v1.y,
        v0 * v1.z);
}

[Differentiable]
float4x4 inverse(float4x4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

// See http://metamerist.blogspot.com/2006/10/linear-algebra-for-graphics-geeks-svd.html
float2 SVD_AbsScaleOnly(float2x2 matrixIn) {
    float4 vectorized = float4(matrixIn[0], matrixIn[1]);
    float Q = dot(vectorized, vectorized);
    float R = determinant(matrixIn); // ad-bc
    float discriminant = sqrt(Q * Q - 4 * R * R);
    float2 scale = sqrt(float2(Q + discriminant, Q - discriminant) / 2);
    return scale;
}

float2x2 SVD_EigenvectorGivenEigenvalue(float2x2 matrixIn, float2 eigenvalue) {
    eigenvalue = eigenvalue * eigenvalue;
    float M01 = dot(matrixIn[0], matrixIn[1]);
    float M00 = dot(matrixIn[0], matrixIn[0]);
    float2 eigenVectorNumerator = float2(M01, M01);
    float2 eigenVectorDenominator = eigenvalue - float2(M00, M00);
    float2 eigenFraction = eigenVectorNumerator / eigenVectorDenominator;
    float2x2 eigenMatrix = {
        normalize(float2(eigenFraction.x, 1)),
        normalize(float2(eigenFraction.y, 1)),
    };
    eigenMatrix = transpose(eigenMatrix);
    return eigenMatrix;
}

void svd_partial(float2x2 matrixIn, out float2x2 U, out float2 scale) {
    scale = SVD_AbsScaleOnly(matrixIn);
    U = SVD_EigenvectorGivenEigenvalue(matrixIn, scale);
}

void svd(float2x2 matrixIn, out float2x2 U, out float2 scale, out float2x2 V) {
    svd_partial(matrixIn, U, scale);
    float2 inverseScale = 1.0 / scale;
    float2x2 E = {
        inverseScale.x, 0,
        0, inverseScale.y,
    };
    V = mul(E, mul(transpose(U), matrixIn));
}

float2x2 pinv(float2x2 matrix) {
    float2x2 U; float2 S; float2x2 Vh;
    svd(matrix, U, S, Vh);
    // Invert the singular values in S to create Sigma^+
    const float2x2 S_pinv = float2x2(1.0 / S[0], 0, 0, 1.0 / S[1]);
    // Compute the pseudo - inverse as V *Sigma ^ +*U ^ T
    return mul(transpose(Vh), mul(S_pinv, transpose(U)));
}

#endif // !_SRENDERER_LINEAR_ALGEBRA_HEADER_