/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "math.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include "cuda_runtime.h"
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,  float *dL_dtau)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	dL_dtau[6 * idx + 0] += -dL_dmean.x;
	dL_dtau[6 * idx + 1] += -dL_dmean.y;
	dL_dtau[6 * idx + 2] += -dL_dmean.z;

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	SE3 T_CW(view_matrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t_ = T_CW.t();
	mat33 dpC_drho = mat33::identity();
	mat33 dpC_dtheta = -mat33::skew_symmetric(t);
	float dL_dt[6];
	for (int i = 0; i < 3; i++) {
		float3 c_rho = dpC_drho.cols[i];
		float3 c_theta = dpC_dtheta.cols[i];
		dL_dt[i] = dL_dtx * c_rho.x + dL_dty * c_rho.y + dL_dtz * c_rho.z;
		dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;

	float dL_dW00 = J[0][0] * dL_dT00;
	float dL_dW01 = J[0][0] * dL_dT01;
	float dL_dW02 = J[0][0] * dL_dT02;
	float dL_dW10 = J[1][1] * dL_dT10;
	float dL_dW11 = J[1][1] * dL_dT11;
	float dL_dW12 = J[1][1] * dL_dT12;
	float dL_dW20 = J[0][2] * dL_dT00 + J[1][2] * dL_dT10;
	float dL_dW21 = J[0][2] * dL_dT01 + J[1][2] * dL_dT11;
	float dL_dW22 = J[0][2] * dL_dT02 + J[1][2] * dL_dT12;

	float3 c1 = R.cols[0];
	float3 c2 = R.cols[1];
	float3 c3 = R.cols[2];

	float dL_dW_data[9];
	dL_dW_data[0] = dL_dW00;
	dL_dW_data[3] = dL_dW01;
	dL_dW_data[6] = dL_dW02;
	dL_dW_data[1] = dL_dW10;
	dL_dW_data[4] = dL_dW11;
	dL_dW_data[7] = dL_dW12;
	dL_dW_data[2] = dL_dW20;
	dL_dW_data[5] = dL_dW21;
	dL_dW_data[8] = dL_dW22;

	mat33 dL_dW(dL_dW_data);
	float3 dL_dWc1 = dL_dW.cols[0];
	float3 dL_dWc2 = dL_dW.cols[1];
	float3 dL_dWc3 = dL_dW.cols[2];

	mat33 n_W1_x = -mat33::skew_symmetric(c1);
	mat33 n_W2_x = -mat33::skew_symmetric(c2);
	mat33 n_W3_x = -mat33::skew_symmetric(c3);

	float3 dL_dtheta = {};
	dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0]) +
				dot(dL_dWc3, n_W3_x.cols[0]);
	dL_dtheta.y = dot(dL_dWc1, n_W1_x.cols[1]) + dot(dL_dWc2, n_W2_x.cols[1]) +
				dot(dL_dWc3, n_W3_x.cols[1]);
	dL_dtheta.z = dot(dL_dWc1, n_W1_x.cols[2]) + dot(dL_dWc2, n_W2_x.cols[2]) +
				dot(dL_dWc3, n_W3_x.cols[2]);

	dL_dtau[6 * idx + 3] += dL_dtheta.x;
	dL_dtau[6 * idx + 4] += dL_dtheta.y;
	dL_dtau[6 * idx + 5] += dL_dtheta.z;


}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA_no_tau(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	//float3* dL_dmeans,
	float* dL_dcov
	)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float* proj,
	const float *proj_raw,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float *dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float *dL_dtau)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	float c = proj_raw[10];
	float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	float dL_dpCz = dL_ddepth[idx];
	dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
	dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
	dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
		dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
	}



	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dtau);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template <uint32_t COLOR_CHANNELS, uint32_t SEMANTIC_CHANNELS>
__global__ void language_preprocessCUDA(int P,
										int D,
										int M,
										int language_M,
										const float3* means,
										const int* radii,
										const int* radii_lang,
										const float* shs,
										//const float* language_shs,
										const bool* clamped,
										const glm::vec3* scales,
										const glm::vec3* scales_lang,
										const glm::vec4* rotations,
										const glm::vec4* rotations_lang,
										const float scale_modifier,
										const float* viewmatrix,
										const float* proj,
										const float* proj_raw,
										const glm::vec3* campos,
										const float3* dL_dmean2D,
										//const float3* dL_dmean2D_lang,
										glm::vec3* dL_dmeans,
										float* dL_dcolor,
										float* dL_dlanguage,
										float* dL_ddepth,
										float* dL_dcov3D,
										float* dL_dcov3D_lang,
										float* dL_dsh,
										//float* dL_dlanguage_sh,
										glm::vec3* dL_dscale,
										glm::vec3* dL_dscale_lang,
										glm::vec4* dL_drot,
										glm::vec4* dL_drot_lang,
										float* dL_dtau) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0) || !(radii_lang[idx] > 0))
		return;
	
	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	// dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * (dL_dmean2D[idx].x + dL_dmean2D_lang[idx].x) + (proj[1] * m_w - proj[3] * mul2) * (dL_dmean2D[idx].y + dL_dmean2D_lang[idx].y);
	// dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * (dL_dmean2D[idx].x + dL_dmean2D_lang[idx].x) + (proj[5] * m_w - proj[7] * mul2) * (dL_dmean2D[idx].y + dL_dmean2D_lang[idx].y);
	// dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * (dL_dmean2D[idx].x + dL_dmean2D_lang[idx].x) + (proj[9] * m_w - proj[11] * mul2) * (dL_dmean2D[idx].y + dL_dmean2D_lang[idx].y);

	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	float alpha = 1.0f * m_w;
	float beta = -m_hom.x * m_w * m_w;
	float gamma = -m_hom.y * m_w * m_w;

	float a = proj_raw[0];
	float b = proj_raw[5];
	// float c = proj_raw[10];
	// float d = proj_raw[14];
	float e = proj_raw[11];

	SE3 T_CW(viewmatrix);
	mat33 R = T_CW.R().data();
	mat33 RT = R.transpose();
	float3 t = T_CW.t();
	float3 p_C = T_CW * m;
	mat33 dp_C_d_rho = mat33::identity();
	mat33 dp_C_d_theta = -mat33::skew_symmetric(p_C);

	float3 d_proj_dp_C1 = make_float3(alpha * a, 0.f, beta * e);
	float3 d_proj_dp_C2 = make_float3(0.f, alpha * b, gamma * e);

	float3 d_proj_dp_C1_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C1; // x.T A = A.T x
	float3 d_proj_dp_C2_d_rho = dp_C_d_rho.transpose() * d_proj_dp_C2;
	float3 d_proj_dp_C1_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C1;
	float3 d_proj_dp_C2_d_theta = dp_C_d_theta.transpose() * d_proj_dp_C2;

	float2 dmean2D_dtau[6];
	dmean2D_dtau[0].x = d_proj_dp_C1_d_rho.x;
	dmean2D_dtau[1].x = d_proj_dp_C1_d_rho.y;
	dmean2D_dtau[2].x = d_proj_dp_C1_d_rho.z;
	dmean2D_dtau[3].x = d_proj_dp_C1_d_theta.x;
	dmean2D_dtau[4].x = d_proj_dp_C1_d_theta.y;
	dmean2D_dtau[5].x = d_proj_dp_C1_d_theta.z;

	dmean2D_dtau[0].y = d_proj_dp_C2_d_rho.x;
	dmean2D_dtau[1].y = d_proj_dp_C2_d_rho.y;
	dmean2D_dtau[2].y = d_proj_dp_C2_d_rho.z;
	dmean2D_dtau[3].y = d_proj_dp_C2_d_theta.x;
	dmean2D_dtau[4].y = d_proj_dp_C2_d_theta.y;
	dmean2D_dtau[5].y = d_proj_dp_C2_d_theta.z;

	float dL_dt[6];
	for (int i = 0; i < 6; i++) {
		dL_dt[i] = dL_dmean2D[idx].x * dmean2D_dtau[i].x + 
					dL_dmean2D[idx].y * dmean2D_dtau[i].y;
	}
	for (int i = 0; i < 6; i++) {
		dL_dtau[6 * idx + i] += dL_dt[i];
	}

	// Compute gradient update due to computing depths
	// p_orig = m
	// p_view = transformPoint4x3(p_orig, viewmatrix);
	// depth = p_view.z;
	float dL_dpCz = dL_ddepth[idx];
	dL_dmeans[idx].x += dL_dpCz * viewmatrix[2];
	dL_dmeans[idx].y += dL_dpCz * viewmatrix[6];
	dL_dmeans[idx].z += dL_dpCz * viewmatrix[10];

	for (int i = 0; i < 3; i++) {
		float3 c_rho = dp_C_d_rho.cols[i];
		float3 c_theta = dp_C_d_theta.cols[i];
		dL_dtau[6 * idx + i] += dL_dpCz * c_rho.z;
		dL_dtau[6 * idx + i + 3] += dL_dpCz * c_theta.z;
	}

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx,
						   D,
						   M,
						   (glm::vec3*)means,
						   *campos,
						   shs,
						   clamped,
						   (glm::vec3*)dL_dcolor,
						   (glm::vec3*)dL_dmeans,
						   (glm::vec3*)dL_dsh,
						   dL_dtau);
		
	// Compute gradient updates due to computing language from SHs
	if (scales){
		computeCov3D(idx,
					 scales[idx],
					 scale_modifier,
					 rotations[idx],
					 dL_dcov3D,
					 dL_dscale,
					 dL_drot);
	}
	if (scales_lang){
		computeCov3D(idx,
					 scales_lang[idx],
					 scale_modifier,
					 rotations_lang[idx],
					 dL_dcov3D_lang,
					 dL_dscale_lang,
					 dL_drot_lang);
	}
}

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_depth,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
    
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
	__shared__ float dL_ddepths_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float dL_dpixel_depth = 0;
	if (inside) {
		#pragma unroll
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		dL_dpixel_depth = dL_dpixels_depth[pix_id];
	}

	float last_alpha = 0.f;
	float last_color[C] = { 0.f };
	float last_depth = 0.f;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			#pragma unroll
			for (int i = 0; i < C; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
				
			}
			collected_depths[tid] = depths[coll_id];
		}
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; // EQ.4 in Gauaasian Splatting paper
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;

			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}


			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float local_dL_dcolors[3];
			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}
			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			const float depth = collected_depths[j];
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++) {
				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			}
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

			render_cuda_reduce_sum(block, 
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared, 
				dL_ddepths_shared
			);	
			
			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float dL_ddepths_acc = dL_ddepths_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_acc.z);
				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);
			}
		}
	}
}

template <uint32_t C, uint32_t F>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
language_render_cuda(
	const uint2* __restrict__ ranges,
	const uint2* __restrict__ ranges_lang,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ point_list_lang,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ conic_opacity_lang,
	const float* __restrict__ colors,
	const float* __restrict__ language_feature,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const float* __restrict__ final_Ts_lang,
	const uint32_t* __restrict__ n_contrib,
	const uint32_t* __restrict__ n_contrib_lang,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixels_language,
	const float* __restrict__ dL_dpixels_depth,
	float3* __restrict__ dL_dmean2D,
	//float3* __restrict__ dL_dmean2D_lang,
	float4* __restrict__ dL_dconic2D,
	float4* __restrict__ dL_dconic2D_lang,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dopacity_lang,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dlanguage_feature,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
    
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const uint2 range_lang = ranges_lang[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	const int rounds_lang = ((range_lang.y - range_lang.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	bool done_lang = !inside;
	int toDo = range.y - range.x;
	int toDo_lang = range_lang.y - range_lang.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ int collected_id_lang[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float2 collected_xy_lang[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity_lang[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_language[F * BLOCK_SIZE];

	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
	//__shared__ float2 dL_dmean2D_shared_lang[BLOCK_SIZE];
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
	__shared__ float dL_ddepths_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared_lang[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared_lang[BLOCK_SIZE];
	__shared__ float collected_dL_dlanguage_feature[F * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	const float T_final_lang = inside ? final_Ts_lang[pix_id] : 0;
	float T = T_final;
	float T_lang = T_final_lang;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	uint32_t contributor_lang = toDo_lang;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int last_contributor_lang = inside ? n_contrib_lang[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	float accum_rec_depth = 0;
	float dL_dpixel_depth = 0;
	if (inside) {
		#pragma unroll
		for (int i = 0; i < C; i++) {
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
		dL_dpixel_depth = dL_dpixels_depth[pix_id];
	}

	float last_alpha = 0.f;
	float last_alpha_lang = 0.f;
	float last_color[C] = { 0.f };
	float last_depth = 0.f;

	float last_language_feature[F] = {0};
	float dL_dpixel_F[F] = {0};
	float accum_rec_F[F] = {0};

	if (true) {
		if (inside) {
			#pragma unroll
			for (int i = 0; i < F; i++) {
				dL_dpixel_F[i] = dL_dpixels_language[i * H * W + pix_id];
			}
		}
	}

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	__shared__ int skip_counter;
	__shared__ int skip_counter_lang;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			#pragma unroll
			for (int i = 0; i < C; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = colors[coll_id * C + i];
				
			}
			collected_depths[tid] = depths[coll_id];
		}

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			block.sync();
			if (tid == 0) {
				skip_counter = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip = done;
			contributor = done ? contributor : contributor - 1;
			skip |= contributor >= last_contributor;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;

			if (skip) {
				atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE) {
				continue;
			}

			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float local_dL_dcolors[3];
			#pragma unroll
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}
			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			const float depth = collected_depths[j];
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < C; i++) {
				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			}
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			// No need to add lang version since no lang bg

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

			render_cuda_reduce_sum(block, 
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared, 
				dL_ddepths_shared
			);
			
			
			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float dL_ddepths_acc = dL_ddepths_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * C + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * C + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * C + 2], dL_dcolors_acc.z);
				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);
			}
		}
	}
	for (int i = 0; i < rounds_lang; i++, toDo_lang -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range_lang.x + progress < range_lang.y)
		{
			if (true) {
				//const int coll_id_lang = point_list_lang[range.y - progress - 1];
				const int coll_id_lang = point_list_lang[range_lang.y - progress - 1];
				collected_id_lang[tid] = coll_id_lang;
				collected_xy_lang[tid] = points_xy_image[coll_id_lang];
				collected_conic_opacity_lang[tid] = conic_opacity_lang[coll_id_lang];
				#pragma unroll
				for (int i = 0; i < F; i++) {
					collected_language[i * BLOCK_SIZE + tid] = language_feature[coll_id_lang * F + i];
				}
			}
		}

		for (int j = 0; j < min(BLOCK_SIZE, toDo_lang); j++) {
			block.sync();
			if (tid == 0) {
				skip_counter_lang = 0;
			}
			block.sync();

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip_lang = done_lang;
			contributor_lang = done_lang ? contributor_lang : contributor_lang - 1;
			skip_lang |= contributor_lang >= last_contributor_lang;

			// Compute blending values, as before.
			const float2 xy = collected_xy_lang[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o_lang = collected_conic_opacity_lang[j];
			const float power_lang = -0.5f * (con_o_lang.x * d.x * d.x + con_o_lang.z * d.y * d.y) - con_o_lang.y * d.x * d.y;
			skip_lang |= power_lang > 0.0f;

			const float G_lang = exp(power_lang);
			const float alpha_lang = min(0.99f, con_o_lang.w * G_lang);
			skip_lang |= alpha_lang < 1.0f / 255.0f;

			if (skip_lang) {
				atomicAdd(&skip_counter_lang, 1);
			}
			block.sync();
			if (skip_counter_lang == BLOCK_SIZE) {
				continue;
			}

			T_lang = skip_lang ? T_lang : T_lang / (1.f - alpha_lang);
			const float dchannel_dcolor_lang = alpha_lang * T_lang;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha_lang = 0.0f;
			const int global_id = collected_id_lang[j];

			if (true) {
				for (int ch = 0; ch < F; ch++)
				{
					const float f = collected_language[ch * BLOCK_SIZE + j];
					// Update last color (to be used in the next iteration)
					accum_rec_F[ch] = last_alpha_lang * last_language_feature[ch] + (1.f - last_alpha_lang) * accum_rec_F[ch];
					last_language_feature[ch] = f;

					const float dL_dchannel_F = dL_dpixel_F[ch];
					dL_dalpha_lang += (f - accum_rec_F[ch]) * dL_dchannel_F;
					collected_dL_dlanguage_feature[ch * BLOCK_SIZE + tid] = skip_lang ? 0.0f : dchannel_dcolor_lang * dL_dchannel_F;
				}
			}

			dL_dalpha_lang *= T_lang;
			// Update last alpha (to be used in the next iteration)
			last_alpha_lang = skip_lang ? last_alpha_lang : alpha_lang;

			const float dL_dG_lang = con_o_lang.w * dL_dalpha_lang;
			const float gdx_lang = G_lang * d.x;
			const float gdy_lang = G_lang * d.y;

			dL_dconic2D_shared_lang[tid].x = skip_lang ? 0.f : -0.5f * gdx_lang * d.x * dL_dG_lang;
			dL_dconic2D_shared_lang[tid].y = skip_lang ? 0.f : -0.5f * gdx_lang * d.y * dL_dG_lang;
			dL_dconic2D_shared_lang[tid].w = skip_lang ? 0.f : -0.5f * gdy_lang * d.y * dL_dG_lang;
			dL_dopacity_shared_lang[tid] = skip_lang ? 0.f : G_lang * dL_dalpha_lang;

			render_cuda_reduce_sum(block, 
				//dL_dmean2D_shared_lang,
				dL_dconic2D_shared_lang,
				dL_dopacity_shared_lang
			);
			
			if (tid == 0) {
				float4 dL_dconic2D_acc_lang = dL_dconic2D_shared_lang[0];
				atomicAdd(&dL_dconic2D_lang[global_id].x, dL_dconic2D_acc_lang.x);
				atomicAdd(&dL_dconic2D_lang[global_id].y, dL_dconic2D_acc_lang.y);
				atomicAdd(&dL_dconic2D_lang[global_id].w, dL_dconic2D_acc_lang.w);

				if (true) {
					float dL_dopacity_acc_lang = dL_dopacity_shared_lang[0];
					atomicAdd(&dL_dopacity_lang[global_id], dL_dopacity_acc_lang);
					for (int ch = 0; ch < F; ch++) {
						atomicAdd(&(dL_dlanguage_feature[global_id * F + ch]), collected_dL_dlanguage_feature[ch * BLOCK_SIZE]);
					}
				}
			}
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dtau)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dtau);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_COLOR_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dtau);
}

void BACKWARD::language_preprocess(
	int P, int D, int M,
	int language_M,
	const float3* means,
	const int* radii,
	const int* radii_lang,
	const float* shs,
	//const float* language_shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec3* scales_lang,
	const glm::vec4* rotations,
	const glm::vec4* rotations_lang,
	const float scale_modifier,
	const float* cov3Ds,
	const float* cov3Ds_lang,
	const float* view_matrix,
	const float* projection_matrix,
	const float* projection_matirx_raw,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	//const float3* dL_dmean2D_lang,
	const float* dL_dconics,
	const float* dL_dconics_lang,
	glm::vec3* dL_dmeans,
	//glm::vec3* dL_dmeans_lang,
	float* dL_dcolor,
	float* dL_dlanguage,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dcov3D_lang,
	float* dL_dsh,
	//float* dL_dlanguage_sh,
	glm::vec3* dL_dscale,
	glm::vec3* dL_dscale_lang,
	glm::vec4* dL_drot,
	glm::vec4* dL_drot_lang,
	float* dL_dtau
	)
	//float* dL_dtau_lang
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		view_matrix,
		dL_dconics,
		(float3*)dL_dmeans,
		dL_dcov3D,
		dL_dtau);

	computeCov2DCUDA_no_tau << <(P + 255) / 256, 256 >> > (
		P,
		means,
		radii_lang,
		cov3Ds_lang,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		view_matrix,
		dL_dconics_lang,
		//(float3*)dL_dmeans_lang,
		dL_dcov3D_lang
		//dL_dtau_lang
		); // not used

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	language_preprocessCUDA<NUM_COLOR_CHANNELS, NUM_LANGUAGE_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M, language_M,
		(float3*)means,
		radii,
		radii_lang,
		shs,
		//language_shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec3*)scales_lang,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_lang,
		scale_modifier,
		view_matrix,
		projection_matrix,
		projection_matirx_raw,
		campos,
		(float3*)dL_dmean2D,
		//(float3*)dL_dmean2D_lang,
		(glm::vec3*)dL_dmeans, // only dL_means since this will be used to cal SH
		dL_dcolor,
		dL_dlanguage,
		dL_ddepth,
		dL_dcov3D,
		dL_dcov3D_lang,
		dL_dsh,
		//dL_dlanguage_sh,
		dL_dscale,
		dL_dscale_lang,
		dL_drot,
		dL_drot_lang,
		dL_dtau);

}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixels_depth,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{
	renderCUDA<NUM_COLOR_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixels_depth,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
	);
}

void BACKWARD::language_render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint2* ranges_lang,
	const uint32_t* point_list,
	const uint32_t* point_list_lang,
	int W, int H,
	const float* bg_color,
	//const float* bg_language,
	const float2* means2D,
	const float4* conic_opacity,
	const float4* conic_opacity_lang,
	const float* colors,
	const float* language,
	const float* depths,
	const float* final_Ts,
	const float* final_Ts_lang,
	const uint32_t* n_contrib,
	const uint32_t* n_contrib_lang,
	const float* dL_dpixels,
	const float* dL_dpixels_language,
	const float* dL_dpixels_depth,
	float3* dL_dmean2D,
	//float3* dL_dmean2D_lang,
	float4* dL_dconic2D,
	float4* dL_dconic2D_lang,
	float* dL_dopacity,
	float* dL_dopacity_lang,
	float* dL_dcolors,
	float* dL_dlanguage,
	float* dL_ddepths)
{
	language_render_cuda<NUM_COLOR_CHANNELS, NUM_LANGUAGE_CHANNELS> << <grid, block >> >(
		ranges,
		ranges_lang,
		point_list,
		point_list_lang,
		W, H,
		bg_color,
		//bg_language,
		means2D,
		conic_opacity,
		conic_opacity_lang,
		colors,
		language,
		depths,
		final_Ts,
		final_Ts_lang,
		n_contrib,
		n_contrib_lang,
		dL_dpixels,
		dL_dpixels_language,
		dL_dpixels_depth,
		dL_dmean2D,
		//dL_dmean2D_lang,
		dL_dconic2D,
		dL_dconic2D_lang,
		dL_dopacity,
		dL_dopacity_lang,
		dL_dcolors,
		dL_dlanguage,
		dL_ddepths);
}