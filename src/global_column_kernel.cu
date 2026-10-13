#include "millstability/global_column_kernel.cuh"
#include "millstability/column_kernel.cuh"
#include "millstability/matrix_math.cuh"
#include "millstability/register_inverse.cuh"
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace millstability::detail {
namespace {

// Pack each update row as four current-state and four delay coefficients.
__device__ __forceinline__ void prepare_step_coefficients(
    float* packed_D, const float* phi,
    const float* d_hxx_shared, const float* d_hxy_shared,
    const float* d_hyx_shared, const float* d_hyy_shared,
    int x, int y, int i, int stx, int sty,
    float o_st, float o_fi, float w_st, float w_fi, int N, int m) {
    const float o = o_st + (float)x * (o_fi - o_st) / stx;
    const float tau = 60.0f / o / N;
    const float dt = tau / m;
    const float w = w_st + (float)y * (w_fi - w_st) / sty;
    const float* s_Fi0 = phi + x * 64;
    const float* s_Fi1 = s_Fi0 + 16;
    const float* s_Fi2 = s_Fi0 + 32;
    const float* s_Fi3 = s_Fi0 + 48;
    float D[4][8];

    float A0k[16] = {0};
    if (i < m) {
        A0k[2*4 + 0] = -w * d_hxx_shared[i + 1];
        A0k[2*4 + 1] = -w * d_hxy_shared[i + 1];
        A0k[3*4 + 0] = -w * d_hyx_shared[i + 1];
        A0k[3*4 + 1] = -w * d_hyy_shared[i + 1];
    }

    float A1k[16] = {0};
    if (i < m) {
        A1k[2*4 + 0] = w * (d_hxx_shared[i + 1] - d_hxx_shared[i]) / dt;
        A1k[2*4 + 1] = w * (d_hxy_shared[i + 1] - d_hxy_shared[i]) / dt;
        A1k[3*4 + 0] = w * (d_hyx_shared[i + 1] - d_hyx_shared[i]) / dt;
        A1k[3*4 + 1] = w * (d_hyy_shared[i + 1] - d_hyy_shared[i]) / dt;
    }

    // F01=(Phi2*A0k+Phi3*A1k)/dt (Ding et al., 2010, Eq. 11).
    float F01[16] = {0};

    float temp1[16] = {0};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            for (int k = 0; k < 4; k++) {
                temp1[row*4 + col] += s_Fi2[row*4 + k] * A0k[k*4 + col];
            }
            temp1[row*4 + col] /= dt;
        }
    }

    float temp2[16] = {0};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            for (int k = 0; k < 4; k++) {
                temp2[row*4 + col] += s_Fi3[row*4 + k] * A1k[k*4 + col];
            }
            temp2[row*4 + col] /= dt;
        }
    }

    for (int idx = 0; idx < 16; idx++) {
        F01[idx] = temp1[idx] + temp2[idx];
    }

    // Fkp1=(Phi1-Phi2/dt)*A0k+(Phi2-Phi3/dt)*A1k (Eq. 12).
    float Fkp1[16] = {0};

    float temp3[16] = {0};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            for (int k = 0; k < 4; k++) {
                float Fi1_minus_Fi2_dt = s_Fi1[row*4 + k] - s_Fi2[row*4 + k] / dt;
                temp3[row*4 + col] += Fi1_minus_Fi2_dt * A0k[k*4 + col];
            }
        }
    }

    float temp4[16] = {0};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            for (int k = 0; k < 4; k++) {
                float Fi2_minus_Fi3_dt = s_Fi2[row*4 + k] - s_Fi3[row*4 + k] / dt;
                temp4[row*4 + col] += Fi2_minus_Fi3_dt * A1k[k*4 + col];
            }
        }
    }

    for (int idx = 0; idx < 16; idx++) {
        Fkp1[idx] = temp3[idx] + temp4[idx];
    }

    float I_minus_Fkp1[16];

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int idx = row*4 + col;
            if (row == col) {
                I_minus_Fkp1[idx] = 1.0f - Fkp1[idx];
            } else {
                I_minus_Fkp1[idx] = -Fkp1[idx];
            }
        }
    }

    float invOfImFkp1[16];
    inverse_register_4x4(I_minus_Fkp1, invOfImFkp1);

    float Fi0_plus_F01[16];
    for(int idx = 0; idx < 16; idx++) {
        Fi0_plus_F01[idx] = s_Fi0[idx] + F01[idx];
    }

    for(int row = 0; row < 4; row++) {
        for(int col = 0; col < 4; col++) {
            D[row][col] = 0.0f;
            for(int k = 0; k < 4; k++) {
                D[row][col] += invOfImFkp1[row*4 + k] * Fi0_plus_F01[k*4 + col];
            }
        }
    }

    float Fkp1_12[8];
    for(int row = 0; row < 4; row++) {
        Fkp1_12[row*2 + 0] = Fkp1[row*4 + 0];
        Fkp1_12[row*2 + 1] = Fkp1[row*4 + 1];
    }

    for(int row = 0; row < 4; row++) {
        for(int col = 0; col < 2; col++) {
            D[row][4 + col] = 0.0f;
            for(int k = 0; k < 4; k++) {
                D[row][4 + col] -= invOfImFkp1[row*4 + k] * Fkp1_12[k*2 + col];
            }
        }
    }

    float F01_12[8];
    for(int row = 0; row < 4; row++) {
        F01_12[row*2 + 0] = F01[row*4 + 0];
        F01_12[row*2 + 1] = F01[row*4 + 1];
    }

    for(int row = 0; row < 4; row++) {
        for(int col = 0; col < 2; col++) {
            D[row][6 + col] = 0.0f;
            for(int k = 0; k < 4; k++) {
                D[row][6 + col] -= invOfImFkp1[row*4 + k] * F01_12[k*2 + col];
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 8; ++col) {
            packed_D[row * 8 + col] = D[row][col];
        }
    }
}

// One block per process point writes its contiguous [m,36] step workspace.
__global__ void global_column_prepare_kernel(
    float* steps, const float* phi,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m) {
    const int point = blockIdx.x;
    const int x = point / (sty + 1);
    const int y = point - x * (sty + 1);
    float* point_steps = steps + static_cast<size_t>(point) * m * 36;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        prepare_step_coefficients(point_steps + static_cast<size_t>(i) * 36,
            phi, hxx, hxy, hyx, hyy,
            x, y, i, stx, sty, o_st, o_fi, w_st, w_fi, N, m);
    }
}

// One block per point; each thread propagates strided output columns.
__global__ void global_propagate_kernel(
    float* result, const float* steps, int points, int m,
    const int* active_map, int reduced_order) {
    const int point = blockIdx.x;
    if (point >= points) return;
    const int order = 2 * m + 4;
    const bool reduced = active_map != nullptr;
    const int output_order = reduced ? reduced_order : order;
    float* output = result + static_cast<size_t>(point) * output_order * output_order;
    const float* point_steps = steps + static_cast<size_t>(point) * m * 36;
    for (int output_col = threadIdx.x; output_col < output_order;
         output_col += blockDim.x) {
        const int col = reduced ? active_map[output_col + 1] : output_col;
        float current[4];
        #pragma unroll
        for (int row = 0; row < 4; ++row)
            current[row] = col == row ? 1.0f : 0.0f;
        float* history_output = output + (output_order - 2) * output_order + output_col;
        int history_row = output_order - 2;
        int last_basis = order - 2;
        for (int i = 0; i < m; ++i) {
            const int previous_basis = i == m - 1 ? 0 : last_basis - 2;
            if (!reduced) {
                history_output[0] = current[0];
                history_output[output_order] = current[1];
                history_output -= 2 * output_order;
            } else if (history_row >= 4 && active_map[history_row + 1] == last_basis) {
                history_output[0] = current[0];
                history_output[output_order] = current[1];
                history_output -= 2 * output_order;
                history_row -= 2;
            }
            const float* coefficients = point_steps + static_cast<size_t>(i) * 36;
            float next[4];
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                const float4 head = *reinterpret_cast<const float4*>(coefficients + row * 8);
                const float4 delay = *reinterpret_cast<const float4*>(coefficients + row * 8 + 4);
                float value = head.x * current[0] + head.y * current[1]
                            + head.z * current[2] + head.w * current[3];
                value += delay.x * (col == previous_basis ? 1.0f : 0.0f);
                value += delay.y * (col == previous_basis + 1 ? 1.0f : 0.0f);
                value += delay.z * (col == last_basis ? 1.0f : 0.0f);
                value += delay.w * (col == last_basis + 1 ? 1.0f : 0.0f);
                next[row] = value;
            }
            #pragma unroll
            for (int row = 0; row < 4; ++row) current[row] = next[row];
            last_basis -= 2;
        }
        #pragma unroll
        for (int row = 0; row < 4; ++row)
            output[row * output_order + output_col] = current[row];
    }
}

// Retain current-state columns and delay pairs with nonzero endpoint coefficients.
// map[0] is the count; following entries are sorted original column indices.
__global__ void global_active_columns_kernel(
    int* map, int* active, const float* hxx, const float* hxy,
    const float* hyx, const float* hyy, int m) {
    const int order = 2 * m + 4;
    for (int col = threadIdx.x; col < order; col += blockDim.x)
        active[col] = col < 4 ? 1 : 0;
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const bool used = hxx[i] != 0.0f || hxy[i] != 0.0f ||
                          hyx[i] != 0.0f || hyy[i] != 0.0f ||
                          hxx[i + 1] != 0.0f || hxy[i + 1] != 0.0f ||
                          hyx[i + 1] != 0.0f || hyy[i + 1] != 0.0f;
        if (used) {
            const int last = order - 2 - 2 * i;
            const int previous = i == m - 1 ? 0 : last - 2;
            atomicExch(active + previous, 1);
            atomicExch(active + previous + 1, 1);
            atomicExch(active + last, 1);
            atomicExch(active + last + 1, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int count = 0;
        for (int col = 0; col < order; ++col)
            if (active[col]) map[++count] = col;
        map[0] = count;
    }
}

// A parameter group stores four row-major 4x4 integration matrices in 64 floats.
__device__ __forceinline__ void global_parameter_phi(
    float* phi, const float* p, int N, int m, float o) {
    const float w0x = p[2], w0y = p[3], zetax = p[4], zetay = p[5];
    const float m_tx = p[6], m_ty = p[7];
    const float dt = 60.0f / o / N / m;
    float A0[16] = {0.0f};
    A0[0] = -zetax * w0x; A0[2] = 1.0f / m_tx;
    A0[8] = (zetax * zetax - 1.0f) * m_tx * w0x * w0x;
    A0[5] = -zetay * w0y; A0[7] = 1.0f / m_ty;
    A0[10] = -zetax * w0x;
    A0[13] = (zetay * zetay - 1.0f) * m_ty * w0y * w0y;
    A0[15] = -zetay * w0y;
    float invA0[16]; inverse_register_4x4(A0, invA0);
    float A0dt[16];
    for (int i = 0; i < 16; ++i) A0dt[i] = A0[i] * dt;
    float fi0[16]; matrix_expm_4x4(A0dt, 1.0f, fi0);
    float fi1[16], fi2[16], fi3[16];
    float tmp[16];
    for (int i = 0; i < 16; ++i) tmp[i] = fi0[i] - (i % 5 == 0 ? 1.0f : 0.0f);
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float v = 0.0f; for (int k = 0; k < 4; ++k) v += invA0[r*4+k] * tmp[k*4+c]; fi1[r*4+c] = v;
    }
    for (int i = 0; i < 16; ++i) tmp[i] = fi0[i] * dt - fi1[i];
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float v = 0.0f; for (int k = 0; k < 4; ++k) v += invA0[r*4+k] * tmp[k*4+c]; fi2[r*4+c] = v;
    }
    for (int i = 0; i < 16; ++i) tmp[i] = fi0[i] * dt * dt - 2.0f * fi2[i];
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
        float v = 0.0f; for (int k = 0; k < 4; ++k) v += invA0[r*4+k] * tmp[k*4+c]; fi3[r*4+c] = v;
    }
    for (int i = 0; i < 16; ++i) { phi[i] = fi0[i]; phi[16+i] = fi1[i]; phi[32+i] = fi2[i]; phi[48+i] = fi3[i]; }
}

// One block per parameter group prepares integration matrices, cutting samples and steps.
__global__ void global_parameter_prepare_kernel(
    float* workspace, const float* parameters, int N, float aD,
    int up_or_down, int m, float w, float o, int n_params) {
    const int point = blockIdx.x;
    if (point >= n_params) return;
    float* phi = workspace + static_cast<size_t>(point) * 64;
    const size_t h_stride = static_cast<size_t>(4) * (m + 1);
    float* h = workspace + static_cast<size_t>(n_params) * 64 + point * h_stride;
    float* steps = workspace + static_cast<size_t>(n_params) * (64 + h_stride);
    if (threadIdx.x == 0)
        global_parameter_phi(phi, parameters + static_cast<size_t>(point) * 8, N, m, o);
    const float fist = up_or_down == 1 ? 0.0f : acosf(2.0f * aD - 1.0f);
    const float fiex = up_or_down == 1 ? acosf(1.0f - 2.0f * aD) : M_PI;
    const float dtr = 2.0f * M_PI / N / m;
    const float Kt = parameters[point * 8], Kn = parameters[point * 8 + 1];
    float* hxx = h, *hxy = h + m + 1, *hyx = h + 2 * (m + 1), *hyy = h + 3 * (m + 1);
    for (int i = threadIdx.x; i <= m; i += blockDim.x) {
        float xx=0,xy=0,yx=0,yy=0;
        for (int j=1; j<=N; ++j) {
            const float fi = (i + 1) * dtr + (j - 1) * 2.0f * M_PI / N;
            if (fist <= fi && fi <= fiex) {
                const float c=cosf(fi), s=sinf(fi);
                xx += (Kt*c + Kn*s)*s; xy += (Kt*c + Kn*s)*c;
                yx += (-Kt*s + Kn*c)*s; yy += (-Kt*s + Kn*c)*c;
            }
        }
        hxx[i]=xx; hxy[i]=xy; hyx[i]=yx; hyy[i]=yy;
    }
    __syncthreads();
    float* point_steps = steps + static_cast<size_t>(point) * m * 36;
    for (int i=threadIdx.x; i<m; i+=blockDim.x)
        prepare_step_coefficients(point_steps + static_cast<size_t>(i)*36,
            phi, hxx, hxy, hyx, hyy, 0, 0, i, 1, 1, o, o, w, w, N, m);
}

}

void launch_global_column_grid(
    float* result, float* phi_workspace, float* step_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, const int* active_map, int reduced_order,
    cudaStream_t stream) {
    const int points = (stx + 1) * (sty + 1);
    launch_phi_workspace(phi_workspace, A0, invA0, stx, o_st, o_fi, N, m, stream);
    global_column_prepare_kernel<<<points, 256, 0, stream>>>(
        step_workspace, phi_workspace, hxx, hxy, hyx, hyy,
        stx, sty, o_st, o_fi, w_st, w_fi, N, m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    global_propagate_kernel<<<points, 256, 0, stream>>>(
        result, step_workspace, points, m, active_map, reduced_order);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_global_active_columns(
    int* map, int* active, const float* hxx, const float* hxy,
    const float* hyx, const float* hyy, int m, cudaStream_t stream) {
    global_active_columns_kernel<<<1, 256, 0, stream>>>(map, active, hxx, hxy, hyx, hyy, m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_global_parameter_matrices(
    float* result, float* workspace, const float* parameters,
    int N, float aD, int up_or_down, int m, float w, float o,
    int n_params, cudaStream_t stream) {
    global_parameter_prepare_kernel<<<n_params, 256, 0, stream>>>(
        workspace, parameters, N, aD, up_or_down, m, w, o, n_params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t h_stride = int64_t(4) * (m + 1);
    const int64_t steps_offset = int64_t(n_params) * (64 + h_stride);
    const float* steps = workspace + steps_offset;
    global_propagate_kernel<<<n_params, 256, 0, stream>>>(
        result, steps, n_params, m, nullptr, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
