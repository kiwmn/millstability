#include "millstability/column_kernel.cuh"
#include <c10/cuda/CUDAException.h>
#include "millstability/matrix_math.cuh"
#include "millstability/register_inverse.cuh"

namespace millstability::detail {
namespace {

struct ColumnGridConfig {
    static constexpr int kPrepareThreads = 128;
    static constexpr int kColumnThreads = 96;
    static constexpr int kMaxOrder = 84;
    static constexpr int kMaxSteps = 40;
    static constexpr int kPhiElements = 64;
    // 32 coefficients plus padding keep each step aligned for float4 loads.
    static constexpr int kStepStride = 36;
    static_assert(kColumnThreads >= kMaxOrder);
    static_assert(kColumnThreads >= kMaxSteps);
    static_assert(kColumnThreads % 32 == 0);
};

// One thread per speed prepares four integration matrices shared by all depths.
__global__ void prepare_phi_kernel(
    float* phi, const float* d_A0, const float* d_invA0,
    int stx, float o_st, float o_fi, int N, int m) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > stx) return;
    const float o = o_st + (float)x * (o_fi - o_st) / stx;
    const float tau = 60.0f / o / N;
    const float dt = tau / m;
    float local_phi[ColumnGridConfig::kPhiElements];
    float* s_Fi0 = local_phi;
    float* s_Fi1 = local_phi + 16;
    float* s_Fi2 = local_phi + 32;
    float* s_Fi3 = local_phi + 48;

    float A0_dt[16];
    for (int i = 0; i < 16; i++) {
        A0_dt[i] = d_A0[i] * dt;
    }

    matrix_expm_4x4(A0_dt, 1.0f, s_Fi0);

    float Fi0_minus_I[16];
    for (int i = 0; i < 16; i++) {
        Fi0_minus_I[i] = s_Fi0[i] - (i % 5 == 0 ? 1.0f : 0.0f);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s_Fi1[i*4+j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                s_Fi1[i*4+j] += d_invA0[i*4+k] * Fi0_minus_I[k*4+j];
            }
        }
    }

    float Fi0_dt_minus_Fi1[16];
    for (int i = 0; i < 16; i++) {
        Fi0_dt_minus_Fi1[i] = s_Fi0[i] * dt - s_Fi1[i];
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s_Fi2[i*4+j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                s_Fi2[i*4+j] += d_invA0[i*4+k] * Fi0_dt_minus_Fi1[k*4+j];
            }
        }
    }

    float Fi0_dt2_minus_2Fi2[16];
    for (int i = 0; i < 16; i++) {
        Fi0_dt2_minus_2Fi2[i] = s_Fi0[i] * dt * dt - 2.0f * s_Fi2[i];
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s_Fi3[i*4+j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                s_Fi3[i*4+j] += d_invA0[i*4+k] * Fi0_dt2_minus_2Fi2[k*4+j];
            }
        }
    }
    #pragma unroll
    for (int entry = 0; entry < ColumnGridConfig::kPhiElements; ++entry) {
        phi[x * 64 + entry] = local_phi[entry];
    }
}

// Pack each active row as four current-state coefficients and four delay coefficients.
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

    // A0k holds the next sample of cutting coefficients, multiplied by -depth.
    float A0k[16] = {0};
    if (i < m) {
        A0k[2*4 + 0] = -w * d_hxx_shared[i + 1];
        A0k[2*4 + 1] = -w * d_hxy_shared[i + 1];
        A0k[3*4 + 0] = -w * d_hyx_shared[i + 1];
        A0k[3*4 + 1] = -w * d_hyy_shared[i + 1];
    }

    // A1k is depth times the coefficient difference divided by dt.
    float A1k[16] = {0};
    if (i < m) {
        A1k[2*4 + 0] = w * (d_hxx_shared[i + 1] - d_hxx_shared[i]) / dt;
        A1k[2*4 + 1] = w * (d_hxy_shared[i + 1] - d_hxy_shared[i]) / dt;
        A1k[3*4 + 0] = w * (d_hyx_shared[i + 1] - d_hyx_shared[i]) / dt;
        A1k[3*4 + 1] = w * (d_hyy_shared[i + 1] - d_hyy_shared[i]) / dt;
    }

    // F01 = (Fi2*A0k + Fi3*A1k) / dt.
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

    // Fkp1 = (Fi1 - Fi2/dt)*A0k + (Fi2 - Fi3/dt)*A1k.
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

    // The step update uses inverse(I - Fkp1).
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

// One block owns a process point. The first m threads prepare step coefficients;
// then each column thread propagates four state components without further barriers.
template <int StaticSteps>
__global__ void fused_column_kernel(
    float* result, const float* phi,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int runtime_m) {
    static_assert(StaticSteps == 0 || StaticSteps == ColumnGridConfig::kMaxSteps);
    const int m = StaticSteps == 0 ? runtime_m : StaticSteps;
    int x, y, point;
    if constexpr (StaticSteps != 0) {

        x = blockIdx.y;
        y = blockIdx.x;
        point = x * (sty + 1) + y;
    } else {

        point = blockIdx.x;
        x = point / (sty + 1);
        y = point - x * (sty + 1);
    }
    const int col = threadIdx.x;

    __shared__ __align__(16) float steps[ColumnGridConfig::kMaxSteps][ColumnGridConfig::kStepStride];
    if (col < m) {
        prepare_step_coefficients(
            steps[col], phi, hxx, hxy, hyx, hyy,
            x, y, col, stx, sty, o_st, o_fi, w_st, w_fi, N, runtime_m);
    }
    __syncthreads();

    const int order = 2 * m + 4;
    if (col >= order) return;
    float current[4];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        current[row] = col == row ? 1.0f : 0.0f;
    }
    float* output = result + point * order * order;
    float* history_output = output + (order - 2) * order + col;
    const float* coefficients = steps[0];
    int last_basis = order - 2;

    // Keep the time recurrence as a loop to limit generated instruction size.
    #pragma unroll 1
    for (int i = 0; i < m; ++i) {
        const int previous_basis = i == m - 1 ? 0 : last_basis - 2;
        // Write each pair of displacement history rows directly to its final position.
        history_output[0] = current[0];
        history_output[order] = current[1];
        float next[4];
        #pragma unroll
        for (int row = 0; row < 4; ++row) {

            const float4 head = *reinterpret_cast<const float4*>(coefficients + row * 8);
            const float4 delay = *reinterpret_cast<const float4*>(coefficients + row * 8 + 4);
            float value = 0.0f;
            value += head.x * current[0];
            value += head.y * current[1];
            value += head.z * current[2];
            value += head.w * current[3];
            value += delay.x * (col == previous_basis ? 1.0f : 0.0f);
            value += delay.y * (col == previous_basis + 1 ? 1.0f : 0.0f);
            value += delay.z * (col == last_basis ? 1.0f : 0.0f);
            value += delay.w * (col == last_basis + 1 ? 1.0f : 0.0f);
            next[row] = value;
        }
        #pragma unroll
        for (int row = 0; row < 4; ++row) current[row] = next[row];
        history_output -= 2 * order;
        coefficients += ColumnGridConfig::kStepStride;
        last_basis -= 2;
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        output[row * order + col] = current[row];
    }
}

}

void launch_column_grid(
    float* result, float* phi_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, cudaStream_t stream) {
    constexpr int prepare_threads = ColumnGridConfig::kPrepareThreads;
    prepare_phi_kernel<<<(stx + 1 + prepare_threads - 1) / prepare_threads,
                         prepare_threads, 0, stream>>>(
        phi_workspace, A0, invA0, stx, o_st, o_fi, N, m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // The two-dimensional launch allows at most 65,535 speed blocks in grid.y.
    if (m == ColumnGridConfig::kMaxSteps && stx < 65535) {
        const dim3 point_grid(sty + 1, stx + 1);
        fused_column_kernel<ColumnGridConfig::kMaxSteps>
            <<<point_grid, ColumnGridConfig::kColumnThreads, 0, stream>>>(
                result, phi_workspace, hxx, hxy, hyx, hyy,
                stx, sty, o_st, o_fi, w_st, w_fi, N, m);
    } else {
        const int points = (stx + 1) * (sty + 1);
        fused_column_kernel<0>
            <<<points, ColumnGridConfig::kColumnThreads, 0, stream>>>(
                result, phi_workspace, hxx, hxy, hyx, hyy,
                stx, sty, o_st, o_fi, w_st, w_fi, N, m);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
