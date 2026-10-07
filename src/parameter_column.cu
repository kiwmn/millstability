#include "millstability/parameter_column.cuh"
#include "millstability/matrix_math.cuh"
#include "millstability/register_inverse.cuh"
#include <c10/cuda/CUDAException.h>

namespace millstability::detail {
namespace {

struct ParameterColumnConfig {
    static constexpr int kColumnThreads = 96;
    static constexpr int kMaxSteps = 40;
    static constexpr int kMaxOrder = 84;
    // 32 coefficients plus padding keep each step aligned for float4 loads.
    static constexpr int kStepStride = 36;
    static_assert(kColumnThreads >= kMaxOrder);
    static_assert(kColumnThreads >= kMaxSteps + 1);
    static_assert(kColumnThreads % 32 == 0);
};

// Thread 0 prepares the four integration matrices for one parameter row.
__device__ __forceinline__ void prepare_parameter_phi(
    float* phi, const float* parameters, int N, int m, float o) {

    constexpr int param_offset = 0;
    float w0x = parameters[param_offset + 2];
    float w0y = parameters[param_offset + 3];
    float zetax = parameters[param_offset + 4];
    float zetay = parameters[param_offset + 5];
    float m_tx = parameters[param_offset + 6];
    float m_ty = parameters[param_offset + 7];

    float tau = 60.0f / o / N;
    float dt = tau / m;

    float A0[16] = {0};
    A0[0*4 + 0] = -zetax * w0x;
    A0[0*4 + 2] = 1.0f/m_tx;
    A0[2*4 + 0] = (zetax*zetax - 1)*m_tx * w0x * w0x;
    A0[1*4 + 1] = -zetay * w0y;
    A0[2*4 + 2] = -zetax * w0x;
    A0[1*4 + 3] = 1.0f/m_ty;
    A0[3*4 + 1] = (zetay*zetay - 1)*m_ty * w0y * w0y;
    A0[3*4 + 3] = -zetay * w0y;

    float invA0[16] = {0};

    inverse_register_4x4(A0, invA0);

    float A0_dt[16];
    for (int i = 0; i < 16; i++) {
        A0_dt[i] = A0[i] * dt;
    }

    float s_Fi0[16];
    matrix_expm_4x4(A0_dt, 1.0f, s_Fi0);

    float s_Fi1[16] = {0}, s_Fi2[16] = {0}, s_Fi3[16] = {0};

    float Fi0_minus_I[16];
    for (int i = 0; i < 16; i++) {
        Fi0_minus_I[i] = s_Fi0[i] - (i % 5 == 0 ? 1.0f : 0.0f);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s_Fi1[i*4+j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                s_Fi1[i*4+j] += invA0[i*4+k] * Fi0_minus_I[k*4+j];
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
                s_Fi2[i*4+j] += invA0[i*4+k] * Fi0_dt_minus_Fi1[k*4+j];
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
                s_Fi3[i*4+j] += invA0[i*4+k] * Fi0_dt2_minus_2Fi2[k*4+j];
            }
        }
    }

    float* phi_output = phi;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        phi_output[i] = s_Fi0[i];
        phi_output[16 + i] = s_Fi1[i];
        phi_output[32 + i] = s_Fi2[i];
        phi_output[48 + i] = s_Fi3[i];
    }
}

// Tooth angles and engagement are shared by all parameter rows.
__global__ void prepare_parameter_geometry_kernel(
    float4* geometry, int N, float aD, int up_or_down, int runtime_m) {
    const int i = threadIdx.x;
    if (i > runtime_m) return;
    float fist, fiex;
    if (up_or_down == 1) {
        fist = 0.0f;
        fiex = acosf(1.0f - 2.0f * aD);
    } else {
        fist = acosf(2.0f * aD - 1.0f);
        fiex = M_PI;
    }
    float dtr = 2.0f * M_PI / N / runtime_m;
    for (int j = 1; j <= N; j++) {
        float fi = (i + 1) * dtr + (j - 1) * 2.0f * M_PI / N;
        int g = (fist <= fi && fi <= fiex) ? 1 : 0;
        float cos_fi = 0.0f;
        float sin_fi = 0.0f;
        if (g) {
            cos_fi = cosf(fi);
            sin_fi = sinf(fi);
        }
        geometry[i * N + (j - 1)] = make_float4(cos_fi, sin_fi, float(g), 0.0f);
    }
}

// Pack each active row as four current-state coefficients and four delay coefficients.
__device__ __forceinline__ void prepare_parameter_step_coefficients(
    float* packed_D, const float* phi,
    const float* d_hxx_shared, const float* d_hxy_shared,
    const float* d_hyx_shared, const float* d_hyy_shared,
    int i, float w, float o, int N, int m) {
    const float tau = 60.0f / o / N;
    const float dt = tau / m;
    const float* s_Fi0 = phi;
    const float* s_Fi1 = phi + 16;
    const float* s_Fi2 = phi + 32;
    const float* s_Fi3 = phi + 48;
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

// One block owns a parameter row; one thread propagates each matrix column.
// The launch bounds constrain register use for the 96-thread block.
template <int StaticSteps, bool PreparedGeometry>

__global__ __launch_bounds__(ParameterColumnConfig::kColumnThreads, 10)
void parameter_fused_column_kernel(
    float* result, const float* parameters, const float4* geometry,
    float w, float o, int N, float aD, int up_or_down, int runtime_m) {
    static_assert(StaticSteps == 0 || StaticSteps == ParameterColumnConfig::kMaxSteps);
    const int m = StaticSteps == 0 ? runtime_m : StaticSteps;
    const int point = blockIdx.x;
    const int col = threadIdx.x;
    __shared__ float phi[64];
    __shared__ float cutting[4][ParameterColumnConfig::kMaxSteps + 1];
    __shared__ __align__(16) float steps[ParameterColumnConfig::kMaxSteps][ParameterColumnConfig::kStepStride];

    if (col == 0) {
        prepare_parameter_phi(phi, parameters + point * 8, N, runtime_m, o);
    }

    // The first m + 1 threads sum tooth contributions independently for each sample.
    if (col <= m) {
        const int i = col;
        const float Kt = parameters[point * 8];
        const float Kn = parameters[point * 8 + 1];
        float hxx_sum = 0.0f, hxy_sum = 0.0f, hyx_sum = 0.0f, hyy_sum = 0.0f;
        if constexpr (PreparedGeometry) {
            for (int j = 1; j <= N; j++) {
                const float4 tooth = geometry[i * N + (j - 1)];
                if (tooth.z != 0.0f) {
                    const float cos_fi = tooth.x;
                    const float sin_fi = tooth.y;
                    hxx_sum += (Kt * cos_fi + Kn * sin_fi) * sin_fi;
                    hxy_sum += (Kt * cos_fi + Kn * sin_fi) * cos_fi;
                    hyx_sum += (-Kt * sin_fi + Kn * cos_fi) * sin_fi;
                    hyy_sum += (-Kt * sin_fi + Kn * cos_fi) * cos_fi;
                }
            }
        } else {

            float fist, fiex;
            if (up_or_down == 1) {
                fist = 0.0f;
                fiex = acosf(1.0f - 2.0f * aD);
            } else {
                fist = acosf(2.0f * aD - 1.0f);
                fiex = M_PI;
            }
            float dtr = 2.0f * M_PI / N / runtime_m;

            for (int j = 1; j <= N; j++) {
                float fi = (i + 1) * dtr + (j - 1) * 2.0f * M_PI / N;
                int g = (fist <= fi && fi <= fiex) ? 1 : 0;

                if (g) {
                    float cos_fi = cosf(fi);
                    float sin_fi = sinf(fi);

                    hxx_sum += (Kt * cos_fi + Kn * sin_fi) * sin_fi;
                    hxy_sum += (Kt * cos_fi + Kn * sin_fi) * cos_fi;
                    hyx_sum += (-Kt * sin_fi + Kn * cos_fi) * sin_fi;
                    hyy_sum += (-Kt * sin_fi + Kn * cos_fi) * cos_fi;
                }
            }

        }

        cutting[0][i] = hxx_sum;
        cutting[1][i] = hxy_sum;
        cutting[2][i] = hyx_sum;
        cutting[3][i] = hyy_sum;
    }
    __syncthreads();
    if (col < m) {
        prepare_parameter_step_coefficients(
            steps[col], phi,
            cutting[0], cutting[1], cutting[2], cutting[3],
            col, w, o, N, runtime_m);
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
        coefficients += ParameterColumnConfig::kStepStride;
        last_basis -= 2;
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        output[row * order + col] = current[row];
    }
}

}

void launch_parameter_columns(
    float* result, float* geometry_workspace,
    const float* parameters, int N, float aD, int up_or_down,
    int m, float w, float o, int n_params, cudaStream_t stream) {
    if (N <= kParameterGeometryMaxTeeth) {
        prepare_parameter_geometry_kernel<<<1, 64, 0, stream>>>(
            reinterpret_cast<float4*>(geometry_workspace), N, aD, up_or_down, m);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (m == ParameterColumnConfig::kMaxSteps) {
            parameter_fused_column_kernel<ParameterColumnConfig::kMaxSteps, true>
                <<<n_params, ParameterColumnConfig::kColumnThreads, 0, stream>>>(
                    result, parameters,
                    reinterpret_cast<const float4*>(geometry_workspace),
                    w, o, N, aD, up_or_down, m);
        } else {
            parameter_fused_column_kernel<0, true>
                <<<n_params, ParameterColumnConfig::kColumnThreads, 0, stream>>>(
                    result, parameters,
                    reinterpret_cast<const float4*>(geometry_workspace),
                    w, o, N, aD, up_or_down, m);
        }
    } else {
        if (m == ParameterColumnConfig::kMaxSteps) {
            parameter_fused_column_kernel<ParameterColumnConfig::kMaxSteps, false>
                <<<n_params, ParameterColumnConfig::kColumnThreads, 0, stream>>>(
                    result, parameters, nullptr,
                    w, o, N, aD, up_or_down, m);
        } else {
            parameter_fused_column_kernel<0, false>
                <<<n_params, ParameterColumnConfig::kColumnThreads, 0, stream>>>(
                    result, parameters, nullptr,
                    w, o, N, aD, up_or_down, m);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
