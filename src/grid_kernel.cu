#include "millstability/matrix_kernels.cuh"
#include "millstability/matrix_math.cuh"

namespace millstability {
namespace detail {

// Matrix recurrence for m = 1: each block owns a speed, each thread a depth.
__global__ void stability_analysis_kernel(
    float* result_ptr, float* d_A0, float* d_invA0,
    float* d_hxx, float* d_hxy, float* d_hyx, float* d_hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m
) {

    int x = blockIdx.x;
    int y = threadIdx.x;

    __shared__ float s_Fi0[16];
    __shared__ float s_Fi1[16];
    __shared__ float s_Fi2[16];
    __shared__ float s_Fi3[16];

    if (x <= stx && y <= sty) {
        float o = o_st + (float)x * (o_fi - o_st) / stx;
        // One tooth-passing period in seconds; dt is one of its m time steps.
        float tau = 60.0f / o / N;
        float dt = tau / m;

        if (y  == 0) {

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
        }

        extern __shared__ float shared_memory[];
        float* d_hxx_shared = shared_memory;
        float* d_hxy_shared = &d_hxx_shared[m+1];
        float* d_hyx_shared = &d_hxy_shared[m+1];
        float* d_hyy_shared = &d_hyx_shared[m+1];

        if(threadIdx.x == 0) {
            for(int idx = 0; idx < m+1; idx++) {
                d_hxx_shared[idx] = d_hxx[idx];
                d_hxy_shared[idx] = d_hxy[idx];
                d_hyx_shared[idx] = d_hyx[idx];
                d_hyy_shared[idx] = d_hyy[idx];
            }
        }

        __syncthreads();

        float w = w_st + (float)y * (w_fi - w_st) / sty;

        int matrix_size = 2*m + 4;

        float D[84][84] = {0};
        float Fi[84][84] = {0};

        float d[82];
        for(int idx = 0; idx < 2*m+2; idx++) {
            d[idx] = 1.0f;
        }

        for(int idx = 0; idx < 4; idx++) {
            d[idx] = 0.0f;
        }

        for(int idx = 0; idx < 2*m+2; idx++) {
            int row = idx + 2;
            int col = idx;
            if(row < 84 && col < 84) {
                D[row][col] = d[idx];
            }
        }

        // The remaining matrix rows shift the two displacement histories.
        D[4][0] = 1.0f;

        D[5][1] = 1.0f;

        for(int idx = 0; idx < 84; idx++) {
            Fi[idx][idx] = 1.0f;
        }

        for (int i = 0; i < m; i++) {

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
            matrix_inverse_4x4(I_minus_Fkp1, invOfImFkp1);

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
                    D[row][2*m + col] = 0.0f;
                    for(int k = 0; k < 4; k++) {
                        D[row][2*m + col] -= invOfImFkp1[row*4 + k] * Fkp1_12[k*2 + col];
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
                    D[row][2*m + 2 + col] = 0.0f;
                    for(int k = 0; k < 4; k++) {
                        D[row][2*m + 2 + col] -= invOfImFkp1[row*4 + k] * F01_12[k*2 + col];
                    }
                }
            }

            float new_Fi[84][84] = {0};

            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < matrix_size; ++col) {
                    float value = 0.0f;
                    // Overlapping current-state and delay columns require the full row sum.
                    if (m == 1) {
                        for (int k = 0; k < matrix_size; ++k) {
                            value += D[row][k] * Fi[k][col];
                        }
                    } else {
                        for (int k = 0; k < 4; ++k) {
                            value += D[row][k] * Fi[k][col];
                        }
                        for (int k = 2 * m; k < matrix_size; ++k) {
                            value += D[row][k] * Fi[k][col];
                        }
                    }
                    new_Fi[row][col] = value;
                }
            }
            for (int col = 0; col < matrix_size; ++col) {
                new_Fi[4][col] = Fi[0][col];
                new_Fi[5][col] = Fi[1][col];
            }
            for (int row = 6; row < matrix_size; ++row) {
                for (int col = 0; col < matrix_size; ++col) {
                    new_Fi[row][col] = Fi[row - 2][col];
                }
            }

            for(int row = 0; row < matrix_size; row++) {
                for(int col = 0; col < matrix_size; col++) {
                    Fi[row][col] = new_Fi[row][col];
                }
            }

        }

        if(x <= stx && y <= sty){

            int result_idx = x * (sty + 1) + y;
            for(int row = 0; row < 2*m + 4; row++){
                for(int col = 0; col < 2*m + 4; col++){
                    result_ptr[result_idx * (2*m + 4) * (2*m + 4) + row * (2*m + 4) + col] = Fi[row][col];
                }
            }
        }

    }
}

// Sum the engaged teeth at each sample; these coefficients serve the whole grid.
__global__ void compute_cutting_coefficients_kernel(
    float* hxx, float* hxy, float* hyx, float* hyy,
    float fist, float fiex, float Kt, float Kn,
    int N, int m, float dtr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m + 1) return;

    float hxx_sum = 0.0f, hxy_sum = 0.0f, hyx_sum = 0.0f, hyy_sum = 0.0f;

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

    hxx[i] = hxx_sum;
    hxy[i] = hxy_sum;
    hyx[i] = hyx_sum;
    hyy[i] = hyy_sum;
}

}
}
