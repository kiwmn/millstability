#include "millstability/matrix_kernels.cuh"
#include "millstability/matrix_math.cuh"

namespace millstability {
namespace detail {

// Matrix recurrence for m = 1, with one thread per physical parameter row.
__global__ void multi_parameter_single_point_kernel(
    float* result_ptr, float* parameters,
    int N, float aD, int up_or_down, int m,
    float w, float o, int n_params
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (param_idx >= n_params) return;

    int param_offset = param_idx * 8;
    float Kt = parameters[param_offset + 0];
    float Kn = parameters[param_offset + 1];
    float w0x = parameters[param_offset + 2];
    float w0y = parameters[param_offset + 3];
    float zetax = parameters[param_offset + 4];
    float zetay = parameters[param_offset + 5];
    float m_tx = parameters[param_offset + 6];
    float m_ty = parameters[param_offset + 7];

    float fist, fiex;
    if (up_or_down == 1) {
        fist = 0.0f;
        fiex = acosf(1.0f - 2.0f * aD);
    } else {
        fist = acosf(2.0f * aD - 1.0f);
        fiex = M_PI;
    }

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
    float augmented[32];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            augmented[i * 8 + j] = A0[i * 4 + j];
            augmented[i * 8 + j + 4] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int pivot = 0; pivot < 4; pivot++) {

        int max_row = pivot;
        for (int i = pivot + 1; i < 4; i++) {
            if (fabsf(augmented[i * 8 + pivot]) > fabsf(augmented[max_row * 8 + pivot])) {
                max_row = i;
            }
        }

        if (max_row != pivot) {
            for (int j = 0; j < 8; j++) {
                float temp = augmented[pivot * 8 + j];
                augmented[pivot * 8 + j] = augmented[max_row * 8 + j];
                augmented[max_row * 8 + j] = temp;
            }
        }

        float pivot_val = augmented[pivot * 8 + pivot];
        if (fabsf(pivot_val) > 1e-10f) {
            for (int j = 0; j < 8; j++) {
                augmented[pivot * 8 + j] /= pivot_val;
            }

            for (int i = 0; i < 4; i++) {
                if (i != pivot) {
                    float factor = augmented[i * 8 + pivot];
                    for (int j = 0; j < 8; j++) {
                        augmented[i * 8 + j] -= factor * augmented[pivot * 8 + j];
                    }
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            invA0[i * 4 + j] = augmented[i * 8 + j + 4];
        }
    }

    float dtr = 2.0f * M_PI / N / m;
    float hxx[101], hxy[101], hyx[101], hyy[101];

    for (int i = 0; i <= m; i++) {
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

    int matrix_size = 2*m + 4;
    float Fi[84][84] = {0};
    float D[84][84] = {0};

    float d[82];
    for(int idx = 0; idx < 82; idx++) {
        d[idx] = 1.0f;
    }

    for(int idx = 0; idx < 4; idx++) {
        d[idx] = 0.0f;
    }

    for(int idx = 0; idx < 82; idx++) {
        int row = idx + 2;
        int col = idx;
        if(row < matrix_size && col < matrix_size) {
            D[row][col] = d[idx];
        }
    }

    D[4][0] = 1.0f;
    D[5][1] = 1.0f;

    for(int idx = 0; idx < matrix_size; idx++) {
        Fi[idx][idx] = 1.0f;
    }

    for (int i = 0; i < m; i++) {

        // A0k holds the next sample of cutting coefficients, multiplied by -depth.
        float A0k[16] = {0};
        if (i < m) {
            A0k[2*4 + 0] = -w * hxx[i + 1];
            A0k[2*4 + 1] = -w * hxy[i + 1];
            A0k[3*4 + 0] = -w * hyx[i + 1];
            A0k[3*4 + 1] = -w * hyy[i + 1];
        }

        // A1k is depth times the coefficient difference divided by dt.
        float A1k[16] = {0};
        if (i < m) {
            A1k[2*4 + 0] = w * (hxx[i + 1] - hxx[i]) / dt;
            A1k[2*4 + 1] = w * (hxy[i + 1] - hxy[i]) / dt;
            A1k[3*4 + 0] = w * (hyx[i + 1] - hyx[i]) / dt;
            A1k[3*4 + 1] = w * (hyy[i + 1] - hyy[i]) / dt;
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

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < matrix_size; col++) {
                D[row][col] = 0.0f;
            }
        }

        float Fi0_plus_F01[16];
        for(int idx = 0; idx < 16; idx++) {
            Fi0_plus_F01[idx] = s_Fi0[idx] + F01[idx];
        }

        for(int row = 0; row < 4; row++) {
            for(int col = 0; col < 4; col++) {
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
                for(int k = 0; k < 4; k++) {
                    D[row][2*m + 2 + col] -= invOfImFkp1[row*4 + k] * F01_12[k*2 + col];
                }
            }
        }

        float new_Fi[84][84] = {0};

        for(int row = 0; row < matrix_size; row++) {
            for(int col = 0; col < matrix_size; col++) {
                for(int k = 0; k < matrix_size; k++) {
                    new_Fi[row][col] += D[row][k] * Fi[k][col];
                }
            }
        }

        for(int row = 0; row < matrix_size; row++) {
            for(int col = 0; col < matrix_size; col++) {
                Fi[row][col] = new_Fi[row][col];
            }
        }
    }

    int result_offset = param_idx * matrix_size * matrix_size;
    for(int row = 0; row < matrix_size; row++) {
        for(int col = 0; col < matrix_size; col++) {
            result_ptr[result_offset + row * matrix_size + col] = Fi[row][col];
        }
    }

}

}
}
