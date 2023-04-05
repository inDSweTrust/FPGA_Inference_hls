#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include "math.h"
#include "ap_fixed.h"

using namespace std;


// Data Types
typedef ap_fixed<26, 16> MM_DATA_T;
typedef float MM_DATA_T_REF;


//*************************************
//Neural Network
//*************************************

// Conv Layer
#define FILTER_ROWS 3
#define FILTER_COLS 3
#define FILTER_SIZE 9
#define PADDING 0
#define GEMM_ROWS 26
#define GEMM_COLS 26

// MaxPooling Layer
#define FILTER_ROWS_P 2
#define FILTER_COLS_P 2
#define FILTER_SIZE_P 4
#define GEMM_ROWS_P 13
#define GEMM_COLS_P 13
#define POOL_STRIDE 2

// Linear Layers
#define N 1
#define M1 169
#define M2 100
#define C 10

//Activation
#define POLY_COEFFS_SIGMOID_FILENAME "sigmoid.dat"
#define kNumActivationFunctions 1
#define POLY_ORDER 2
#define SIGMOID 0
#define ZMIN_LOC -4
#define ZMAX_LOC 4
#define RELU 1
#define DROPOUT_RATE 0.2


void im2col_img_ref(MM_DATA_T_REF image[28][28], MM_DATA_T_REF out[FILTER_SIZE][GEMM_ROWS*GEMM_COLS]);
void im2col_conv_ref(MM_DATA_T_REF conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T_REF out[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P]);


void GemmConv2d_ref(MM_DATA_T_REF image[FILTER_SIZE][GEMM_ROWS*GEMM_COLS], MM_DATA_T_REF w1[FILTER_SIZE], MM_DATA_T_REF out[FILTER_SIZE]);
void MaxPool2d_ref(MM_DATA_T_REF input[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P], MM_DATA_T_REF out[GEMM_ROWS_P][GEMM_ROWS_P]);
void Flatten_ref(MM_DATA_T_REF input[GEMM_ROWS_P][GEMM_ROWS_P], MM_DATA_T_REF out[GEMM_ROWS_P*GEMM_COLS_P]);
void Linear1_ref(MM_DATA_T_REF input[M1], MM_DATA_T_REF w[M1][M2], MM_DATA_T_REF b[M2], MM_DATA_T_REF out[M2]);
int Linear2_ref(MM_DATA_T_REF input[M2], MM_DATA_T_REF w[M2][C], MM_DATA_T_REF b[C]);

void im2col(MM_DATA_T img[28][28], MM_DATA_T img_gemm[FILTER_SIZE][GEMM_ROWS*GEMM_COLS]);
void im2col_conv(MM_DATA_T_REF conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T_REF out[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P]);

void GemmConv2d(MM_DATA_T image[28][28], MM_DATA_T image_gemm[FILTER_SIZE][GEMM_ROWS*GEMM_COLS], MM_DATA_T w1[FILTER_SIZE], MM_DATA_T b1, MM_DATA_T out[GEMM_ROWS][GEMM_COLS]);
void MaxPool2d(MM_DATA_T conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T conv_gemm[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P], MM_DATA_T out[GEMM_ROWS_P][GEMM_ROWS_P]);
void Flatten(MM_DATA_T input[GEMM_ROWS_P][GEMM_ROWS_P], MM_DATA_T out[GEMM_ROWS_P*GEMM_COLS_P]);
void Linear1(MM_DATA_T input[M1], MM_DATA_T w[M1][M2], MM_DATA_T b[M2], MM_DATA_T out[M2]);
int Linear2(MM_DATA_T input[M2], MM_DATA_T w[M2][C], MM_DATA_T b[C], MM_DATA_T coeffs[kNumActivationFunctions][POLY_ORDER+1]);

int cnn(MM_DATA_T image[28][28],
		MM_DATA_T image_gemm[FILTER_SIZE][GEMM_ROWS*GEMM_COLS],
		MM_DATA_T conv_out[GEMM_ROWS][GEMM_COLS],
		MM_DATA_T gemm_conv[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P],
		MM_DATA_T pool_out[GEMM_ROWS_P][GEMM_COLS_P],
		MM_DATA_T flattened[GEMM_ROWS_P*GEMM_COLS_P],
		MM_DATA_T linear1_out[M2],
		MM_DATA_T w1[FILTER_SIZE],
		MM_DATA_T w2[M1][M2],
		MM_DATA_T w3[M2][C],
		MM_DATA_T b1,
		MM_DATA_T b2[M2],
		MM_DATA_T b3[C],
		MM_DATA_T coeffs[kNumActivationFunctions][POLY_ORDER+1]);





