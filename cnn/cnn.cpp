#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <math.h>  

#include "cnn.h"

using namespace std;

void im2col(MM_DATA_T img[28][28], MM_DATA_T img_gemm[FILTER_SIZE][GEMM_ROWS*GEMM_COLS]){

	int col = 0;
	im2col_1:for (int i=0; i<(28-FILTER_ROWS+1); i++){
		im2col_2:for (int j=0; j<(28-FILTER_COLS+1); j++){

			int row_offset = 0;
			int k = 0;

			im2col_3:for (int ii=0; ii<FILTER_COLS; ii++){
				im2col_4:for (int jj=0; jj<FILTER_ROWS; jj++){

					img_gemm[row_offset+k][col] = img[j + jj][i + ii];
					k++;
				}
		    }
			col++;
		}
	}
}

void im2col_conv(MM_DATA_T conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T out[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P]){

	int col = 0;
	for (int i=0; i<(GEMM_ROWS-FILTER_ROWS_P+1); i=i+POOL_STRIDE){
		for (int j=0; j<(GEMM_COLS-FILTER_COLS_P+1); j=j+POOL_STRIDE){

			int row_offset = 0;
			int k = 0;

			for (int ii=0; ii<FILTER_COLS_P; ii++){
				for (int jj=0; jj<FILTER_ROWS_P; jj++){

					out[row_offset+k][col] = conv[i + ii][j + jj];
					k++;
				}
		    }
			col++;
		}
	}
}



void GemmConv2d(MM_DATA_T image[28][28], MM_DATA_T image_gemm[FILTER_SIZE][GEMM_ROWS*GEMM_COLS], MM_DATA_T w1[FILTER_SIZE], MM_DATA_T b1, MM_DATA_T out[GEMM_ROWS][GEMM_COLS]){

	im2col(image, image_gemm);

	MM_DATA_T acc=0.0;
	MM_DATA_T zero=0.0;

    int i = 0;
    int j = 0;
	gemm_col:for (int col = 0; col < GEMM_ROWS*GEMM_COLS; col++){
		if (col % 26 == 0){
			j = 0;
			i++;
		}
		gemm_row:for (int row = 0; row < FILTER_SIZE; row++){
			if (row == 0) acc = 0.0;
			acc	 += image_gemm[row][col] * w1[row];  //Matrix multiplication result

			if (row == FILTER_SIZE - 1) {
				out[i][j] = acc + b1;  //No ReLU after last layer
				if (RELU == 1) {out[i][j] = max(out[i][j], zero);} //ReLU
				j++;
			}

		}

	}

}

void MaxPool2d(MM_DATA_T conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T conv_gemm[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P], MM_DATA_T out[GEMM_ROWS_P][GEMM_ROWS_P]){

	im2col_conv(conv, conv_gemm);

	MM_DATA_T out_gemm[GEMM_ROWS_P*GEMM_COLS_P];

	for (int col = 0; col < GEMM_ROWS_P*GEMM_COLS_P; col++){
		MM_DATA_T maxval = -INFINITY;
		for (int row = 0; row < FILTER_SIZE_P; row++){
			maxval = max(maxval, conv_gemm[row][col]);
		}
        out_gemm[col] = maxval;
	}

	// Reshape output
	int k = 0;
	for (int i=0; i<GEMM_ROWS_P; i++) {
		for (int j=0; j<GEMM_COLS_P; j++) {
			out[i][j] = out_gemm[k];
		    k++;
	    }
	}
}

void Flatten(MM_DATA_T input[GEMM_ROWS_P][GEMM_ROWS_P], MM_DATA_T out[GEMM_ROWS_P*GEMM_COLS_P]) {
	int k = 0;
	for (int i=0; i<GEMM_ROWS_P; i++) {
		for (int j=0; j<GEMM_COLS_P; j++) {
			out[k] = input[i][j];
		    k++;
	    }
	}
}

void Linear1(MM_DATA_T input[M1], MM_DATA_T w[M1][M2], MM_DATA_T b[M2], MM_DATA_T out[M2]){

	MM_DATA_T acc=0.0;
	MM_DATA_T zero=0.0;

	for (int j = 0; j < M2; j++)
	{
		for (int k = 0; k < M1; k++)
		{
			if (k==0) acc = 0.0;
			acc	 += input[k] * w[k][j];  //Matrix multiplication result

			if (k == M1 - 1)
			out[j] = acc + b[j];
			// out[j] = (acc + b[j]) * (1-DROPOUT_RATE); //Scale out by dropout rate
			if (RELU == 1) out[j] = max(out[j], zero); //ReLU
		}
	}
}

int Linear2(MM_DATA_T input[M2], MM_DATA_T w[M2][C], MM_DATA_T b[C], MM_DATA_T coeffs[kNumActivationFunctions][POLY_ORDER+1]){

	MM_DATA_T out[C];
	MM_DATA_T argmax, imax;

	MM_DATA_T acc=0.0;
	MM_DATA_T zero=0.0;

	MM_DATA_T abs_z;
	MM_DATA_T y_tmp;

	for (int j = 0; j < C; j++)
	{
		for (int k = 0; k < M2; k++)
		{
			if (k==0) acc = 0.0;
			acc	+= input[k] * w[k][j];  //Matrix multiplication result

			if (k == M2 - 1)
			out[j] = acc + b[j];

			abs_z = abs(float(out[j]));

			y_tmp = coeffs[SIGMOID][0] + coeffs[SIGMOID][1]*abs_z + coeffs[SIGMOID][2]*abs_z*abs_z;

			if ( out[j]  < 0.0 )
				y_tmp = 1 - y_tmp;

			if ( out[j] > ZMAX_LOC )
				y_tmp = 1;
			else if ( out[j] < ZMIN_LOC )
				y_tmp = 0;


			out[j] = y_tmp;

		}
	}

	//Prediction
	argmax = -INFINITY;
	for (int i = 0; i < C; i++) {

		if (out[i] > argmax) {          //find most confident class
			argmax = out[i] ;
			imax = i;
		}
	}

	return imax;
}


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
		MM_DATA_T coeffs[kNumActivationFunctions][POLY_ORDER+1]){

	GemmConv2d(image, image_gemm, w1, b1, conv_out);
	MaxPool2d(conv_out, gemm_conv, pool_out);
	Flatten(pool_out, flattened);
	Linear1(flattened, w2, b2, linear1_out);
	int pred = Linear2(linear1_out, w3, b3, coeffs);


	return pred;
}
