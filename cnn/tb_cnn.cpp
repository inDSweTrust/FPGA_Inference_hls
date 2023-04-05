#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <math.h>  

#include "cnn.h"

using namespace std;

void im2col_img_ref(MM_DATA_T_REF image[28][28], MM_DATA_T_REF out[FILTER_SIZE][GEMM_ROWS*GEMM_COLS]){

	int col = 0;
	for (int i=0; i<(28-FILTER_ROWS+1); i++){
		for (int j=0; j<(28-FILTER_COLS+1); j++){

			int row_offset = 0;
			int k = 0;

			for (int ii=0; ii<FILTER_COLS; ii++){
				for (int jj=0; jj<FILTER_ROWS; jj++){

					out[row_offset+k][col] = image[j + jj][i + ii];
					k++;
				}
		    }
			col++;
		}
	}
}

void im2col_conv_ref(MM_DATA_T_REF conv[GEMM_ROWS][GEMM_COLS], MM_DATA_T_REF out[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P]){
	
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


void GemmConv2d_ref(MM_DATA_T_REF image[FILTER_SIZE][GEMM_ROWS*GEMM_COLS], MM_DATA_T_REF w1[FILTER_SIZE], MM_DATA_T_REF b1, MM_DATA_T_REF out[GEMM_ROWS][GEMM_COLS]){

	MM_DATA_T_REF out_gemm[GEMM_ROWS*GEMM_COLS];

	MM_DATA_T_REF acc=0.0;
	MM_DATA_T_REF zero=0.0;

	for (int col = 0; col < GEMM_ROWS*GEMM_COLS; col++)
	{
		for (int row = 0; row < FILTER_SIZE; row++)
		{
			if (row == 0) acc = 0.0;
			acc	 += image[row][col] * w1[row];  //Matrix multiplication result

			if (row == FILTER_SIZE - 1)
			out_gemm[col] = acc + b1;  //No ReLU after last layer
			if (RELU == 1) out_gemm[col] = max(out_gemm[col], zero); //ReLU
		}
	}

	// Reshape output
	int k = 0;
	for (int i=0; i<GEMM_ROWS; i++) {
		for (int j=0; j<GEMM_COLS; j++) {
			out[i][j] = out_gemm[k];
		    k++;
	    }
	}
}

void MaxPool2d_ref(MM_DATA_T_REF input[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P], MM_DATA_T_REF out[GEMM_ROWS_P][GEMM_ROWS_P]){

	MM_DATA_T_REF out_gemm[GEMM_ROWS_P*GEMM_COLS_P];

	for (int col = 0; col < GEMM_ROWS_P*GEMM_COLS_P; col++){
		MM_DATA_T_REF maxval = -INFINITY;
		for (int row = 0; row < FILTER_SIZE_P; row++){
			maxval = max(maxval, input[row][col]);
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

void Flatten_ref(MM_DATA_T_REF input[GEMM_ROWS_P][GEMM_ROWS_P], MM_DATA_T_REF out[GEMM_ROWS_P*GEMM_COLS_P]) {
	int k = 0;
	for (int i=0; i<GEMM_ROWS_P; i++) {
		for (int j=0; j<GEMM_COLS_P; j++) {
			out[k] = input[i][j];
		    k++;
	    }
	}
}

void Linear1_ref(MM_DATA_T_REF input[M1], MM_DATA_T_REF w[M1][M2], MM_DATA_T_REF b[M2], MM_DATA_T_REF out[M2]){
	
	MM_DATA_T_REF acc=0.0;
	MM_DATA_T_REF zero=0.0;

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

int Linear2_ref(MM_DATA_T_REF input[M2], MM_DATA_T_REF w[M2][C], MM_DATA_T_REF b[C]){

	MM_DATA_T_REF out[C];
	MM_DATA_T_REF argmax, imax;

	MM_DATA_T_REF acc=0.0;
	MM_DATA_T_REF zero=0.0;

	for (int j = 0; j < C; j++)
	{
		for (int k = 0; k < M2; k++)
		{
			if (k==0) acc = 0.0;
			acc	+= input[k] * w[k][j];  //Matrix multiplication result

			if (k == M2 - 1)
			out[j] = acc + b[j];
			// out[j] = (acc + b[j]) * (1-DROPOUT_RATE); //Scale out by dropout rate
			out[j] = 1.0 / (1.0 + exp(-out[j]));  //SIGMOID
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



int main() {


    //**************************************
    // Initialize arrays & variables
	//**************************************
	MM_DATA_T_REF targets_ref[10000];

	MM_DATA_T_REF w1_ref[FILTER_SIZE];
	MM_DATA_T_REF w2_ref[M1][M2];
	MM_DATA_T_REF w3_ref[M2][C];

	MM_DATA_T_REF b1_ref;
	MM_DATA_T_REF b2_ref[M2];
	MM_DATA_T_REF b3_ref[C];

	MM_DATA_T targets[10000];

	MM_DATA_T w1[FILTER_SIZE];
	MM_DATA_T w2[M1][M2];
	MM_DATA_T w3[M2][C];

	MM_DATA_T b1;
	MM_DATA_T b2[M2];
	MM_DATA_T b3[C];

	int row, col;
	int pred, pred_ref;
	int index = 0;

	MM_DATA_T_REF accuracy_ref;
	MM_DATA_T_REF num_correct_ref = 0.0;

	MM_DATA_T accuracy;
	MM_DATA_T num_correct = 0.0;

	// ************************************
    // Read weights and biases
    //*************************************

	// Read w1
	//**************************************
    string line_w1;
    ifstream f_w1 ("w0.csv", ifstream::in);

    if (!f_w1.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + string("w0.csv")).c_str());
        return 0;
    }
    std::cout << "opening file: " << "w0.csv" << std::endl;

    col=0;
    while (getline (f_w1, line_w1)) {
        string val;
        stringstream s (line_w1);
        while (getline (s, val, ',')){
        	w1_ref[col] = stof(val);
        	col++;
        }
    }
    f_w1.close();

	// Read w2
	//**************************************
    string line_w2;
    ifstream f_w2 ("w3.csv", ifstream::in);

    if (!f_w2.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + string("w3.csv")).c_str());
        return 0;
    }
    std::cout << "opening file: " << "w3.csv" << std::endl;

    row=0;
    while (getline (f_w2, line_w2)) {
        string val;
        stringstream s (line_w2);
		col = 0;
        while (getline (s, val, ',')){
        	w2_ref[row][col] = stof(val);
        	col++;
        }
		row++;
    }
    f_w2.close();

	
	// Read w3
	//**************************************
    string line_w3;
    ifstream f_w3 ("w5.csv", ifstream::in);

    if (!f_w3.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + string("w5.csv")).c_str());
        return 0;
    }
    std::cout << "opening file: " << "w5.csv" << std::endl;

    row=0;
    while (getline (f_w3, line_w3)) {
        string val;
        stringstream s (line_w3);
		col = 0;
        while (getline (s, val, ',')){
        	w3_ref[row][col] = stof(val);
        	col++;
        }
		row++;
    }
    f_w3.close();


    // Read b1
	//**************************************
	string line_b1;
	ifstream f_b1 ("b0.csv", ifstream::in);

	if (!f_b1.is_open()) {     /* validate file open for reading */
		perror (("error while opening file " + string("b0.csv")).c_str());
		return 0;
	}
	std::cout << "opening file: " << "b0.csv" << std::endl;

	while (getline (f_b1, line_b1)) {
		string val;
		stringstream s (line_b1);
		while (getline (s, val, ',')){
			b1_ref = stof(val);
		}
	}
	f_b1.close();

    // Read b2
	//**************************************
	string line_b2;
	ifstream f_b2 ("b3.csv", ifstream::in);

	if (!f_b2.is_open()) {     /* validate file open for reading */
		perror (("error while opening file " + string("b3.csv")).c_str());
		return 0;
	}
	std::cout << "opening file: " << "b3.csv" << std::endl;
    row = 0;
	while (getline (f_b2, line_b2)) {
		string val;
		stringstream s (line_b2);
		while (getline (s, val, ',')){
			b2_ref[row] = stof(val);
		}
		row++;
	}
	f_b2.close();

    // Read b3
	//**************************************
	string line_b3;
	ifstream f_b3 ("b5.csv", ifstream::in);

	if (!f_b3.is_open()) {     /* validate file open for reading */
		perror (("error while opening file " + string("b5.csv")).c_str());
		return 0;
	}
	std::cout << "opening file: " << "b5.csv" << std::endl;
    row = 0;
	while (getline (f_b3, line_b3)) {
		string val;
		stringstream s (line_b3);
		while (getline (s, val, ',')){
			b3_ref[row] = stof(val);
		}
		row++;
	}
	f_b3.close();


	// Copy floating point data to fixed point arrays
	//*************************************************
	std::cout << "Creating fixed point arrays... " << std::endl;

	for (int i = 0; i < FILTER_SIZE; i++){
			w1[i] = w1_ref[i];
		}

	for (int i = 0; i < M1; i++){
		for (int j = 0; j < M2; j++){
			w2[i][j] = w2_ref[i][j];
		}
	}
	for (int i = 0; i < M2; i++){
		for (int j = 0; j < C; j++){
			w3[i][j] = w3_ref[i][j];
		}
	}

	b1 = b1_ref;

	for (int i = 0; i < M2; i++){
		b2[i] = b2_ref[i];
	}

	for (int i = 0; i < C; i++){
		b3[i] = b3_ref[i];
	}

    // ************************************
    // Read Sigmoid approximation coeffs
    //*************************************

	std::ifstream poly_coeffs_sigmoid;						// coefficients for polynomial approximation - sigmoid
	MM_DATA_T coeffs[kNumActivationFunctions][POLY_ORDER+1];  // polynomial coefficients


	poly_coeffs_sigmoid.open(POLY_COEFFS_SIGMOID_FILENAME);
	if (poly_coeffs_sigmoid.is_open() == 0){
		std::cout << "failure opening file: " << POLY_COEFFS_SIGMOID_FILENAME << std::endl;
		exit(0);
	}
	else{
		std::cout << "opening file: " << POLY_COEFFS_SIGMOID_FILENAME << std::endl;
	}

	// read coefficients - sigmoid
	float floating_point_tmp;

	for (int j = 0; j <= POLY_ORDER; j++)
	{
		poly_coeffs_sigmoid >> floating_point_tmp;
		coeffs[SIGMOID][j] =  floating_point_tmp;
	}

	poly_coeffs_sigmoid.close();



    // Load Targets
	//**************************************
	string line_t;
	ifstream f_t ("mnist_test_target.csv", ifstream::in);

	if (!f_t.is_open()) {     /* validate file open for reading */
		perror (("error while opening file " + string("mnist_test_target.csv")).c_str());
		return 0;
	}
	std::cout << "opening file: " << "mnist_test_target.csv" << std::endl;
	row = 0;
	while (getline (f_t, line_t)) {
		string val;
		stringstream s (line_t);
		while (getline (s, val, ',')){
			targets_ref[row] = stof(val);
		}
		row++;
	}
	f_t.close();

	for (int i = 0; i < 10000; i++){
	    targets[i] = targets_ref[i];
	}

    //**************************************************
    // Load Images & pass each image though conv layer
	//**************************************************
	string line;
	ifstream f ("mnist_test_images.csv", ifstream::in);

	if (!f.is_open()) {
		perror (("error while opening file " + string("mnist_test_images.csv")).c_str());
		return 1;
	}
	std::cout << "opening file: " << "mnist_test_images.csv" << std::endl;

	std::cout << "Running Inference..." << std::endl;

	while (getline (f, line)) {
		string val;
		//floating point
		MM_DATA_T_REF image_ref[28][28];
		MM_DATA_T_REF gemm_image_ref[FILTER_SIZE][GEMM_ROWS*GEMM_COLS];
		MM_DATA_T_REF conv_out_ref[GEMM_ROWS][GEMM_COLS];
		MM_DATA_T_REF gemm_conv_ref[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P];
		MM_DATA_T_REF pool_out_ref[GEMM_ROWS_P][GEMM_COLS_P];
		MM_DATA_T_REF flattened_ref[GEMM_ROWS_P*GEMM_COLS_P];
		MM_DATA_T_REF linear1_out_ref[M2];
		//fixed point
		MM_DATA_T image[28][28];
		MM_DATA_T gemm_image[FILTER_SIZE][GEMM_ROWS*GEMM_COLS];
		MM_DATA_T conv_out[GEMM_ROWS][GEMM_COLS];
		MM_DATA_T gemm_conv[FILTER_SIZE_P][GEMM_ROWS_P*GEMM_COLS_P];
		MM_DATA_T pool_out[GEMM_ROWS_P][GEMM_COLS_P];
		MM_DATA_T flattened[GEMM_ROWS_P*GEMM_COLS_P];
		MM_DATA_T linear1_out[M2];
 
		stringstream s(line);
		col = 0;
        row = 0;
		while (getline(s, val, ',')) {
            if (col==28) {
                col = 0;
                row++;
            }
            image_ref[col][row] = stof(val);	
			col++;
		}


//		 cout << "IMAGE" << endl;
//		 for (int i=0; i<28; i++) {
//		 	for (int j=0; j<28; j++) {
//		 		cout << image_ref[i][j] << " ";
//		 	}
//		 	cout << endl;
//		 }

		// Conv Layer
        im2col_img_ref(image_ref, gemm_image_ref);
        GemmConv2d_ref(gemm_image_ref, w1_ref, b1_ref, conv_out_ref);
		// Max Pooling
		im2col_conv_ref(conv_out_ref, gemm_conv_ref);
		MaxPool2d_ref(gemm_conv_ref, pool_out_ref);
		// Flatten 
		Flatten_ref(pool_out_ref, flattened_ref);
		// Linear 1
		Linear1_ref(flattened_ref, w2_ref, b2_ref, linear1_out_ref);
		// Linear 2 and Prediction 
	    pred_ref = Linear2_ref(linear1_out_ref, w3_ref, b3_ref);
		if (pred_ref == targets_ref[index]) {num_correct_ref++;}


		for (int i=0; i<28; i++) {
			for (int j=0; j<28; j++) {
				image[i][j] = image_ref[i][j];
			}
		 }

		//RUN HLS IMPL
		pred = cnn(image, gemm_image, conv_out, gemm_conv, pool_out, flattened, linear1_out, w1, w2, w3, b1, b2, b3, coeffs);
		if (pred == targets[index]) {num_correct++;}



//		break;

		index++;

	}
	f.close();


	cout << "NUM CORRECT FIXED POINT " << num_correct << "  ";
	cout << "\n";
    cout << "NUM CORRECT REFERENCE " << num_correct_ref << "  ";
    cout << "\n";

    // Evaluate and print Accuracy
	accuracy = num_correct / 10000;
	accuracy_ref = num_correct_ref / 10000;

    cout << "ACCURACY HLS " << accuracy << "  ";
    cout << "\n";

    cout << "ACCURACY REFERENCE " << accuracy_ref << "  ";
    cout << "\n";

    return 0;
}
