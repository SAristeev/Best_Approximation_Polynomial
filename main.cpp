#define _CRT_SECURE_NO_WARNINGS // ��� fopen � visual studio
#include <cmath>
#include <iostream>
#include <cstdio>
#include "Eigen/Dense"


using Eigen::MatrixXd;
using Eigen::VectorXd;

double f(double x)
{
	return 4*sin(3*x) + 2*x*x;
}

void py() {
	// fast hardcode style
	// ���� ���� ������ �������� python � c++ �����, �������� � tg
	// �������� ������ ������ ��� ��������� ��������� ���������� ��� ���������� � ����� ������� ����. 
	FILE* res;
	res = fopen("plot.py", "w");

	fprintf(res, "import numpy as np\n");
	fprintf(res, "import matplotlib.pyplot as plt\n");
	fprintf(res, "Params = np.genfromtxt('Params.txt', delimiter = ', ')\n");
	fprintf(res, "mesh = np.genfromtxt('mesh.txt', delimiter = ', ')\n");
	fprintf(res, "mesh = np.genfromtxt('mesh.txt', delimiter = ',')\n");
	fprintf(res, "Fmesh = np.genfromtxt('Fmesh.txt', delimiter = ',')\n");
	fprintf(res, "X = np.genfromtxt('X.txt', delimiter = ',')\n");
	fprintf(res, "Approx = np.genfromtxt('Approx.txt', delimiter = ',')\n");
	fprintf(res, "F = np.genfromtxt('F.txt', delimiter = ',')\n");
	fprintf(res, "K = np.genfromtxt('K.txt', delimiter = ',')\n");
	fprintf(res, "KF = np.genfromtxt('KF.txt', delimiter = ',')\n");
	fprintf(res, "\n");
	
	fprintf(res, "plt.figure(figsize = (16 * 2, 9 * 2))\n");
	fprintf(res, "plt.title('Best Approximation Polynomial \\n fast hardcode by @SAristeev', fontsize = 15, pad = 15)\n");
	fprintf(res, "plt.xlim([Params[0], Params[1]])\n");
	fprintf(res, "plt.plot(X, F, color = 'C0', label = 'f(x)')\n");
	fprintf(res, "plt.plot(X, Approx, color = 'C1', label = 'Approx(x)')\n");
	fprintf(res, "plt.scatter(mesh, Fmesh, color = 'red', s = 8, label = 'Interpolation points')\n");
	fprintf(res, "plt.scatter(K, KF, color = 'blue', s = 16, label = 'Element boundaries')\n");

	fprintf(res, "plt.ylabel('y')\n");
	fprintf(res, "plt.xlabel('x')\n");
	fprintf(res, "plt.legend()\n");
	fprintf(res, "plt.grid(True)\n");
	fprintf(res, "plt.savefig('plot.png')\n");
	fprintf(res, "plt.show()\n");
	fclose(res);
}

int main()
{
	double a = -2, b = 2; // ������ � ����� �������

	const int K = 5; // ���������� �������� ���������
	const int N = 5; // ���������� ����� �� 1 �������� ��������
	const int L = 40; // ���������� ��������� �����
	const int M_ = K * N; // ���������� �������� (��������) �������

	const int M = K * (N - 1) + 1; // ���������� �������� (��������) �������, ��� �� ���������� �����
	double h = (b - a) / (M - 1); // ��� �� ����������� �����

	double mesh[M]; // ����������� ����� �� �����
	for (int i = 0; i < M; i++)
	{
		mesh[i] = a + i * h; // ��������� ������ �����
	}

	const int Mviz = 1920; // ���������� ����� ��� ���������
	double hviz = (b - a) / (Mviz - 1); // ��� ��� ���

	double X[Mviz]; // ������ ����� ��� ���������
	double Approx[Mviz]; // �������� ���� � ���� ������
	for (int iviz = 0; iviz < Mviz; iviz++)
	{
		X[iviz] = a + iviz * hviz; // ���������� ������� ����� ��� ���������
	}


	///////////////////
	// ������������� //
	///////////////////



	// ������ ������������
	// ��� ��� ���� �� ����������� �����, �� �� ������ �������� ��������
	// ����������� ����� �����������, �� ���� �� ���������� ������ N ������������
	double D[N];
	for (int i = 0; i < N; i++) // ���������� ������������ �����������
	{
		// mesh[ki * (N - 1) + i] - i-�� ����� �������� ��������� ��������
		D[i] = 1;
		for (int j = 0; j < N; j++)
		{
			if (i != j) {
				D[i] *= (mesh[i] - mesh[j]);
			}
		}
	}


	// ������ ���� ��������� ����� ��� �������� ���������� ������������
	// ��� ��� ��� ���������� ���� ����� ������� ������ ����, �� ��� �������� ������� ��� ����� �����
	double random_mesh[K * L];
	//srand(time(0)); // �������������� ������������
	for (int ki = 0; ki < K; ki++) // ����������� �� �������� ���������
	{
		for (int l = 0; l < L; l++) // ��� ������� �������� ������� L �����
		{
			// mesh[ki * (N - 1)] - ������ �������� ��������� ��������
			// mesh[(ki + 1) * (N - 1)] - ������ ���������� ��������� ��������
			// random_mesh[ki * L + l] - l-�� ��������� ����� ��������� ��������
			random_mesh[ki * L + l] = mesh[ki * (N - 1)] + (mesh[(ki + 1) * (N - 1)] - mesh[ki * (N - 1)]) * double(rand()) / (RAND_MAX);
		}
	}



	/////////////////////////
	// ���������� �������� //
	/////////////////////////

	// G[i] - �������� �� i-�� ������� 
	// �� ���������� ������� �� ��������, 
	// ������� ����� �� M = K * (N - 1) + 1
	// G[i][l] - �������� i-�� ������� �� l-�� ��������� �����
	// �� ���� G[i] - ������ �������� ������� �� L ��������� ������
	// ��� ��� � ��� ���� �������� �� ������� �������� ��������� �������
	// �� ��� G[i], ��� i % N - 1 == 0 (�������� �� ��������)
	// � ��� ���������� 2 �������� ���������
	// ����� ��������� ������� ����� ������� �� 2*L ������
	// � ��������� ��������� ������������
	// �� �������������� �������� ���������

	double G[M][2 * L];
	double F[M][2 * L];

	// ����������� �� �������� ���������
	// ����� ����������� ������
	// N = 4 K = 3
	// 0 0 0 * 0 0 * 0 0 0
	// 0 - ������� ������������ �� ������ ����
	// * - ������� ������������ �� ���� �����
	// �� ���� * ��������� �������
	// � ���� ����� ����� ������� �������� ���� ������� � ��������� ������
	for (int ki = 0; ki < K; ki++)
	{
		for (int i = 0; i < N; i++)
		{
			if (i == N - 1 && ki != K - 1) // ������� ��������� � * �������, ���� ��� ��������� � �������� ��������
			{
				for (int l = 0; l < L; l++)
				{
					G[ki * (N - 1) + i][l] = 1 / D[N - 1]; // �������� ������� �� ����� �������� ��������
					G[ki * (N - 1) + i][l + L] = 1 / D[0]; // �������� ������� �� ������ �������� ��������

					// (f,g) ��� ������ ����� � ����
					// ��� ��� �������� f �� 0 �� ���� ������� [a,b]
					// �� ������ g �� 0 ������ �� ����� ��� ���� �������� ���������
					// �� ���������� ������� ������ �� ����������� �������� ���������

					F[ki * (N - 1) + i][l] = f(random_mesh[ki * L + l]); // �������� f �� ����� �������� ��������
					F[ki * (N - 1) + i][l + L] = f(random_mesh[(ki + 1) * L + l]); // �������� f �� ������ �������� ��������

					for (int j = 0; j < N; j++)
					{
						if (j != N - 1) // ��� ������ ��������� �������� ������� ��������� �� ��������� ���� ����� ��������
						{
							G[ki * (N - 1) + i][l] *= (random_mesh[ki * L + l] - mesh[ki * (N - 1) + j]);

						}
						if (j != 0) // ��� ������� ��������� �������� ������� ��������� �� ������ ���� ����� ��������
						{
							G[ki * (N - 1) + i][l + L] *= (random_mesh[(ki + 1) * L + l] - mesh[(ki + 1) * (N - 1) + j]);
						}
					}
				}
			}
			else if (i != 0 || ki == 0) // ��� ��������� (i == 0 && ki != 0) - �� �� ������� ��� �������, �� ��������� �� ��� �� ���������� �������
			{
				for (int l = 0; l < L; l++)
				{
					// ����������, �� ������ ��� ������, ����������� ��������� ��������
					G[ki * (N - 1) + i][l] = 1 / D[i];
					F[ki * (N - 1) + i][l] = f(random_mesh[ki * L + l]);
					for (int j = 0; j < N; j++)
					{
						if (i != j)
						{
							G[ki * (N - 1) + i][l] *= (random_mesh[ki * L + l] - mesh[ki * (N - 1) + j]);
						}
					}
				}
			}
		}
	}

	//// ����� ������ ������ Ax=b
	//// ������� ����������� ����� ������ � ��������
	////
	
	/////////////////////
	// �������� ������ //
	/////////////////////
	
	// ������������ ������� (�������� ��� �������)
	MatrixXd Eigen_A = MatrixXd::Zero(M, M);
	
	// ��������� ������� (�������� ������ ������� �����)
	// ��� ������� ������������ � ������������ �������
	double A_[M][N];

	// ��������� ������� (�������� ������ ������� �����)
	// ������������ � ������ ������
	// �������� ��� �������
	double Gauss_A[M][N];


	// ������� ��� ���������� �������� � �� �����
	// ����� �� ������ �������� �������
	// �� ���� ���������� ������� ������� A �� ��� �� ����������
	
	// ��� ��������� ������� LU ����������
	// U_ - �����������������, �������� ������ ������� �����
	// L_ - ����������������, �������� �� ����������������� �����, �� ��� �����������������
	double L_[M][N], U_[M][N];

	// ��������� ������� ���������� ���������
	// �_ - �����������������, �������� ������ ������� �����
	double C_[M][N];


	// ���������� ������ ������
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			A_[i][j] = 0;
			Gauss_A[i][j] = 0;
			U_[i][j] = 0;
			C_[i][j] = 0;
			if (j == 0) {
				L_[i][j] = 1; // � ������� L �� ��������� ����� 1
			}
			else {
				L_[i][j] = 0;
			}
		}
	}

	///////////////////////
	// �������� �������� //
	///////////////////////

	// ������������ ������
	VectorXd Eigen_b = VectorXd::Zero(M);

	// ��� ������� ��� LU ����������
	// Ax=b <=> LUx=b
	// Ux=y, ��� y ������ �� Ly=b
	double LU_x[M], LU_y[M];

	// ��� ������� ��� ���������� ���������
	// ���� �� ��, ��� � LU ����������
	// ������ 2 �������
	double Cholesky_x[M], Cholesky_y[M];

	// ������ ������ � ������-����� ������ ����� b
	// �� ���������, ��� ��� � ������ ���������� ������ �������
	// ��� � ������ ������ �����
	double Gauss_x[M], Gauss_b[M];

	// ����� ��� ���� ������� ������ ������ �����
	double b_[M];

	// ���������� �������� ������
	for (int i = 0; i < M; i++) {
		LU_x[i] = 0;
		LU_y[i] = 0;
		Cholesky_x[i] = 0;
		Cholesky_y[i] = 0;
		Gauss_b[i] = 0;
		Gauss_x[i] = 0;
		b_[i] = 0;
	}

	// �������� � ���������� ��������
	// ��� ������������ ������� ��������� ����� ������
	// ������������� ��������




	//////////////////////////////////
	// ���������� ������ � �������� //
	//////////////////////////////////
	for (int ki = 0; ki < K; ki++) // �������� �� �������� ���������
	{
		for (int i = 0; i < N; i++) // ����������� �� �������� ������ ������� ��������� ��������
		{
			double scalar_product_ii = 0; // (gi,gi) i � ��������� ���� �� ������
			double scalar_product_fi = 0; // (f,gi)
			int local = ki * (N - 1);
			// ������� ��������� ������������ �������� �������
			// �� ���� �������� [*][0] �� ���� ��������
			
			// � ������ ������������ ������� �������
			// ��� ���������� ������ �� 1 �������� ��������
			// �� ������� ������� �� ������ L ������
			if ((i > 0 && i < N - 1) || (i == 0 && ki == 0) || (i == N - 1 && ki == K - 1))
			{
				for (int l = 0; l < L; l++)
				{
					scalar_product_ii += G[ki * (N - 1) + i][l] * G[ki * (N - 1) + i][l];
					scalar_product_fi += G[ki * (N - 1) + i][l] * F[ki * (N - 1) + i][l];
				}


				// ���������� ������� A
				A_[local + i][0] = scalar_product_ii; 
				// ���������� ������� A ��� ������, ����� �������� �
				Gauss_A[local + i][0] = scalar_product_ii; 
				// ���������� ������� ������������ A, � ������ �����
				Eigen_A(ki * (N - 1) + i, ki * (N - 1) + i) = scalar_product_ii;

				// ���������� ������������� ������� b
				Eigen_b(ki * (N - 1) + i) = scalar_product_fi;
				// ���������� ������� b
				b_[ki * (N - 1) + i] = scalar_product_fi;
				// ���������� ������� b ��� ������, ����� ��������� b
				Gauss_b[ki* (N - 1) + i] = scalar_product_fi;
			}
			// � ������ ������������ ������� �������
			// ��� ���������� ������ �� 2 �������� ���������
			// �� ������� ������� �� 2 * L ������
			else {
				// ���� �� � ����� ��������� ��������, �� ���������� ���������
				// ��� ���������� ��������� �������� ������� ����� �� L ������
				// �� ������� ������� ���������� �� 1 �������� ��������
				if (i == N - 1 && ki != K - 1)
				{
					for (int l = 0; l < L; l++)
					{
						scalar_product_ii += G[ki * (N - 1) + i][l] * G[ki * (N - 1) + i][l];
						scalar_product_ii += G[ki * (N - 1) + i][L + l] * G[ki * (N - 1) + i][L + l];
						scalar_product_fi += G[ki * (N - 1) + i][l] * F[ki * (N - 1) + i][l];
						scalar_product_fi += G[ki * (N - 1) + i][L + l] * F[ki * (N - 1) + i][L + l];
					}
					// ���������� ������� A
					A_[local + i][0] = scalar_product_ii;
					// ���������� ������� A ��� ������, ����� �������� �
					Gauss_A[local + i][0] = scalar_product_ii;
					// ���������� ������� ������������ A, � ������ �����
					Eigen_A(ki * (N - 1) + i, ki * (N - 1) + i) = scalar_product_ii;

					// ���������� ������������� ������� b
					Eigen_b(ki * (N - 1) + i) = scalar_product_fi;
					// ���������� ������� b
					b_[ki * (N - 1) + i] = scalar_product_fi;
					// ���������� ������� b ��� ������, ����� ��������� b
					Gauss_b[ki * (N - 1) + i] = scalar_product_fi;
					}
			}
			


			// ���������� ������������ �������� U_[*][0]
			for (int q = 1; q < i + 1; q++) {
				U_[local + i][0] -= U_[local + i - q][q] * L_[local + i - q][q];
			}
			U_[local + i][0] += scalar_product_ii;

			// ���������� ������������ �������� C_[*][0]
			if (ki == 0 && i == 0) {
				C_[0][0] = sqrt(scalar_product_ii);
			}
			else if (i != 0){
				for (int q = 1; q < i + 1; q++) {
					C_[local + i][0] -= C_[local + i - q][q] * C_[local + i - q][q];
				}
				C_[local + i][0] += scalar_product_ii;
				C_[local + i][0] = sqrt(C_[local + i][0]);
			}



			// ��������������� ��������
			for (int j = i + 1; j < N; j++)
			{
				// res - ����� ��������� � ������������ �������
				int res = j - i;
				double scalar_product_ij = 0; // ���������� scalar_product_ii

				// ���� ������� �������, �� �� ��������� ������������ ����� ��������� 
				// ������ �� ������ ����� �����
				if (i == 0 && ki != 0)  
				{
					// G[ki * (N - 1)][l + L] - ������� �������, ������ ����� �����
					for (int l = 0; l < L; l++)
					{	
						scalar_product_ij += G[ki * (N - 1)][l + L] * G[ki * (N - 1) + j][l];
					}
				}
				else 
				{
					// � ��������� ������ ������� �������� ������� �� ������ L ������
					for (int l = 0; l < L; l++)
					{
						scalar_product_ij += G[ki * (N - 1) + i][l] * G[ki * (N - 1) + j][l];
					}
				}

						
				// ���������� �������������� ��������� ������ LU ����������
				for (int q = 1; q < i + 1; q++) {
					U_[local + i][res] -= U_[local + i - q][res + q] * L_[local + i - q][q];
					L_[local + i][res] -= L_[local + i - q][res + q] * U_[local + i - q][q];
				}
				U_[local + i][res] += scalar_product_ij;
				L_[local + i][res] += scalar_product_ij;
				L_[local + i][res] /= U_[local + i][0];

				// ���������� �������������� ��������� ������ ���������� ���������
				if (i == 0 && ki == 0) {
					C_[local + i][res] = scalar_product_ij/ C_[local + i][0];
				}
				else {
					for (int q = 1; q < i + 1; q++) {
						C_[local + i][res] -= C_[local + i - q][res + q] * C_[local + i - q][q];
					}
					C_[local + i][res] += scalar_product_ij;
					C_[local + i][res] /= C_[local + i][0];
					
				}

				// ���������� ������� A
				A_[local + i][res] = scalar_product_ij;
				// ���������� ������� A ��� ������, ����� �������� �
				Gauss_A[local + i][res] = scalar_product_ij;
				// ���������� ������� ������������ A, � ������ �����
				Eigen_A(ki * (N - 1) + i, ki * (N - 1) + j) = scalar_product_ij;
				Eigen_A(ki * (N - 1) + j, ki * (N - 1) + i) = scalar_product_ij;
			}	
		}
	}

	////////////////////////////////////
	//          ������� ����          //
	////////////////////////////////////
	
	//////////////////
	// Gauss        //
	//////////////////

	// ������ ���
	// ��� �������� ������� ��������
	for (int m = 0; m < M; m++) {
		for (int j = 1; j < N && m + j < M; j++) {
			Gauss_b[m + j] -= Gauss_b[m] * Gauss_A[m][j] / Gauss_A[m][0];
			for (int q = 0; q < N - j; q++) {
				Gauss_A[m + j][q] -= Gauss_A[m][j + q] * Gauss_A[m][j] / Gauss_A[m][0];
			}
		}
	}	
	// �������� ���
	// ������� �������� �����������������
	for (int m = M - 1; m >= 0; m--) {
		double tmp = 0;
		for (int j = 1; j < N && m + j < M; j++) {
			tmp += Gauss_x[m + j] * Gauss_A[m][j];
		}
		Gauss_x[m] = (Gauss_b[m] - tmp) / Gauss_A[m][0];
	}

	//////////////////
	// LU           //
	//////////////////

	// ������ ����������������� �������
	for (int m = 0; m < M; m++) {
		double tmp = 0;
		for (int j = 1; j < N && m - j >= 0; j++) {
			tmp += LU_y[m - j] * L_[m - j][j];
		}
		LU_y[m] = b_[m] - tmp;
	}
	// ������ ���������������� �������
	for (int m = M - 1; m >= 0; m--) {
		double tmp = 0;
		for (int j = 1; j < N && m + j < M; j++) {
			tmp += LU_x[m + j] * U_[m][j];
		}
		LU_x[m] = (LU_y[m] - tmp) / U_[m][0];
	}

	//////////////////
	// CHOLESKY    //
	//////////////////

	// ������ ����������������� �������
	for (int m = 0; m < M; m++) {
		double tmp = 0;
		for (int j = 1; j < N && m - j >= 0; j++) {
			tmp += Cholesky_y[m - j] * C_[m - j][j];
		}
		Cholesky_y[m] = (b_[m] - tmp) / C_[m][0];
	}

	// ������ ���������������� �������
	for (int m = M - 1; m >= 0; m--) {
		double tmp = 0;
		for (int j = 1; j < N && m + j < M; j++) {
			tmp += Cholesky_x[m + j] * C_[m][j];
		}
		Cholesky_x[m] = (Cholesky_y[m] - tmp) / C_[m][0];
	}

	////////////////////////////////////
	// ����� ����������� ��������� CG //
	////////////////////////////////////
	

	// �������������
	double CG_x[M], CG_r[M], CG_z[M], CG_Az[M], CG_Ax[M];
	double alpha;

	// x0 ��������� ������������ ������� - ������ �� ������
	for (int i = 0; i < M; i++) {
		CG_x[i] = 1;
	}


	// ��������� ������� r0 = b - Ax0

	// ��������-��������� ��������� CG_Ax = A_ * CG_x
	for (int ki = 0; ki < K; ki++) {
		for (int i = 0; i < N; i++) {
			if (i == 0 && ki != 0) {
				continue;
			}
			int local = ki * (N - 1);
			CG_Ax[local + i] = 0;
			for (int j = 0; j < N && local + i + j < M; j++) {
				CG_Ax[local + i] += A_[local + i][j] * CG_x[local + i + j];
			}
			for (int q = 1; q <= i; q++) {
				CG_Ax[local + i] += A_[local + i - q][q] * CG_x[local + i - q];
			}
		}
	}

	for (int i = 0; i < M; i++) {
		// r0 = b - Ax0
		CG_r[i] = b_[i] - CG_Ax[i];
		// z0 = r0
		CG_z[i] = CG_r[i];
	}

	// ����������� �������� �������
	double CG_eps = 1e-9;
	// ������� ����������
	double CG_res = 0;

	// ���������� inf ����� �������
	for (int i = 0; i < M; i++) {
		if (abs(CG_r[i]) > CG_res) {
			CG_res = abs(CG_r[i]);
		}
	}

	// ����� ������� ������� ����������� ����������
	while (CG_res > CG_eps) {

		//1. ����� alpha
		double r = 0;
		// ���������� (r(k-1),r(k-1))
		for (int i = 0; i < M; i++) {
			r += CG_r[i] * CG_r[i];
		}
		// ��������-��������� ���������
		// ���� �� ��������
		for (int ki = 0; ki < K; ki++) {
			for (int i = 0; i < N; i++) {
				if (i == 0 && ki != 0) {
					continue;
				}
				int local = ki * (N - 1);
				CG_Az[local + i] = 0;
				for (int j = 0; j < N && local + i + j < M; j++) {
					CG_Az[local + i] += A_[local + i][j] * CG_z[local + i + j];
				}	
				for (int q = 1; q <= i; q++) {
					CG_Az[local + i] += A_[local + i - q][q] * CG_z[local + i - q];
				}
			}
		}
		// ���������� (Az(k-1),z(k-1))
		double az = 0;
		for (int i = 0; i < M; i++) {
			az += CG_Az[i] * CG_z[i];
		}
		// alpha = (r(k-1),r(k-1))/(Az(k-1),z(k-1))
		alpha = r/az;

		//2. ��������� ������� x(k) = x(k-1) + alpha * z(k-1)
		for (int i = 0; i < M; i++) {
			CG_x[i] +=alpha * CG_z[i];
		}

		//3. ��������� ������� r(k) = r(k-1) - alpha * A*z(k-1)
		for (int i = 0; i < M; i++) {
			CG_r[i] -= alpha * CG_Az[i];
		}

		//4. ���������� beta
		
		// ���������� (r(k),r(k))
		double rr = 0;
		for (int i = 0; i < M; i++) {
			rr += CG_r[i] * CG_r[i];
		}
		// beta = (r(k),r(k)) / (r(k-1),r(k-1))
		double beta = rr / r;

		//5. ��������� z
		
		// z(k) = r(k) + beta * z(k-1)
		for (int i = 0; i < M; i++) {
			CG_z[i] = CG_r[i] + beta * CG_z[i];
		}


		// ���������� inf ����� �������
		CG_res = 0;
		for (int i = 0; i < M; i++) {
			if (abs(CG_r[i]) > CG_res) {
				CG_res = abs(CG_r[i]);
			}
		}
	}
	 
	
	///////////////////////////////////////////////////
	// ����� ���������������� ������� ���������� SOR //
	///////////////////////////////////////////////////
	
	// ������� �� ������� ��������
	double SOR_x[M];
	// ������� �� ���������� ��������
	double SOR_xprev[M];


	double w = 1.5;

	// x0 ��������� ������������ ������� - ������ �� ������
	for (int i = 0; i < M; i++) {
		SOR_x[i] = 1;
		SOR_xprev[i] = SOR_x[i];
	}

	// ���������� �������� ��� ������
	int SOR_N = 10;
	// ���������� ��������, �� ������� 
	// ������� ����������� ������
	int SOR_cur = 0;

	
	// |z(k)-z(k-1)| < eps * |z[k]| + delta
	double SOR_delta = 1e-20;
	double SOR_eps = 1e-8;  
	
	// ||z(k)-z(k-1)|| - L2 
	double SOR_res = 0;
	// ||z(k)|| - L2 
	double SOR_norm_cur = 0;


	while (SOR_cur < SOR_N) {

		// ���������� ������� �� �����������
		for (int ki = 0; ki < K; ki++) {
			for (int i = 0; i < N; i++) {
				double tmpprev = 0, tmpcur = 0;
				int local = ki * (N - 1);
				for (int j = 1; j < N; j++) {
					tmpprev += A_[local + i][j] * SOR_xprev[local + i + j];
				}
				for (int q = 1; q <= i; q++) {
					tmpcur += A_[local + i - q][q] * SOR_x[local + i - q];
				}
				SOR_x[local + i] = (1-w)*SOR_xprev[local + i] + w * (b_[local + i] - tmpcur - tmpprev) / A_[local + i][0];
			}
		}
		
		// ���������� L2 ����� z(k)-z(k-1)
		SOR_res = 0;
		for (int m = 0; m < M; m++) {
			SOR_res += (SOR_x[m] - SOR_xprev[m]) * (SOR_x[m] - SOR_xprev[m]);
		}
		SOR_res = sqrt(SOR_res);

		// ���������� L2 ����� z(k)
		SOR_norm_cur = 0;
		for (int m = 0; m < M; m++) {	
			SOR_norm_cur = SOR_x[m] * SOR_x[m];
		}
		SOR_norm_cur = sqrt(SOR_norm_cur);

		// ���������� ����������� ������� �����������
		for (int m = 0; m < M; m++) {
			SOR_xprev[m] = SOR_x[m];
		}

		// ���� �������� �����������
		// ����������� �������
		if (SOR_res < SOR_eps * SOR_norm_cur + SOR_delta) {
			SOR_cur++;
		}
		else {
			// ����� �������� �������
			SOR_cur = 0;
		}
	}

	// ������������ �������
	VectorXd Eigen_x = Eigen_A.colPivHouseholderQr().solve(Eigen_b);











	//////////////////////////////////////
	// �������� ������� � ������ � ���� //
	//////////////////////////////////////

	// ��������� � ������������ ��������
	// *_Eigen_# 
	// * - ����� �������
	// # - ��� �����

	double Gauss_Eigen_1 = 0, Gauss_Eigen_2 = 0, Gauss_Eigen_inf = 0;
	double LU_Eigen_1 = 0, LU_Eigen_2 = 0, LU_Eigen_inf = 0;
	double Cholesky_Eigen_1 = 0, Cholesky_Eigen_2 = 0, Cholesky_Eigen_inf = 0;
	double CG_Eigen_1 = 0, CG_Eigen_2 = 0, CG_Eigen_inf = 0;
	double SOR_Eigen_1 = 0, SOR_Eigen_2 = 0, SOR_Eigen_inf = 0;

	for (int m = 0; m < M; m++) {
		Gauss_Eigen_1 += abs(Gauss_x[m] - Eigen_x(m));
		LU_Eigen_1 += abs(LU_x[m] - Eigen_x(m));
		Cholesky_Eigen_1 += abs(Cholesky_x[m] - Eigen_x(m));
		CG_Eigen_1 += abs(CG_x[m] - Eigen_x(m));
		SOR_Eigen_1 += abs(SOR_x[m] - Eigen_x(m));


		Gauss_Eigen_2 += (Gauss_x[m] - Eigen_x(m)) * (Gauss_x[m] - Eigen_x(m));
		LU_Eigen_2 += (LU_x[m] - Eigen_x(m)) * (LU_x[m] - Eigen_x(m));
		Cholesky_Eigen_2 += (Cholesky_x[m] - Eigen_x(m)) * (Cholesky_x[m] - Eigen_x(m));
		CG_Eigen_2 += (CG_x[m] - Eigen_x(m)) * (CG_x[m] - Eigen_x(m));
		SOR_Eigen_2 += (SOR_x[m] - Eigen_x(m)) * (SOR_x[m] - Eigen_x(m));

		if (abs(Gauss_x[m] - Eigen_x(m)) > Gauss_Eigen_inf) {
			Gauss_Eigen_inf = abs(Gauss_x[m] - Eigen_x(m));
		}
		if (abs(LU_x[m] - Eigen_x(m)) > LU_Eigen_inf) {
			LU_Eigen_inf = abs(LU_x[m] - Eigen_x(m));
		}
		if (abs(Cholesky_x[m] - Eigen_x(m)) > Cholesky_Eigen_inf) {
			Cholesky_Eigen_inf = abs(Cholesky_x[m] - Eigen_x(m));
		}
		if (abs(CG_x[m] - Eigen_x(m)) > CG_Eigen_inf) {
			CG_Eigen_inf = abs(CG_x[m] - Eigen_x(m));
		}
		if (abs(SOR_x[m] - Eigen_x(m)) > SOR_Eigen_inf) {
			SOR_Eigen_inf = abs(SOR_x[m] - Eigen_x(m));
		}
	}
	
	// ���������� r = b - Ax - ����������� �������
	// *_res_# 
	// * - ����� �������
	// # - ��� �����
	double Gauss_res_1 = 0, Gauss_res_2 = 0, Gauss_res_inf = 0;
	double LU_res_1 = 0, LU_res_2 = 0, LU_res_inf = 0;
	double Cholesky_res_1 = 0, Cholesky_res_2 = 0, Cholesky_res_inf = 0;
	double CG_res_1 = 0, CG_res_2 = 0, CG_res_inf = 0;
	double SOR_res_1 = 0, SOR_res_2 = 0, SOR_res_inf = 0;
	double Eigen_res_1 = 0, Eigen_res_2 = 0, Eigen_res_inf = 0;


	// ������� ������������
	double Gauss_res_vector[M];
	double LU_res_vector[M];
	double Cholesky_res_vector[M];
	double CG_res_vector[M];
	double SOR_res_vector[M];
	double Eigen_res_vector[M];

	// ��������� �� ������
	for (int i = 0; i < M; i++) {
		Gauss_res_vector[i] = 0;
		LU_res_vector[i] = 0;
		Cholesky_res_vector[i] = 0;
		CG_res_vector[i] = 0;
		SOR_res_vector[i] = 0;
		Eigen_res_vector[i] = 0;
	}
	// ��������-��������� ��������� � ����������
	for (int ki = 0; ki < K; ki++) {
		for (int i = 0; i < N; i++) {
			if (i == 0 && ki != 0) {
				continue;
			}
			int local = ki * (N - 1);
			// ���������
			for (int j = 0; j < N && local + i + j < M; j++) {
				Gauss_res_vector[local + i] += A_[local + i][j] * Gauss_x[local + i + j];
				LU_res_vector[local + i] += A_[local + i][j] * LU_x[local + i + j];
				Cholesky_res_vector[local + i] += A_[local + i][j] * Cholesky_x[local + i + j];
				CG_res_vector[local + i] += A_[local + i][j] * CG_x[local + i + j];
				SOR_res_vector[local + i] += A_[local + i][j] * SOR_x[local + i + j];
				Eigen_res_vector[local + i] += A_[local + i][j] * Eigen_x(local + i + j);
			}
			for (int q = 1; q <= i; q++) {
				Gauss_res_vector[local + i] += A_[local + i - q][q] * Gauss_x[local + i - q];
				LU_res_vector[local + i] += A_[local + i - q][q] * LU_x[local + i - q];
				Cholesky_res_vector[local + i] += A_[local + i - q][q] * Cholesky_x[local + i - q];
				CG_res_vector[local + i] += A_[local + i - q][q] * CG_x[local + i - q];
				SOR_res_vector[local + i] += A_[local + i - q][q] * SOR_x[local + i - q];
				Eigen_res_vector[local + i] += A_[local + i - q][q] * Eigen_x(local + i - q);
			}
			// ���������
			Gauss_res_vector[local + i] -= b_[local + i];
			LU_res_vector[local + i] -= b_[local + i];
			Cholesky_res_vector[local + i] -= b_[local + i];
			CG_res_vector[local + i] -= b_[local + i];
			SOR_res_vector[local + i] -= b_[local + i];
			Eigen_res_vector[local + i] -= b_[local + i];
		}
	}

	// ������� ���� ������������
	for (int m = 0; m < M; m++) {
		// L1
		Gauss_res_1 += abs(Gauss_res_vector[m]);
		LU_res_1 += abs(LU_res_vector[m]);
		Cholesky_res_1 += abs(Cholesky_res_vector[m]);
		CG_res_1 += abs(CG_res_vector[m]);
		SOR_res_1 += abs(SOR_res_vector[m]);
		Eigen_res_1 += abs(Eigen_res_vector[m]);

		// L2
		Gauss_res_2 += (Gauss_res_vector[m]) * (Gauss_res_vector[m]);
		LU_res_2 += (LU_res_vector[m]) * (LU_res_vector[m]);
		Cholesky_res_2 += (Cholesky_res_vector[m]) * (Cholesky_res_vector[m]);
		CG_res_2 += (CG_res_vector[m]) * (CG_res_vector[m]);
		SOR_res_2 += (SOR_res_vector[m]) * (SOR_res_vector[m]);
		Eigen_res_2 += (Eigen_res_vector[m]) * (Eigen_res_vector[m]);

		// Linf
		if (abs(Gauss_res_vector[m]) > Gauss_res_inf) {
			Gauss_res_inf = abs(Gauss_res_vector[m]);
		}
		if (abs(LU_res_vector[m]) > LU_res_inf) {
			LU_res_inf = abs(LU_res_vector[m]);
		}
		if (abs(Cholesky_res_vector[m]) > Cholesky_res_inf) {
			Cholesky_res_inf = abs(Cholesky_res_vector[m]);
		}
		if (abs(CG_res_vector[m]) > CG_res_inf) {
			CG_res_inf = abs(CG_res_vector[m]);
		}
		if (abs(SOR_res_vector[m]) > SOR_res_inf) {
			SOR_res_inf = abs(SOR_res_vector[m]);
		}
		if (abs(Eigen_res_vector[m]) > Eigen_res_inf) {
			Eigen_res_inf = abs(Eigen_res_vector[m]);
		}
	}


	// ������ � ���� ��������� � ����������� ��������
	FILE* res;
	res = fopen("res.txt", "w");
		
	fprintf(res, "|----|-----------------------------------------------------------------------------------------|\n");
	fprintf(res, "|norm| Gauss        | LU           | Cholesky     | CG           | SOR          | Eigen        |\n");
	fprintf(res, "|----+-----------------------------------------------------------------------------------------|\n");
	fprintf(res, "|    | Compare with Eigen                                                                      |\n");
	fprintf(res, "|----+-----------------------------------------------------------------------------------------|\n");
	fprintf(res, "| L1 | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |              |\n", Gauss_Eigen_1, LU_Eigen_1, Cholesky_Eigen_1, CG_Eigen_1, SOR_Eigen_1);
	fprintf(res, "| L2 | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |              |\n", Gauss_Eigen_2, LU_Eigen_2, Cholesky_Eigen_2, CG_Eigen_2, SOR_Eigen_2);
	fprintf(res, "|Linf| %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |              |\n", Gauss_Eigen_inf, LU_Eigen_inf, Cholesky_Eigen_inf, CG_Eigen_inf, SOR_Eigen_inf);
	fprintf(res, "|----+-----------------------------------------------------------------------------------------|\n");
	fprintf(res, "|    | Compute residual norm                                                                   |\n");
	fprintf(res, "|----+-----------------------------------------------------------------------------------------|\n");
	fprintf(res, "| L1 | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |\n", Gauss_res_1, LU_res_1, Cholesky_res_1, CG_res_1, SOR_res_1, Eigen_res_1);
	fprintf(res, "| L2 | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |\n", Gauss_res_2, LU_res_2, Cholesky_res_2, CG_res_2, SOR_res_2, Eigen_res_2);
	fprintf(res, "|Linf| %9.6e | %9.6e | %9.6e | %9.6e | %9.6e | %9.6e |\n", Gauss_res_inf, LU_res_inf, Cholesky_res_inf, CG_res_inf, SOR_res_inf, Eigen_res_inf);
	fprintf(res, "|----+-----------------------------------------------------------------------------------------|\n");
	fclose(res);



	for (int iviz = 0, ki = 0; iviz < Mviz; iviz++) // ������� ������������� � ������ ��� �����������
	{
		if (X[iviz] > mesh[(ki + 1) * (N - 1)]) // ������ �������� �� ���� ������, ������� ����� ������� �� ������� �������� ���������
		{
			ki++;
		}
		Approx[iviz] = 0; // �������������� ����� �����
		for (int i = 0; i < N; i++)
		{
			double gviz = 1 / D[i]; // ������� �������� �������� ������� � �����
			for (int j = 0; j < N; j++)
			{
				if (i != j)
				{
					gviz *= (X[iviz] - mesh[ki * (N - 1) + j]);
				}
			}
			Approx[iviz] += Eigen_x(ki * (N - 1) + i) * gviz; // � �������� �� ������ ���������
		}

	}




	FILE* ParamsFile; // ���� ���������� - � ��� a,b � ��� �����
	ParamsFile = fopen("Params.txt", "w");
	fprintf(ParamsFile, "%f, %f", a, b);
	fclose(ParamsFile);

	FILE* meshFile, * FmeshFile; // ����� ����� ������������ � �������� ������� � ���� ������ - ��� ��������� �������
	meshFile = fopen("mesh.txt", "w");
	FmeshFile = fopen("Fmesh.txt", "w");
	for (int i = 0; i < M - 1; i++)
	{
		fprintf(meshFile, "%f, ", mesh[i]);
		fprintf(FmeshFile, "%f, ", f(mesh[i]));
	}
	fprintf(meshFile, "%f\n", mesh[M - 1]);
	fprintf(FmeshFile, "%f\n", f(mesh[M - 1]));
	fclose(meshFile);
	fclose(FmeshFile);

	FILE* XFile, * ApproxFile, * FFile; // ����� ����� ���������� ��������
	XFile = fopen("X.txt", "w"); // ����� X
	ApproxFile = fopen("Approx.txt", "w"); // �������� �������� �������� � ���� ������
	FFile = fopen("F.txt", "w"); // �������� ������� � ���� ������
	for (int i = 0; i < Mviz - 1; i++)
	{
		fprintf(XFile, "%f, ", X[i]);
		fprintf(FFile, "%f, ", f(X[i]));
		fprintf(ApproxFile, "%f, ", Approx[i]);
	}
	fprintf(XFile, "%f\n", X[Mviz - 1]);
	fprintf(FFile, "%f\n", f(X[Mviz - 1]));
	fprintf(ApproxFile, "%f\n", Approx[Mviz - 1]);
	fclose(XFile);
	fclose(ApproxFile);
	fclose(FFile);


	FILE* KFile, * KFFile;
	KFile = fopen("K.txt", "w"); // �������� ������� � ���� ������
	KFFile = fopen("KF.txt", "w"); // �������� ������� � ���� ������
	for (int ki = 0; ki < K - 1; ki++)
	{
		fprintf(KFile, "%f, ", mesh[ki * (N - 1)]);
		fprintf(KFFile, "%f, ", f(mesh[ki * (N - 1)]));
	}
	fprintf(KFile, "%f\n", mesh[M - 1]);
	fprintf(KFFile, "%f\n", f(mesh[M - 1]));
	fclose(KFile);
	fclose(KFFile);

	py();
	std::system("python plot.py"); // ��� ������� �������� ��������� ������ � �������� ����������� ����� ������
	std::system("del /s /q Params.txt X.txt F.txt Approx.txt mesh.txt Fmesh.txt K.txt KF.txt");

	return 0;
}