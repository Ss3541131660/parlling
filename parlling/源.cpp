#include <iostream>
#include<ctime>
#include<Windows.h>
#include <immintrin.h>
using namespace std;
void test() {
    int n = 128;//n为数据规模，取64、128、256、512、1024五个值
    float** A = new float* [n];
    float** m = new float* [n];
    srand(time(0));
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        m[i] = new float[n];
    }
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;//对角线为1.0
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10;
        }

    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] += A[k][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = A[i][j];//两个完全相同的矩阵，一个用于串行算法，一个用于SSE
        }
    }
    //测试用例的生成

    long long head1, tail1, freq1; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head1);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
    cout << "数据规模为n=" << n << endl;
    cout << "串行算法测试时间为 " << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;






    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < n; k++) {
        float vt = A[k][k];//vt ← dupTo4Float(A[k,k]);
        __m128 va;//float型寄存器
        for (int j = k + 1; j + 4 <= n; j += 4) {
            va = _mm_load_ps(&A[k][j]);//va ← load4FloatFrom(&A[k,j]) ; 
            va = _mm_div_ps(va, _mm_set1_ps(vt)); //va ← va/vt ; // 这里是向量对位相除
            _mm_store_ps(&A[k][j], va);//store4FloatTo(&A[k, j], va);
        }
        //4路向量化
        //for (int j = k + 1; j < n; j++) {
        //    A[k][j] = A[k][j] / A[k][k]; //A[k, j] = A[k, j] / A[k, k];
        //}
        A[k][k] = 1.0;//A[k,k] ← 1.0;

        __m128 vaik, vakj, vaij, vx;//float型寄存器
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set1_ps(A[i][k]);//vaik ← dupToVector4(A[i,k]);
            for (int j = k + 1; j + 4 <= n; j += 4) {
                vakj = _mm_load_ps(&A[k][j]);//vakj ← load4FloatFrom(&A[k,j]);
                vaij = _mm_load_ps(&A[i][j]);//vaij ← load4FloatFrom(&A[i,j]);
                vx = _mm_mul_ps(vakj, vaik);//vx ← vakj*vaik;
                vaij = _mm_sub_ps(vaij, vx);//vaij ← vaij-vx;
                _mm_store_ps(&A[i][j], vaij);//store4FloatTo(&A[i,j],vaij);
            }
            //for (int j = k + 1; j < n; j++) {
            //    A[i][j] =A[i][j]- A[k][j] * A[i][k];//A[i,j] ← A[i,j] -A[k,j]*A[i,k];
            //}
            A[i][k] = 0;//A[i,k] ← 0;
        }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE并行测试时间为 " << (tail - head) * 1000.0 / freq << "ms" << endl;
}
int main() {
    test();
   
    test();

    test();

    test();

    test();
    return 0;
}
