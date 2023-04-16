#include <iostream>
#include<ctime>
#include<Windows.h>
#include <immintrin.h>
#include <intrin.h>
#include<iomanip>
using namespace std;
double time1 = 0, time2 = 0, time3 = 0, time4 = 0;
void test() {
    int n = 512;//nΪ���ݹ�ģ��ȡ64��128��256��512��1024���ֵ
    float** m = new float* [n];//���о���
    float** A = new float* [n];//����ѭ��SSE����
    float** A1 = new float* [n];//����ѭ��SSE����
    float** A2 = new float* [n];//SSE�Ż�����
    srand(time(0));
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        m[i] = new float[n];
        A1[i] = new float[n];
        A2[i] = new float[n];
    }
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;//�Խ���Ϊ1.0
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
            m[i][j] = A[i][j];//������ȫ��ͬ�ľ���һ�����ڴ����㷨��һ�����ڶ���ѭ����SSE������
            A1[i][j] = A[i][j];//������ȫ��ͬ�ľ���һ�����ڶ���ѭ����SSE��������һ����������ѭ����SSE������
            A2[i][j] = A[i][j];//A2Ϊ��ȫSSE������
        }
    }
    //��������������


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
    cout << "���ݹ�ģΪn=" << n << endl;
    time1 += (tail1 - head1) * 1000.0 / freq1;
    cout << "�����㷨����ʱ��Ϊ " << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;


/////////////////////////////////////////////////////////////////////����ѭ��SSE������//////////////////////////////////////////////////////////////////////////////////////////
    long long head2, tail2, freq2; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head2);

    for (int k = 0; k < n; k++) {
        float vt = A1[k][k];//vt �� dupTo4Float(A[k,k]);
        __m128 va;//float�ͼĴ���
        for (int j = k + 1; j + 4 < n; j += 4) {
            va = _mm_load_ps(&A1[k][j]);//va �� load4FloatFrom(&A[k,j]) ; 
            va = _mm_div_ps(va, _mm_set1_ps(vt)); //va �� va/vt ; // ������������λ���
            _mm_store_ps(&A1[k][j], va);//store4FloatTo(&A[k, j], va);
        }
        //4·������
        A1[k][k] = 1.0;//A[k,k] �� 1.0;

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A1[i][j] = A1[i][j] - A1[k][j] * A1[i][k];
            }
            A1[i][k] = 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
    cout << "SSE����ѭ������������ʱ��Ϊ " << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;
    time2 += (tail2 - head2) * 1000.0 / freq2;

    /////////////////////////////////////////////////////////////////////����ѭ��SSE������//////////////////////////////////////////////////////////////////////////////////////////

    long long head3, tail3, freq3; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head3);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A2[k][j] = A2[k][j] / A2[k][k];
        }
        A2[k][k] = 1.0;//A[k,k] �� 1.0;

        __m128 vaik, vakj, vaij, vx;//float�ͼĴ���
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set1_ps(A2[i][k]);//vaik �� dupToVector4(A[i,k]);
            for (int j = k + 1; j + 4 < n; j += 4) {
                vakj = _mm_load_ps(&A2[k][j]);//vakj �� load4FloatFrom(&A[k,j]);
                vaij = _mm_load_ps(&A2[i][j]);//vaij �� load4FloatFrom(&A[i,j]);
                vx = _mm_mul_ps(vakj, vaik);//vx �� vakj*vaik;
                vaij = _mm_sub_ps(vaij, vx);//vaij �� vaij-vx;
                _mm_store_ps(&A2[i][j], vaij);//store4FloatTo(&A[i,j],vaij);
            }
            A2[i][k] = 0;//A[i,k] �� 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
    cout << "SSE����ѭ������������ʱ��Ϊ " << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
    time3 += (tail3 - head3) * 1000.0 / freq3;

    /////////////////////////////////////////////////////////////////////��ȫSE������//////////////////////////////////////////////////////////////////////////////////////////

    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < n; k++) {
        float vt = A[k][k];//vt �� dupTo4Float(A[k,k]);
        __m128 va;//float�ͼĴ���
        for (int j = k + 1; j + 4 <n; j += 4) {
            va = _mm_load_ps(&A[k][j]);//va �� load4FloatFrom(&A[k,j]) ; 
            va = _mm_div_ps(va, _mm_set1_ps(vt)); //va �� va/vt ; // ������������λ���
            _mm_store_ps(&A[k][j], va);//store4FloatTo(&A[k, j], va);
        }
        //4·������
        A[k][k] = 1.0;//A[k,k] �� 1.0;

        __m128 vaik, vakj, vaij, vx;//float�ͼĴ���
        for (int i = k + 1; i < n; i++) {
            vaik = _mm_set1_ps(A[i][k]);//vaik �� dupToVector4(A[i,k]);
            for (int j = k + 1; j + 4 < n; j += 4) {
                vakj = _mm_load_ps(&A[k][j]);//vakj �� load4FloatFrom(&A[k,j]);
                vaij = _mm_load_ps(&A[i][j]);//vaij �� load4FloatFrom(&A[i,j]);
                vx = _mm_mul_ps(vakj, vaik);//vx �� vakj*vaik;
                vaij = _mm_sub_ps(vaij, vx);//vaij �� vaij-vx;
                _mm_store_ps(&A[i][j], vaij);//store4FloatTo(&A[i,j],vaij);
            }
            A[i][k] = 0;//A[i,k] �� 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SSE��ȫ����������ʱ��Ϊ " << (tail - head) * 1000.0 / freq << "ms" << endl;
    time4 += (tail - head) * 1000.0 / freq;

    cout << endl;
    cout << endl;

}
//����ΪSSE������
void test1() {
    int n = 512;//nΪ���ݹ�ģ��ȡ64��128��256��512��1024���ֵ
    float** m = new float* [n];//���о���
    float** A = new float* [n];//����ѭ��SSE����
    float** A1 = new float* [n];//����ѭ��SSE����
    float** A2 = new float* [n];//SSE�Ż�����
    srand(time(0));
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        m[i] = new float[n];
        A1[i] = new float[n];
        A2[i] = new float[n];
    }
    for (int i = 0; i < n; i++) {
        A[i][i] = 1.0;//�Խ���Ϊ1.0
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
            m[i][j] = A[i][j];//������ȫ��ͬ�ľ���һ�����ڴ����㷨��һ�����ڶ���ѭ����AVX������
            A1[i][j] = A[i][j];//������ȫ��ͬ�ľ���һ�����ڶ���ѭ����SSE��������һ����������ѭ����AVX������
            A2[i][j] = A[i][j];//A2Ϊ��ȫAVX������
        }
    }
    //��������������

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
    cout << "���ݹ�ģΪn=" << n << endl;
    time1 += (tail1 - head1) * 1000.0 / freq1;
    cout << "�����㷨����ʱ��Ϊ " << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;


    /////////////////////////////////////////////////////////////////////����ѭ��SSE������//////////////////////////////////////////////////////////////////////////////////////////
   
    long long head2, tail2, freq2; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head2);

    for (int k = 0; k < n; k++) {
        float vt = A1[k][k];
        __m256 va;
        for (int j = k + 1; j + 8 <= n; j += 8) {
             va = _mm256_load_ps(&A1[k][j]);
             va = _mm256_div_ps(va, _mm256_set1_ps(vt));
             _mm256_store_ps(&A1[k][j], va);
        }
        A1[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A1[i][j] = A1[i][j] - A1[k][j] * A1[i][k];
            }
            A1[i][k] = 0;
        }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
    cout << "AVX����ѭ������������ʱ��Ϊ " << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;
    time2 += (tail2 - head2) * 1000.0 / freq2;

    /////////////////////////////////////////////////////////////////////����ѭ��SSE������//////////////////////////////////////////////////////////////////////////////////////////

    long long head3, tail3, freq3; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head3);

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A2[k][j] = A2[k][j] / A2[k][k];
        }
        A2[k][k] = 1.0;

        __m256 vaik, vakj, vaij, vx;
         for (int i = k + 1; i < n; i++) {
             vaik = _mm256_set1_ps(A2[i][k]);
             for (int j = k + 1; j + 8 <= n; j += 8) {
                 vakj = _mm256_load_ps(&A2[k][j]);
                 vaij = _mm256_load_ps(&A2[i][j]);
                 vx = _mm256_mul_ps(vakj, vaik);
                 vaij = _mm256_sub_ps(vaij, vx);
                 _mm256_store_ps(&A2[i][j], vaij);
             }
             A2[i][k] = 0;
         }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
    cout << "AVX����ѭ������������ʱ��Ϊ " << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
    time3 += (tail3 - head3) * 1000.0 / freq3;

    /////////////////////////////////////////////////////////////////////��ȫSE������//////////////////////////////////////////////////////////////////////////////////////////

    long long head, tail, freq; // timers
    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int k = 0; k < n; k++) {
        float vt = A[k][k];
        __m256 va;
        for (int j = k + 1; j + 8 <= n; j += 8) {
            va = _mm256_load_ps(&A[k][j]);
            va = _mm256_div_ps(va, _mm256_set1_ps(vt));
            _mm256_store_ps(&A[k][j], va);
        }
        A[k][k] = 1.0;

        __m256 vaik, vakj, vaij, vx;
        for (int i = k + 1; i < n; i++) {
            vaik = _mm256_set1_ps(A[i][k]);
            for (int j = k + 1; j + 8 <= n; j += 8) {
                vakj = _mm256_load_ps(&A[k][j]);
                vaij = _mm256_load_ps(&A[i][j]);
                vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_store_ps(&A[i][j], vaij);
            }
            A[i][k] = 0;
        }
    }


    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX��ȫ����������ʱ��Ϊ " << (tail - head) * 1000.0 / freq << "ms" << endl;
    time4 += (tail - head) * 1000.0 / freq;

    cout << endl;
    cout << endl;
    cout << endl;
}
int main() {
    
    for (int i = 1; i <= 20; i++) {
        test1();
   }

    //cout << "�����㷨ƽ������ʱ��Ϊ" << time1/20 << "ms" << endl;
    //cout << "SSE����ѭ��������ƽ������ʱ��Ϊ " << time2/20 << "ms" << endl;
    //cout << "SSE����ѭ��������ƽ������ʱ��Ϊ " << time3/20 << "ms" << endl;
    //cout << "SSE��ȫ������ƽ������ʱ��Ϊ " << time4/20<<"ms" << endl;



    cout << "�����㷨ƽ������ʱ��Ϊ" << time1 / 20 << "ms" << endl;
    cout << "AVX����ѭ��������ƽ������ʱ��Ϊ " << time2 / 20 << "ms" << endl;
    cout << "AVX����ѭ��������ƽ������ʱ��Ϊ " << time3 / 20 << "ms" << endl;
    cout << "AVX��ȫ������ƽ������ʱ��Ϊ " << time4 / 20 << "ms" << endl;
    return 0;
}
