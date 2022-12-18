#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  {
    int A[10];
    int B[10];

    int A_STORE[6][10];
    int B_STORE[6][10];

    // set up initial conditions
    for (int i = 0; i < 10; ++i) {
      B[i] = 0;
      A[i] = 0;
    }
    A[9] = 100;
    B[9] = 100;
    /// store off intermediate results
    for (int i = 0; i < 10; ++i)
      A_STORE[0][i] = A[i];
    for (int i = 0; i < 10; ++i)
      B_STORE[0][i] = B[i];

    // run jacobian
#define A_m(x) A[x]
#define B_m(x) B[x]
int t1;
int t2;
int t3;


#undef s0
#undef s_0
#undef s1
#undef s_1
#define s_0(t, x)   A_m(x) = (B_m(x-1) + B_m(x) + B_m(x+1))/3
#define s0(t, __x1, x)   s_0(t, x);
#define s_1(t, x)   B_m(x) = (A_m(x-1) + A_m(x) + A_m(x+1))/3
#define s1(t, __x1, x)   s_1(t, x);


t1 = 0;
t2 = 1;
t3 = 0;

for(t1 = 1; t1 <= 5; t1++) {
  for(t3 = 1; t3 <= 8; t3++) {
    A[t3] = (B[t3 - 1] + B[t3] + B[t3 + 1]) / 3;
    ;
  }
  for(t3 = 1; t3 <= 8; t3++) {
    s1(t1,1,t3);
  }

  // store off intermediate results
  for (int i = 0; i < 10; ++i)
    A_STORE[t1][i] = A[i];
  for (int i = 0; i < 10; ++i)
    B_STORE[t1][i] = B[i];
}

#undef s0
#undef s_0
#undef s1
#undef s_1

    // print output
    for (int t = 0; t < 6; ++t) {
      { // print A
        printf("time step %d: A[%d] : [", 2 * t - 1, t);
        bool first = true;
        for (int i = 0; i < 10; ++i) {
          if (first) {
            first = false;
          } else {
            printf(",");
          }
          printf("%d", A_STORE[t][i]);
        }
        printf("]\n");
      }
      { // print B
        printf("time step %d: B[%d] : [", 2 * t, t);
        bool first = true;
        for (int i = 0; i < 10; ++i) {
          if (first) {
            first = false;
          } else {
            printf(",");
          }
          printf("%d", B_STORE[t][i]);
        }
        printf("]\n");
      }
    }
  }
  printf("=================================================\n");
  {
    int A[10];
    int B[10];

    int A_STORE[6][10];
    int B_STORE[6][10];

    // set up initial conditions
    for (int i = 0; i < 10; ++i) {
      B[i] = 0;
      A[i] = 0;
    }
    A[9] = 100;
    B[9] = 100;
    /// store off intermediate results
    for (int i = 0; i < 10; ++i)
      A_STORE[0][i] = A[i];
    for (int i = 0; i < 10; ++i)
      B_STORE[0][i] = B[i];

    // run transformed jacobian
#define A_m(x) A[x]
#define B_m(x) B[x]
int t1;
int t2;
int t3;
int t4;


#undef s0
#undef s_0
#undef s1
#undef s_1
#define s_0(t, x)   A_m(x) = (B_m(x-1) + B_m(x) + B_m(x+1))/3
#define s0(a, __x1, x1, __x3)   s_0(a, x1 + 1);
#define s_1(t, x)   B_m(x) = (A_m(x-1) + A_m(x) + A_m(x+1))/3
#define s1(t0, __x1, t2p, __x3)   s_1(t0, t2p);


t1 = 0;
t2 = 0;
t3 = 0;
t4 = 1;

for(t1 = 1; t1 <= 5; t1++) {
  s0(t1,0,0,0);
  for(t3 = 1; t3 <= 7; t3++) {
    s0(t1,0,t3,0);
    s1(t1,0,t3,1);
  }
  s1(t1,0,8,1);

  /// store off intermediate results
  for (int i = 0; i < 10; ++i)
    A_STORE[t1][i] = A[i];
  for (int i = 0; i < 10; ++i)
    B_STORE[t1][i] = B[i];
}

#undef s0
#undef s_0
#undef s1
#undef s_1

    // print output
    for (int t = 0; t < 6; ++t) {
      { // print A
        printf("time step %d: A[%d] : [", 2 * t - 1, t);
        bool first = true;
        for (int i = 0; i < 10; ++i) {
          if (first) {
            first = false;
          } else {
            printf(",");
          }
          printf("%d", A_STORE[t][i]);
        }
        printf("]\n");
      }
      { // print B
        printf("time step %d: B[%d] : [", 2 * t, t);
        bool first = true;
        for (int i = 0; i < 10; ++i) {
          if (first) {
            first = false;
          } else {
            printf(",");
          }
          printf("%d", B_STORE[t][i]);
        }
        printf("]\n");
      }
    }
  }
}
