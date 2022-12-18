#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  int ub_T = 10;
  int ub_x = 8;
  int lb_x = 1;
  int i;

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
    for (i = 0; i < 10; ++i)
      A_STORE[0][i] = A[i];
    for (i = 0; i < 10; ++i)
      B_STORE[0][i] = B[i];

    // run jacobian
    for (int t = 1; t <= ub_T / 2; ++t) {
      for (i = lb_x; i <= ub_x; ++i)
        A[i] = (B[i - 1] + B[i] + B[i + 1]) / 3;

      for (i = lb_x; i <= ub_x; ++i)
        B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3;

      /// store off intermediate results
      for (i = 0; i < 10; ++i)
        A_STORE[t][i] = A[i];
      for (i = 0; i < 10; ++i)
        B_STORE[t][i] = B[i];
    }

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
    for (i = 0; i < 10; ++i)
      A_STORE[0][i] = A[i];
    for (i = 0; i < 10; ++i)
      B_STORE[0][i] = B[i];

    // run jacobian
    for (int t = 1; t <= ub_T / 2; ++t) {
      int i = lb_x;
      A[i] = (B[i - 1] + B[i] + B[i + 1]) / 3;
      for (i = lb_x; i <= ub_x - 1; ++i) {
        A[i + 1] = (B[i] + B[i + 1] + B[i + 2]) / 3;
        B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3;
      }
      B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3;

      /// store off intermediate results
      for (i = 0; i < 10; ++i)
        A_STORE[t][i] = A[i];
      for (i = 0; i < 10; ++i)
        B_STORE[t][i] = B[i];
    }

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
