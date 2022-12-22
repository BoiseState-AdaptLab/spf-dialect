#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  int X = 3;
  int T = 1;
#undef s0
#undef s_0
#undef s1
#undef s_1
#define s_0(t, x) printf("s0(%d,%d)\n", t, x)
#define s0(a, __x1, x1, __x3) s_0(a, x1 + 1);
#define s_1(t, x) printf("s1(%d,%d)\n", t, x)
#define s1(t0, __x1, t2p, __x3) s_1(t0, t2p);

  int t1 = 0;
  int t2 = 0;
  int t3 = 0;
  int t4 = 1;

  if (X >= 1) {
    for (t1 = 1; t1 <= T; t1++) {
      s0(t1, 0, 0, 0);
      for (t3 = 1; t3 <= X - 1; t3++) {
        s0(t1, 0, t3, 0);
        s1(t1, 0, t3, 1);
      }
      s1(t1, 0, X, 1);
    }
  }

#undef s0
#undef s_0
#undef s1
#undef s_1
}
