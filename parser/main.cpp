#include "../lib/StandaloneTransforms/Parser.h"
#include <vector>

int main() {
  std::vector<const char *> cases{"for(t1 = 0; t1 <= I-1; t1++) {\n"
                                  "  for(t2 = 0; t2 <= K-1; t2++) {\n"
                                  "    for(t3 = 0; t3 <= L-1; t3++) {\n"
                                  "      for(t4 = 0; t4 <= J-1; t4++) {\n"
                                  "        s0(t1,t2,t3,t4);\n"
                                  "      }\n"
                                  "    }\n"
                                  "  }\n"
                                  "}\n",
                                  "for(t1 = 0; t1 <= J-1; t1++) {\n"
                                  "  for(t2 = 0; t2 <= NNZ-1; t2++) {\n"
                                  "    t3=UFi_0(t1,t2);\n"
                                  "    t4=UFk_1(t1,t2);\n"
                                  "    t5=UFl_2(t1,t2);\n"
                                  "    s0(t1,t2,t3,t4,t5);\n"
                                  "  }\n"
                                  "}\n",
                                  "if (X >= 1) {\n"
                                  "  for(t1 = 1; t1 <= T; t1++) {\n"
                                  "    s0(t1,0,0,0);\n"
                                  "    for(t3 = 1; t3 <= X-1; t3++) {\n"
                                  "      s0(t1,0,t3,0);\n"
                                  "      s1(t1,0,t3,1);\n"
                                  "    }\n"
                                  "    s1(t1,0,X,1);\n"
                                  "  }\n"
                                  "}\n"};

  bool first = true;
  for (auto c : cases) {
    if (first) {
      first = false;
    } else {
      printf("==========================\n");
    }

    printf("%s", c);
    auto lexer = mlir::standalone::parser::Lexer(std::move(c));
    auto parser = mlir::standalone::parser::Parser(lexer);
    auto ast = parser.parseProgram();
    if (ast) {
      std::cout << ast->dump();
    }
  }
}