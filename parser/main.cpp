#include "../lib/StandaloneTransforms/Parser.h"
#include "computation/Computation.h"
#include <iostream>

int main() {
  std::vector<Computation *> testCases;
  {
    Computation *jacobi = new Computation();
    jacobi->addDataSpace("A", "double*");
    jacobi->addDataSpace("B", "double*");
    jacobi->addStmt(
        new Stmt(/*stmtSourceCode*/ "A(x) = (B(x-1) + B(x) + B(x+1))/3",
                 /*iterationSpaceStr*/ "{[t,x] : 1<=t<=ub_T and lb_x<=x<=ub_x}",
                 /*executionScheduleStr*/ "{[t,x]->[t,0,x,0]}",
                 /*dataReadStrs*/
                 {{"B", "{[t,x]->[c]: c=x-1}"},
                  {"B", "{[t,x]->[x]}"},
                  {"B", "{[t,x]->[c]: c=x+1}"}},
                 /*dataWriteStrs*/ {{"A", "{[t,x]->[x]}"}}));
    jacobi->addStmt(
        new Stmt(/*stmtSourceCode*/ "B(x) = (A(x-1) + A(x) + A(x+1))/3",
                 /*iterationSpaceStr*/ "{[t,x] : 1<=t<=ub_T and lb_x<=x<=ub_x}",
                 "{[t,x]->[t,1,x,0]}",
                 /*dataReadStrs*/
                 {{"A", "{[t,x]->[c]: c=x-1}"},
                  {"A", "{[t,x]->[c,a]: c=x and a=x}"},
                  {"A", "{[t,x]->[c]: c=x+1}"}},
                 /*dataWriteStrs*/ {{"B", "{[t,x]->[x]}"}}));

    jacobi->addTransformation(0, new Relation("{[a,b,c,d]->[a,0,x,0]:x=c-1}"));
    jacobi->addTransformation(1, new Relation("{[a,b,c,d]->[a,0,c,1]}"));

    testCases.push_back(jacobi);
  }
  {
    Computation *gpu_mttkrp = new Computation();
    gpu_mttkrp->addDataSpace("A", "double*");
    gpu_mttkrp->addDataSpace("B", "double*");
    gpu_mttkrp->addDataSpace("D", "double*");
    gpu_mttkrp->addDataSpace("C", "double*");
    Stmt *s1 =
        new Stmt("A(i,j) += B(z)*D(l,j)*C(k,j)",
                 "{[b,tx,ty,nl,z,i,k,l,j] : 0<=b<BLOCKS and 0<=tx<THREADS_X "
                 "and 0<=ty<THREADS_Y and "
                 "0<=nl<NUM_LOOPS_NNZ and z=UFz(tx,ty,nl) and i=UFi(z) and "
                 "k=UFk(z) and l=UFl(z) and 0<=j<J and z<NNZ}",
                 "{[b,tx,ty,nl,z,i,k,l,j]->[b,tx,ty,nl,z,i,k,l,j]}",
                 {
                     // data reads
                     {"A", "{[b,tx,ty,nl,z,i,k,l,j]->[i,j]}"},
                     {"B", "{[b,tx,ty,nl,z,i,k,l,j]->[i,k,l]}"},
                     {"C", "{[b,tx,ty,nl,z,i,k,l,j]->[k,j]}"},
                     {"D", "{[b,tx,ty,nl,z,i,k,l,j]->[l,j]}"},
                 },
                 {
                     // data writes
                     {"A", "{[b,tx,ty,nl,z,i,k,l,j]->[i,j]}"},
                 });
    gpu_mttkrp->addStmt(s1);

    testCases.push_back(gpu_mttkrp);
  }

  int testIdx = 0;
  for (auto testCase : testCases) {
    std::cout << "test " << testIdx++
              << " ========================================\n";
    std::cout << "codeGen ===========================\n";
    std::cout << testCase->codeGen();

    std::cout << "codeJen ===========================\n";
    auto codeJen = testCase->codeJen();
    auto code = std::get<0>(codeJen);
    auto vOmegaReplacer = std::get<1>(codeJen);

    {
      // https://stackoverflow.com/a/14266139/3217397
      std::string codeCopy = code;
      size_t pos = 0;
      size_t lineNumber = 0;
      std::string line;
      while ((pos = codeCopy.find("\n")) != std::string::npos) {
        line = codeCopy.substr(0, pos);
        std::cout << std::setw(2) << ++lineNumber << "|" << line << std::endl;
        codeCopy.erase(0, pos + 1);
      }
    }

    std::cout << "parse =============================\n";
    auto lexer = mlir::standalone::parser::Lexer(std::move(code));
    auto parser = mlir::standalone::parser::Parser(lexer, vOmegaReplacer);
    auto ast = parser.parseProgram();

    auto red = "\033[31m";
    auto green = "\033[32m";
    auto reset = "\033[0m";
    if (ast) {
      std::cout << ast->dump();
      std::cout << green << "pass " << reset <<"==========================================\n";
    } else {
      std::cout << red << "fail " << reset <<"==========================================\n";
      exit(1);
    }

    delete vOmegaReplacer;
  }
}