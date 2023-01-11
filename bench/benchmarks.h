
#ifndef BENCHMARKS_H
#define BENCHMARKS_H

double cpu_mttkrp_iegenlib(bool debug, char *filename);

double cpu_mttkrp_mlir(bool debug, char *filename);

double gpu_mttkrp_mlir(bool debug, char *filename);

#endif // BENCHMARKS_H