// RUN: spf-opt %s | spf-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        return
    }
}
