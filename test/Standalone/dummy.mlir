// RUN: spf-opt %s | spf-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = spf.foo %{{.*}} : i32
        %res = spf.foo %0 : i32
        return
    }
}
