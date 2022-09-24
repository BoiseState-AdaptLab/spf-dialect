module {
    func.func @main() {
        "standalone.bar"() ({
        ^bb0(%i : index):
        vector.print %i : index
        }) : () -> ()
        return
    }
}
