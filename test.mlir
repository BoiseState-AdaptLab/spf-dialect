module {
    func.func @main() {
        "standalone.bar"() ({
        ^bb0(%i : index, %j : index):
        vector.print %i : index
        vector.print %j : index
        }) : () -> ()
        return
    }
}
