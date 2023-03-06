configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPF_TEST_DEPENDS
        FileCheck count not
        spf-opt
        )

add_lit_testsuite(check-spf "Running the spf regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SPF_TEST_DEPENDS}
        )
set_target_properties(check-spf PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPF ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SPF_TEST_DEPENDS})
