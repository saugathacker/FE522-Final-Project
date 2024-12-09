add_executable(test test.cpp)
target_link_libraries(test PRIVATE GTest::gtest GTest::gtest_main libmatrix)