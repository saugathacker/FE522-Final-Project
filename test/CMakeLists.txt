add_executable(test test.cpp inputOutputTest.cpp)
target_link_libraries(test PRIVATE GTest::gtest GTest::gtest_main libmatrix libfilehandler)

# Add CSV file to the test executable (if needed by inputOutputTest.cpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sampleTest.csv ${CMAKE_CURRENT_BINARY_DIR}/sampleTest.csv COPYONLY)
