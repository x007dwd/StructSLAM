add_executable(
        test_feature_extraction testFeatureExtraction.cpp
)

target_link_libraries(
        test_feature_extraction
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_stereo_matching testStereoMatching.cpp
)

target_link_libraries(
        test_stereo_matching
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_feature_matching testFeatureMatching.cpp
)

target_link_libraries(
        test_feature_matching
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_localBA testLocalBA.cpp
)

target_link_libraries(
        test_localBA
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_localBAIMU testLocalBAWithIMU.cpp
)

target_link_libraries(
        test_localBAIMU
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(test_viewer testViewer.cpp)
target_link_libraries(
        test_viewer
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(test_stereo_init testStereoInit.cpp)
target_link_libraries(
        test_stereo_init
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(test_pure_vision testPureVision.cpp)
target_link_libraries(
        test_pure_vision
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(test_stereo_init_sim testStereoInitSim.cpp)
target_link_libraries(
        test_stereo_init_sim
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_coarse_tracker testCoarseTracker.cpp
)

target_link_libraries(
        test_coarse_tracker
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_line_match testLineMatch.cpp
)

target_link_libraries(
        test_line_match
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)

add_executable(
        test_g2o_line test_g2o_line.cpp
)

target_link_libraries(
        test_g2o_line
        ${YGZ_LIBS}
        ${THIRD_PARTY_LIBS}
)
