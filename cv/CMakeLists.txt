add_library(ygz-cv
        src/DSOCoarseTracker.cpp
        src/ORBExtractor.cpp
        src/ORBMatcher.cpp
        src/Tracker.cpp
        src/SparePoint.cpp
        src/PixelSelector.cpp
        src/Align.cpp
        src/LKFlow.cpp
        src/LineFeature.cpp
        )

target_link_libraries(ygz-cv ygz-common ygz-util ${THIRD_PARTY_LIBS})
