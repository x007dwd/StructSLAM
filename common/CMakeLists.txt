add_library(ygz-common
        src/IMUData.cpp
        src/Settings.cpp
        src/Frame.cpp
        src/MapPoint.cpp
        src/G2OTypes.cpp
        )

target_link_libraries(ygz-common
        ygz-util
        ${THIRD_PARTY_LIBS}
        )