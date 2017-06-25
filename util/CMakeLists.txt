add_library( ygz-util
        src/EurocReader.cpp
        src/Viewer.cpp
        src/KittiReader.cpp
        src/getElement.cpp
        )

target_link_libraries( ygz-util
        ${THIRD_PARTY_LIBS}
)