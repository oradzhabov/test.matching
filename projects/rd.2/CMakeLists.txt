file(GLOB RD2_SOURCES	"*.cpp")
file(GLOB RD2_HEADERS	"*.h" "*.hpp")


source_group("" FILES ${RD2_SOURCES} ${RD2_HEADERS})
source_group("Source Files" FILES "main.cpp")
source_group("Source Files\\Common" FILES "GeometryTypes.hpp" "GeometryTypes.cpp" "DebugHelpers.hpp")
source_group("Source Files\\ImgMatcher" FILES "Pattern.hpp" "Pattern.cpp" "ARPipeline.hpp" "ARPipeline.cpp" "CameraCalibration.hpp" "CameraCalibration.cpp" "PatternDetector.hpp" "PatternDetector.cpp" "OppColorDescriptorExtractor.h" "OppColorDescriptorExtractor.cpp")
source_group("Source Files\\OpenGL" FILES "ARDrawingContext.hpp" "ARDrawingContext.cpp")
source_group("Source Files\\Portrait" FILES "PortraitObs.hpp" "PortraitObs.cpp" "PortraitObsBuilder.hpp" "PortraitObsBuilder.cpp")

include_directories(${OpenCV_INCLUDE_DIR})


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("./")


#
# Do not using the Flag WIN32 makes app as Console instead of Windows
#
add_executable(rd.2 ${RD2_SOURCES} ${RD2_HEADERS})


add_definitions(
	-DUSE_PPL
	-D_UNICODE
	-DUNICODE
)

target_link_libraries(rd.2 ${OPENGL_LIBRARIES} ${OpenCV_LIBRARIES})

