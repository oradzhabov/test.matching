include_directories(${OpenCV_INCLUDE_DIR})

SET(SOURCE
	ARDrawingContext.cpp
	CameraCalibration.cpp
	GeometryTypes.cpp
	main.cpp
	ARPipeline.cpp
	Pattern.cpp
	PatternDetector.cpp
	RobustMatcher.cpp
)

SET(HEADERS
	PatternDetector.hpp
	ARDrawingContext.hpp
	CameraCalibration.hpp
	GeometryTypes.hpp
	ARPipeline.hpp
	Pattern.hpp
	DebugHelpers.hpp
	RobustMatcher.h
)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("./")


#
# Do not using the Flag WIN32 makes app as Console instead of Windows
#
add_executable(rd.2 ${SOURCE} ${HEADERS})

add_definitions(
	-DUSE_PPL
	-D_UNICODE
	-DUNICODE
)

target_link_libraries(rd.2 ${OPENGL_LIBRARIES} ${OpenCV_LIBRARIES})

