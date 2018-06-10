include_directories(${OpenCV_INCLUDE_DIR})

SET(SOURCE
	main.cpp
	Provider/Provider.cpp
	Provider/ImageProvider.cpp
	Provider/VideoProvider.cpp
	State/State.cpp
	Template/Template.cpp
	Helper/Shelf.cpp
	Helper/AdjustPerspective.cpp
	Draw/lines.cpp
	KPDetector/KPDetector.cpp
	KPDetector/FindCameraMatrices.cpp
)

SET(HEADERS
	main.h
	Provider/Provider.h
	Provider/ImageProvider.h
	Provider/VideoProvider.h
	State/State.h
	Template/Template.h
	KPDetector/KPDetector.h
)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("./")


#
# Do not using the Flag WIN32 makes app as Console instead of Windows
#
add_executable(rd.1 ${SOURCE} ${HEADERS})

add_definitions(
	-DUSE_PPL
	-D_UNICODE
	-DUNICODE
)

target_link_libraries(rd.1 ${OpenCV_LIBRARIES} )
