SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all:  shomagick matcher


DEPDIR = deps
$(shell mkdir -p $(DEPDIR) >/dev/null)

CXXFLAGS += -std=c++11
CXXFLAGS += -O3
CXXFLAGS += -g
CXXFLAGS += -Wall -Wextra -pedantic
CXXFLAGS += -Weffc++
CXXFLAGS += -Werror=reorder
CXXFLAGS += -Werror=return-type

OPENCV_LINK_FLAGS += -lopencv_core
OPENCV_LINK_FLAGS += -lopencv_highgui
OPENCV_LINK_FLAGS += -lopencv_imgcodecs
OPENCV_LINK_FLAGS += -lopencv_imgproc
OPENCV_LINK_FLAGS += -lopencv_videoio
OPENCV_LINK_FLAGS += -lopencv_ml
OPENCV_LINK_FLAGS += -lopencv_xfeatures2d
OPENCV_LINK_FLAGS += -lopencv_objdetect

LFLAGS += -lboost_filesystem
LFLAGS += -lboost_system
LFLAGS += -lexiv2
LFLAGS += `pkg-config --libs opencv`
LFLAGS += -g

shomagick : stitch.cpp kdtree.cpp flightsession.cpp shomatcher.cpp RobustMatcher.cpp shotracking.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LFLAGS)

matcher: keypointsmatcher.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LFLAGS) 

clean:
	rm -f $(OBJECTS) $(addprefix $(DEPDIR)/, $(DEPS)) shomagick shomagick.exe matcher matcher.exe