all: meter

.PHONY : clean

meter: meter.o segment.o unionfind.o
	g++ -o meter meter.o segment.o unionfind.o -lopencv_core -lopencv_imgproc -lopencv_highgui

meter.o: meter.cpp
	g++ -c meter.cpp

segment.o: segment.cpp
	g++ -c segment.cpp

unionfind.o: unionfind.cpp unionfind.hpp
	g++ -c unionfind.cpp

clean: 
	rm *.o
