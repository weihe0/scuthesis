all: meter

.PHONY: clean all

meter: meter.o position.o unionfind.o segment.o recognise.o
	g++ -o meter meter.o position.o unionfind.o segment.o recognise.o -lopencv_core -lopencv_imgproc -lopencv_highgui

meter.o: meter.cpp
	g++ -c meter.cpp

position.o: position.cpp position.hpp
	g++ -c position.cpp

segment.o: segment.cpp segment.hpp
	g++ -c segment.cpp

recognise.o :recognise.cpp recognise.hpp
	g++ -c recognise.cpp

unionfind.o: unionfind.cpp unionfind.hpp
	g++ -c unionfind.cpp

clean: 
	rm *.o
