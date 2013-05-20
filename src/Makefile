all: meter

.PHONY: clean all

meter: meter.o position.o unionfind.o
	g++ -o meter meter.o position.o unionfind.o -lopencv_core -lopencv_imgproc -lopencv_highgui

meter.o: meter.cpp
	g++ -c meter.cpp

position.o: position.cpp position.hpp
	g++ -c position.cpp

unionfind.o: unionfind.cpp unionfind.hpp
	g++ -c unionfind.cpp

clean: 
	rm *.o
