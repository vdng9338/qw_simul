EXEC=triangular_uniform triangular_distorted triangular_cone triangular_cone_sigmax triangular_cone_square_uniform triangular_cone_newdisloc
HEADERS=tinycolormap.hpp matplotlibcpp.h

all: $(EXEC)

.PHONY: clean

clean:
	rm -rf $(EXEC)

%: %.cpp $(HEADERS)
	g++ -Wall -Wextra -std=c++17 -O2 -pthread -I/usr/include/eigen3 -I/usr/include/python3.8 -o $@ $< -lpython3.8
