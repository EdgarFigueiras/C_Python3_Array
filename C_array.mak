# ---- Link ---------------------------
#_C_array.so:  C_array.o  C_array.mak
#	gcc -bundle -flat_namespace -undefined suppress -o _C_array.so  C_array.o

# ---- gcc C compile ------------------
#C_array.o:  C_array.c C_array.h C_array.mak
#	gcc -c C_array.c -I/Users/edgarfigueiras/anaconda/envs/env_3.5.2/include/python3.5m -I/Users/edgarfigueiras/anaconda/envs/env_3.5.2/lib/python3.5/site-packages/numpy/core/include/numpy




run:		
	rm -f C_array.cpython-35m-darwin.so 
	rm -rf build
	python setup.py build_ext --inplace
