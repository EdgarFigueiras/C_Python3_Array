/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "C_array.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

/* #### Globals #################################### */

/* ==== Initialize the C_test functions ====================== 

// Boilerplate: Module initialization.
PyMODINIT_FUNC init_C_array(void) {
	(void) Py_InitModule("_C_array", _C_arrayMethods);
	import_array();
}
*/

/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double vector w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 1,1);
}
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
	int n;
	
	n=arrayin->dimensions[0];
	return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublevector: array must be of type Float and 1 dimensional (n).");
		return 1;  }
	return 0;
}


/* #### Matrix Utility functions ######################### */

/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
	free((char*) v);
}
/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;  }
	return 0;
}

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double matrix w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pymatrix(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 2,2);
}

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}

/* ==== Allocate a a *int (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(int ** )                  */
int **ptrintvector(long n)  {
	int **v;
	v=(int **)malloc((size_t) (n*sizeof(int)));
	if (!v)   {
		printf("In **ptrintvector. Allocation of memory for int array failed.");
		exit(0);  }
	return v;
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i,n,m;
	
	n=arrayin->dimensions[0];
	m=arrayin->dimensions[1];
	c=ptrvector(n);
	a=(double *) arrayin->data;  /* pointer to arrayin data as double */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}

/* ==== Free an int *vector (vec of pointers) ========================== */ 
void free_Cint2Darrayptrs(int **v)  {
	free((char*) v);
}
/* ==== Check that PyArrayObject is an int (integer) type and a 2D array ==============
    return 1 if an error and raise exception
    Note:  Use NY_LONG for NumPy integer array, not NP_INT      */ 
int  not_int2Darray(PyArrayObject *mat)  {
	if (mat->descr->type_num != NPY_LONG || mat->nd != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_int2Darray: array must be of type int and 2 dimensional (n x m).");
		return 1;  }
	return 0;
}

/* ==== Make a Python int Array Obj. from a PyObject, ================
     generates a 2D integer array w/ contiguous memory which may be a new allocation if
     the original was not an integer type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyint2Darray(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_LONG, 2,2);
}
/* ==== Create integer 2D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin)  {
	int **c, *a;
	int i,n,m;
	
	n=arrayin->dimensions[0];
	m=arrayin->dimensions[1];
	c=ptrintvector(n);
	a=(int *) arrayin->data;  /* pointer to arrayin data as int */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}

/* #### Matrix Extensions ############################## */

/*Function that returns a random number between a range*/
double randInRange(double min, double max)
{
    double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */

}



/* ==== Operate on Matrix components  =========================
   Access to the introduced matrix data to calculate 3d points using MonteCarlo funciont with random numbers
  
    Returns a NEW NumPy array of the size of the second inserted value and 3, because gives 3d space points
    interface:  Matrix2D(number_points, mat1)
                mat1 is NumPy matrix, x1 is Python float (double)
                returns a NumPy matrix[number_points][3] */
static PyObject* Matrix2D(PyObject* self, PyObject* args)
{
    int n, f, c;
    PyArrayObject *matin, *matout;
    double **cin, **cout;			// Pointers to the contiguous data in the matrices to be used by C 

    if (!PyArg_ParseTuple(args, "iO", &n, &matin))
        return NULL;
    if (NULL == matin)  return NULL;

    f=matin->dimensions[0];
	c=matin->dimensions[1];

	//Vector of dimensions of new array
	int dimensions_cout[2]={n, 3};

	/* Make a new double matrix of same dims */
	matout=(PyArrayObject *) PyArray_FromDims(2,dimensions_cout,NPY_DOUBLE);

	/* Change contiguous arrays into  C ** arrays (Memory is Allocated!) */
	cin=pymatrix_to_Carrayptrs(matin);
	cout=pymatrix_to_Carrayptrs(matout);

	srand(time(NULL));
	double random = randInRange(0,1);
	int random_pointer_x = randInRange(0,f);
	int random_pointer_y = randInRange(0,c);

	for (int i=0; i<dimensions_cout[0]; i++)  {
		random = randInRange(0,1);
		while (random > cin[random_pointer_x][random_pointer_y])
		{
			random = randInRange(0,1);
			random_pointer_x = randInRange(0,f);
			random_pointer_y = randInRange(0,c);
		}
		cout[i][0]= random_pointer_x - n/2; //n has the Z size (len(Z[n](m))), with this substraction operation valuer are better balanced
		cout[i][1]= random_pointer_y - n/2; //because the final objective is show them in a 3D grid
		cout[i][2]= random;
	}

	//printf("%i\n", i*j );
 	free_Carrayptrs(cin);
	free_Carrayptrs(cout);
	return PyArray_Return(matout);
    //return Py_BuildValue("i", Cfib(n));
}

/* ==== Operate on Matrix components  =========================
   Access to the introduced matrix data to calculate 3d points using MonteCarlo funciont with random numbers
  
    Returns a NEW NumPy array of the size of the second inserted value and 4, because gives 3d space points and the probability
    interface:  MatrixProb2D(number_points, mat1)
                mat1 is NumPy matrix, x1 is Python float (double)
                returns a NumPy matrix[number_points][4]                                        */
static PyObject* MatrixProb2D(PyObject* self, PyObject* args)
{
    int n, f, c;
    PyArrayObject *matin, *matout;
    double **cin, **cout;			// Pointers to the contiguous data in the matrices to be used by C 

    if (!PyArg_ParseTuple(args, "iO", &n, &matin))
        return NULL;
    if (NULL == matin)  return NULL;

    f=matin->dimensions[0];
	c=matin->dimensions[1];

	//Vector of dimensions of new array
	int dimensions_cout[2]={n, 4};

	/* Make a new double matrix of same dims */
	matout=(PyArrayObject *) PyArray_FromDims(2,dimensions_cout,NPY_DOUBLE);

	/* Change contiguous arrays into  C ** arrays (Memory is Allocated!) */
	cin=pymatrix_to_Carrayptrs(matin);
	cout=pymatrix_to_Carrayptrs(matout);

	srand(time(NULL));
	double random = randInRange(0,1);
	int random_pointer_x = randInRange(0,f);
	int random_pointer_y = randInRange(0,c);

	for (int i=0; i<dimensions_cout[0]; i++)  {
		random = randInRange(0,1);
		while (random > cin[random_pointer_x][random_pointer_y])
		{
			random = randInRange(0,1);
			random_pointer_x = randInRange(0,f);
			random_pointer_y = randInRange(0,c);
		}
		cout[i][0]= random_pointer_x - n/2; //n has the Z size (len(Z[n](m))), with this substraction operation valuer are better balanced
		cout[i][1]= random_pointer_y - n/2; //because the final objective is show them in a 3D grid
		cout[i][2]= random;
		cout[i][3]= cin[random_pointer_x][random_pointer_y];
	}

	//printf("%i\n", i*j );
 	free_Carrayptrs(cin);
	free_Carrayptrs(cout);
	return PyArray_Return(matout);
    //return Py_BuildValue("i", Cfib(n));
}


static PyObject* version(PyObject* self)
{
    return Py_BuildValue("s", "Version 1.2");
}
 
static PyMethodDef C_array_Methods[] = {
    {"Matrix2D", Matrix2D, METH_VARARGS, "Calculate the matrix values."},
    {"MatrixProb2D", MatrixProb2D, METH_VARARGS, "Calculate the matrix values and return probability too."},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
    {NULL, NULL, 0, NULL}
};
 
static struct PyModuleDef C_array = {
	PyModuleDef_HEAD_INIT,
	"_C_array", //name of module.
	"Matrix operation Module",
	-1,
	C_array_Methods
};

PyMODINIT_FUNC PyInit_C_array(void)
{
	import_array();
    return PyModule_Create(&C_array);
}




// EOF



     
