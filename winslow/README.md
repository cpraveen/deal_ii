# Winslow equation based high order order meshes

The code in "serial" is bit old. It is recommended to use the code in "parallel". The parallel code requires Trilinos, so your deal.II must have been compiled with Trilinos.

```
cd parallel
cmake .
make
```

Now run the code in "run" directory.

```
cd ../run
../parallel/main 0
```

You can also run in parallel 
```
cd ../run
mpirun -np 2 ../parallel/main 0
```
