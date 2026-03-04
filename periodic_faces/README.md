# Periodic faces and MeshWorker

> Though the code uses parallel triangulation, run this in serial only, which is enough to test the periodicity. DO NOT RUN IN PARALLEL.

We make a 3x3 mesh with periodicity in both directions.

Cells are numbered like this.

```
|-----|-----|-----|
|  6  |  7  |  8  |
|-----|-----|-----|
|  3  |  4  |  5  |
|-----|-----|-----|
|  0  |  1  |  2  |
|-----|-----|-----|
```

When you run the code, 

```
----- Cell no = 0----- 
  Periodic neigbor cell = 2
  Periodic neigbor cell = 6
----- Cell no = 1----- 
  Periodic neigbor cell = 7
----- Cell no = 2----- 
  Periodic neigbor cell = 0
  Periodic neigbor cell = 8
----- Cell no = 3----- 
  Periodic neigbor cell = 5
----- Cell no = 4----- 
----- Cell no = 5----- 
  Periodic neigbor cell = 3
----- Cell no = 6----- 
  Periodic neigbor cell = 8
  Periodic neigbor cell = 0
----- Cell no = 7----- 
  Periodic neigbor cell = 1
----- Cell no = 8----- 
  Periodic neigbor cell = 6
  Periodic neigbor cell = 2

No of boundary faces = 12

Beginning of mesh_loop:

Interior face: (cell,neigh) = (0,2)
Interior face: (cell,neigh) = (0,1)
Interior face: (cell,neigh) = (0,6)
Interior face: (cell,neigh) = (0,3)
Interior face: (cell,neigh) = (1,2)
Interior face: (cell,neigh) = (1,7)
Interior face: (cell,neigh) = (1,4)
Interior face: (cell,neigh) = (2,8)
Interior face: (cell,neigh) = (2,5)
Interior face: (cell,neigh) = (3,5)
Interior face: (cell,neigh) = (3,4)
Interior face: (cell,neigh) = (3,6)
Interior face: (cell,neigh) = (4,5)
Interior face: (cell,neigh) = (4,7)
Interior face: (cell,neigh) = (5,8)
Interior face: (cell,neigh) = (6,8)
Interior face: (cell,neigh) = (6,7)
Interior face: (cell,neigh) = (7,8)
```

you see that boundary_worker is never called, because there are no boundary faces. Faces on the boundary are periodic and they are treated as interior faces.

There is a periodic face between `cell 0` and `cell 2`; this face is visited only from `cell 0`, see the output `(0,2)` is present but `(2,0)` is not present.

