set term postscript enhanced color

set out 'sod_den.eps'
set ylabel 'Density'
set yran[0.1:1.1]
set key right top
p 'avg.gpl' u 1:2 t 'DG' w p pt 6 lw 3, \
  'sod.dat' u 1:2 t 'Exact' w l lw 4 lc 3

set out 'sod_vel.eps'
set ylabel 'Velocity'
set yran[-0.1:1.0]
set key bottom center
p 'avg.gpl' u 1:3 t 'DG' w p pt 6 lw 3, \
  'sod.dat' u 1:3 t 'Exact' w l lw 4 lc 3

set out 'sod_pre.eps'
set ylabel 'Pressure'
set yran[-0.0:1.1]
set key right top
p 'avg.gpl' u 1:4 t 'DG' w p pt 6 lw 3, \
  'sod.dat' u 1:4 t 'Exact' w l lw 4 lc 3

set out 'sod_ind.eps'
set ylabel 'Indicator'
set yran[-0.0:1.1]
unset key
p 'avg.gpl' u 1:5 t 'DG' w p pt 6 lw 3, \
  'sod.dat' u 1:2 t 'Exact' w l lw 4 lc 3
