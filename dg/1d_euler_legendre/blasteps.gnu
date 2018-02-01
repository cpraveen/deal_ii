set term postscript enhanced color

set out 'blast.eps'

set ylabel 'Density'
set key right top
p 'avg.gpl' u 1:2 t 'DG' w p pt 6 lw 3, \
  'blast.dat' u 1:2 t 'Exact' w l lw 4 lc 3

set ylabel 'Velocity'
set key bottom center
p 'avg.gpl' u 1:3 t 'DG' w p pt 6 lw 3, \
  'blast.dat' u 1:3 t 'Exact' w l lw 4 lc 3

set ylabel 'Pressure'
set key right top
p 'avg.gpl' u 1:4 t 'DG' w p pt 6 lw 3, \
  'blast.dat' u 1:4 t 'Exact' w l lw 4 lc 3
