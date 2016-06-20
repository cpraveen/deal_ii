set size ratio -1
set xran[-0.02:0.03]
set yran[-0.04:0.04]

unset key

set term postscript enhanced color
set out 'gridq1.eps'
p 'gridq1.gnu' w l lw 2 lt 2,'bd.gnu' w l lw 3 lt 7

set term postscript enhanced color
set out 'gridqk.eps'
p 'gridqk.gnu' w l lw 2 lt 2,'bd.gnu' w l lw 3 lt 7
