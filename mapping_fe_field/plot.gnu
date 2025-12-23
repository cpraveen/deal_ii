set term svg
set out "grid0.svg"
set xran[-0.2:1.2]
set yran[-0.2:1.2]
set size ratio -1
unset key
p "grid0.gnu" w l

set out "grid1.svg"
set xran[-0.2:1.2]
set yran[-0.2:1.2]
set size ratio -1
unset key
p "grid1.gnu" w l
