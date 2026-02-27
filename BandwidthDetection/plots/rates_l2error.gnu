set terminal cairolatex size 7cm, 7cm
set datafile separator comma

unset key

set size square

set style line 1 lw 3 ps 0.6 pt 7 lc 'black'

set grid lt 1 lw 3 lc 'gray'
set border ls 1

set format y '\scriptsize $%.0l\cdot 10^{%L}$'
set format x '\scriptsize $%.0f$'
set logscale y

set xlabel 'iteration'
set ylabel '$L_2$ error'

set ytics  10 logscale
set yrange [1e-11:1e-7]
set output "img/s2_rates_l2error.tex"
plot "plotdata/s2_rates_l2error.csv" using 1:2 with linespoints ls 1

#set grid mytics lt 1 lw 3 lc 'gray'
#set format y '\scriptsize $10^{%L}$'
#set ytics  10 logscale
#set yrange [1e-8:1e-6]
#set output "img/s5_rates_l2error.tex"
#plot "../dat/s5_rates_l2error.csv" using 1:2 with linespoints ls 1

#set format y '\scriptsize $%.0l\cdot 10^{%L}$'
#set ytics  1e-3 nologscale
#set yrange [2e-3:9e-3]
#set output "img/s10_rates_l2error.tex"
#plot "../dat/s10_rates_l2error.csv" using 1:2 with linespoints ls 1
