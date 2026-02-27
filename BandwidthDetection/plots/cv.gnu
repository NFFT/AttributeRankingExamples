set terminal cairolatex size 14cm, 7cm
set datafile separator comma

set key outside Left reverse spacing 1.5 width 5
#set key columns 1
set key samplen 2

set margin 3.5,10,3.3,1

set size square

set style line 1 lw 3 ps 0.6 pt 7 lc 'black'

set grid lt 1 lw 3 lc 'gray'
set border ls 1

set format x '\scriptsize $10^{%L}$'
set format y '\scriptsize $%.0l\cdot 10^{%L}$'
set logscale


set yrange [5e-2:9e-2]
set xrange [300:1e5]

set ytics 0.01, 0.01 nologscale

set xlabel '$B$'

#set output "img/s10_cv.tex"
#sigma = "`cat plotdata/s10_cv_sigma.csv`"
#plot "plotdata/s10_cv_it1.csv" using "B":"cv" with linespoints ls 1 lc "#ffa500" title 'CV',\
#    "plotdata/s10_cv_it1.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
#    "plotdata/s10_cv_it2.csv" using "B":"cv" with linespoints ls 1 lc "#ff4dff" title 'CV',\
#    "plotdata/s10_cv_it2.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
#    "plotdata/s10_cv_it3.csv" using "B":"cv" with linespoints ls 1 lc "#00b3b3" title 'CV',\
#    "plotdata/s10_cv_it3.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$'


#set yrange [5e-3:8e-3]
#set ytics 1e-3, 1e-3 nologscale

set output "img/s2_cv.tex"
sigma = "`cat plotdata/s2_cv_sigma.csv`"
plot "plotdata/s2_cv_it1.csv" using "B":"cv" with linespoints ls 1 lc "#ffa500" title 'CV',\
    "plotdata/s2_cv_it1.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
    "plotdata/s2_cv_it2.csv" using "B":"cv" with linespoints ls 1 lc "#ff4dff" title 'CV',\
    "plotdata/s2_cv_it2.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
    "plotdata/s2_cv_it3.csv" using "B":"cv" with linespoints ls 1 lc "#00b3b3" title 'CV',\
    "plotdata/s2_cv_it3.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$'

#set yrange [1.5e-4:4e-4]
#set ytics 1e-4, 1e-4 nologscale

#set output "img/s5_cv.tex"
#sigma = "`cat ../dat/s5_cv_sigma.csv`"
#plot "../dat/s5_cv_it1.csv" using "B":"cv" with linespoints ls 1 lc "#ffa500" title 'CV',\
#    "../dat/s5_cv_it1.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
#    "../dat/s5_cv_it2.csv" using "B":"cv" with linespoints ls 1 lc "#ff4dff" title 'CV',\
#    "../dat/s5_cv_it2.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$',\
#    "../dat/s5_cv_it3.csv" using "B":"cv" with linespoints ls 1 lc "#00b3b3" title 'CV',\
#    "../dat/s5_cv_it3.csv" using "B":(column("l2error")**2+sigma**2) with points ls 1 pt 2 title '$(L_2$-error$)^2+\sigma^2$'
