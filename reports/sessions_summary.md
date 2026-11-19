# BTC session analysis by weekday (ET), 2024-01-09 -> 2026-09-07

Returns are log returns in %, spot BTC-USD (coinbase). `liq_if_long_40x` = share of sessions whose path fell >= 1.27% below entry (an isolated 40x long with no stop would have been liquidated); `liq_if_short_40x` uses the 1.23% rise. `sharpe_long` = annualised Sharpe of always being long that session on that weekday (52 obs/yr).


## overnight_16to0930  (16:00 -> 09:30 ET; prev close -> today's open)

           session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
overnight_16to0930 Mon 139     0.112       0.111    2.426     0.554   0.547        0.334            0.453             0.590    -17.490     4.693
overnight_16to0930 Tue 138    -0.263      -0.160    1.985     0.457  -1.555       -0.954            0.522             0.428     -7.211     4.338
overnight_16to0930 Wed 139     0.312       0.102    1.969     0.532   1.868        1.143            0.396             0.453     -3.677     7.667
overnight_16to0930 Thu 139     0.198       0.123    1.931     0.532   1.211        0.741            0.374             0.511     -5.866     5.435
overnight_16to0930 Fri 139     0.288       0.284    1.843     0.547   1.843        1.127            0.367             0.525     -4.792     5.873
overnight_16to0930 Sat 139    -0.024       0.049    0.959     0.525  -0.298       -0.182            0.281             0.209     -3.687     2.686
overnight_16to0930 Sun 139     0.069       0.028    1.055     0.525   0.774        0.473            0.230             0.252     -3.649     4.436
overnight_16to0930 ALL 972     0.099       0.049    1.811     0.525   1.711        1.047            0.374             0.424    -17.490     7.667

conditional on the latest known ETF flow print at entry:
           session flow_bucket   n  mean_pct  hit_rate  t_stat
overnight_16to0930      inflow 565     0.189     0.535   2.659
overnight_16to0930  none/stale  23    -0.512     0.348  -1.057
overnight_16to0930     outflow 384     0.004     0.521   0.042


## preopen_0to0930  (00:00 -> 09:30 ET; midnight -> open)

        session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
preopen_0to0930 Mon 139     0.090       0.063    1.569     0.525   0.675        0.413            0.259             0.367     -8.034     5.158
preopen_0to0930 Tue 139    -0.141      -0.053    1.340     0.475  -1.236       -0.756            0.295             0.266     -3.946     3.646
preopen_0to0930 Wed 139     0.253       0.164    1.468     0.554   2.035        1.245            0.273             0.367     -4.274     5.684
preopen_0to0930 Thu 139     0.052       0.072    1.246     0.511   0.495        0.303            0.259             0.324     -4.220     3.924
preopen_0to0930 Fri 139     0.104       0.063    1.345     0.547   0.911        0.557            0.281             0.353     -3.435     4.669
preopen_0to0930 Sat 139    -0.032       0.014    0.696     0.511  -0.542       -0.331            0.101             0.086     -2.875     1.636
preopen_0to0930 Sun 139    -0.016       0.022    0.779     0.504  -0.249       -0.152            0.165             0.122     -1.824     2.702
preopen_0to0930 ALL 973     0.044       0.037    1.248     0.518   1.109        0.678            0.233             0.269     -8.034     5.684

conditional on the latest known ETF flow print at entry:
        session flow_bucket   n  mean_pct  hit_rate  t_stat
preopen_0to0930      inflow 566     0.109     0.528   2.286
preopen_0to0930  none/stale  23     0.463     0.739   1.794
preopen_0to0930     outflow 384    -0.075     0.490  -1.059


## last_hour_preopen  (08:30 -> 09:30 ET; the hour before the bell)

          session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
last_hour_preopen Mon 139    -0.038      -0.056    0.368     0.410  -1.228       -0.751            0.022             0.029     -1.164     0.921
last_hour_preopen Tue 139     0.028       0.005    0.508     0.511   0.651        0.398            0.050             0.043     -1.611     2.192
last_hour_preopen Wed 139     0.002      -0.035    0.686     0.468   0.028        0.017            0.050             0.086     -2.192     2.584
last_hour_preopen Thu 139     0.025       0.031    0.525     0.554   0.555        0.339            0.036             0.058     -2.418     1.733
last_hour_preopen Fri 139     0.014       0.015    0.650     0.504   0.252        0.154            0.065             0.065     -2.521     3.511
last_hour_preopen Sat 139     0.019       0.026    0.295     0.547   0.763        0.467            0.007             0.007     -1.506     1.157
last_hour_preopen Sun 139    -0.045      -0.012    0.379     0.475  -1.400       -0.856            0.014             0.014     -1.558     1.321
last_hour_preopen ALL 973     0.001      -0.005    0.505     0.495   0.036        0.022            0.035             0.043     -2.521     3.511

conditional on the latest known ETF flow print at entry:
          session flow_bucket   n  mean_pct  hit_rate  t_stat
last_hour_preopen      inflow 566     0.021     0.521   1.073
last_hour_preopen  none/stale  23     0.110     0.565   1.108
last_hour_preopen     outflow 384    -0.036     0.453  -1.273


## first_hour_open  (09:30 -> 10:30 ET; the hour after the bell)

        session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
first_hour_open Mon 139     0.118       0.013    1.219     0.511   1.143        0.699            0.158             0.216     -3.468     5.702
first_hour_open Tue 139    -0.115      -0.095    0.947     0.439  -1.432       -0.876            0.209             0.187     -3.063     1.912
first_hour_open Wed 139     0.026       0.029    0.872     0.525   0.354        0.216            0.122             0.144     -2.761     2.484
first_hour_open Thu 139    -0.182      -0.079    0.939     0.453  -2.288       -1.399            0.194             0.079     -4.845     2.023
first_hour_open Fri 139    -0.031      -0.034    1.039     0.475  -0.357       -0.218            0.165             0.209     -3.512     3.034
first_hour_open Sat 139     0.035       0.034    0.246     0.590   1.678        1.027            0.000             0.007     -0.776     1.232
first_hour_open Sun 139    -0.029      -0.019    0.373     0.468  -0.932       -0.570            0.022             0.014     -1.523     1.178
first_hour_open ALL 973    -0.026      -0.004    0.873     0.494  -0.914       -0.559            0.124             0.122     -4.845     5.702

conditional on the latest known ETF flow print at entry:
        session flow_bucket   n  mean_pct  hit_rate  t_stat
first_hour_open      inflow 566    -0.036     0.493  -1.081
first_hour_open  none/stale  23    -0.348     0.304  -2.095
first_hour_open     outflow 384     0.009     0.508   0.187


## ny_session_0930to16  (09:30 -> 16:00 ET; buy at the open, sell at the close)

            session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
ny_session_0930to16 Mon 138     0.284       0.062    2.178     0.507   1.530        0.939            0.377             0.428     -8.010     7.237
ny_session_0930to16 Tue 139    -0.066       0.161    1.766     0.532  -0.439       -0.268            0.504             0.432     -8.864     4.827
ny_session_0930to16 Wed 139     0.060      -0.092    1.791     0.482   0.397        0.243            0.367             0.381     -4.503     6.933
ny_session_0930to16 Thu 139    -0.567      -0.160    2.035     0.460  -3.283       -2.008            0.410             0.345     -9.572     4.121
ny_session_0930to16 Fri 139    -0.048       0.103    1.897     0.532  -0.297       -0.182            0.403             0.482     -5.923     4.713
ny_session_0930to16 Sat 139     0.036       0.072    0.858     0.576   0.496        0.304            0.079             0.108     -4.478     3.298
ny_session_0930to16 Sun 139    -0.010       0.055    1.250     0.547  -0.091       -0.056            0.194             0.094     -3.597     9.984
ny_session_0930to16 ALL 972    -0.045       0.049    1.747     0.520  -0.796       -0.487            0.333             0.324     -9.572     9.984

conditional on the latest known ETF flow print at entry:
            session flow_bucket   n  mean_pct  hit_rate  t_stat
ny_session_0930to16      inflow 565    -0.007     0.526  -0.102
ny_session_0930to16  none/stale  23    -0.270     0.478  -0.998
ny_session_0930to16     outflow 384    -0.086     0.513  -0.935


## post_open_10to16  (10:00 -> 16:00 ET; buy 30 min after the open, sell at the close)

         session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
post_open_10to16 Mon 138     0.238       0.190    1.743     0.558   1.604        0.984            0.312             0.428     -4.090     5.248
post_open_10to16 Tue 139     0.019       0.166    1.759     0.547   0.128        0.078            0.432             0.410    -10.642     4.321
post_open_10to16 Wed 139     0.098       0.002    1.709     0.504   0.677        0.414            0.324             0.396     -4.331     5.860
post_open_10to16 Thu 139    -0.414      -0.162    1.885     0.468  -2.590       -1.584            0.396             0.338     -8.511     3.725
post_open_10to16 Fri 139    -0.121       0.020    1.709     0.504  -0.835       -0.511            0.381             0.403     -4.413     4.047
post_open_10to16 Sat 139     0.039       0.072    0.855     0.633   0.543        0.332            0.072             0.086     -4.352     3.853
post_open_10to16 Sun 139     0.006      -0.015    1.243     0.496   0.059        0.036            0.180             0.115     -3.318    10.349
post_open_10to16 ALL 972    -0.019       0.047    1.601     0.530  -0.379       -0.232            0.299             0.311    -10.642    10.349

conditional on the latest known ETF flow print at entry:
         session flow_bucket   n  mean_pct  hit_rate  t_stat
post_open_10to16      inflow 565     0.040     0.524   0.595
post_open_10to16  none/stale  23    -0.009     0.565  -0.027
post_open_10to16     outflow 384    -0.107     0.536  -1.283


## after_hours_16to24  (16:00 -> 24:00 ET; close -> midnight)

           session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
after_hours_16to24 Mon 138    -0.122      -0.168    1.259     0.413  -1.140       -0.700            0.341             0.304     -4.311     4.069
after_hours_16to24 Tue 139     0.059       0.103    1.336     0.525   0.516        0.316            0.259             0.209     -4.056     6.735
after_hours_16to24 Wed 139     0.146       0.100    1.366     0.540   1.259        0.770            0.230             0.374     -4.050     4.329
after_hours_16to24 Thu 139     0.184       0.210    1.237     0.547   1.756        1.074            0.194             0.353     -5.347     3.591
after_hours_16to24 Fri 139     0.008       0.055    0.748     0.540   0.122        0.075            0.165             0.115     -3.713     2.348
after_hours_16to24 Sat 139     0.086       0.003    0.810     0.504   1.247        0.762            0.122             0.194     -2.888     3.618
after_hours_16to24 Sun 139     0.023       0.216    1.875     0.525   0.143        0.087            0.345             0.432     -9.456     4.494
after_hours_16to24 ALL 972     0.055       0.031    1.281     0.513   1.334        0.816            0.237             0.283     -9.456     6.735

conditional on the latest known ETF flow print at entry:
           session flow_bucket   n  mean_pct  hit_rate  t_stat
after_hours_16to24      inflow 565     0.098     0.515   1.903
after_hours_16to24  none/stale  23    -0.325     0.348  -1.146
after_hours_16to24     outflow 384     0.014     0.521   0.205


## full_day_0to24  (00:00 -> 24:00 ET; calendar day (ET))

       session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
full_day_0to24 Mon 138     0.251       0.016    2.723     0.507   1.085        0.666            0.551             0.652    -10.770     8.277
full_day_0to24 Tue 139    -0.148      -0.062    2.452     0.489  -0.710       -0.434            0.583             0.525     -7.449     8.434
full_day_0to24 Wed 139     0.460       0.152    2.775     0.547   1.953        1.195            0.540             0.612     -7.692     8.697
full_day_0to24 Thu 139    -0.330      -0.330    2.555     0.424  -1.524       -0.932            0.633             0.525     -8.709     7.438
full_day_0to24 Fri 139     0.064       0.100    2.444     0.525   0.308        0.188            0.532             0.568     -7.422     8.700
full_day_0to24 Sat 139     0.090       0.168    1.344     0.561   0.788        0.482            0.223             0.288     -6.081     3.486
full_day_0to24 Sun 139    -0.003       0.080    2.489     0.540  -0.016       -0.010            0.525             0.504    -11.900     7.998
full_day_0to24 ALL 972     0.055       0.030    2.443     0.513   0.696        0.426            0.512             0.525    -11.900     8.700

conditional on the latest known ETF flow print at entry:
       session flow_bucket   n  mean_pct  hit_rate  t_stat
full_day_0to24      inflow 565     0.199     0.533   1.975
full_day_0to24  none/stale  23    -0.132     0.609  -0.318
full_day_0to24     outflow 384    -0.147     0.479  -1.143


## buy_0900_hold_8h  (09:00 -> 17:00 ET; bot's pre-open trade with the 8h horizon)

         session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
buy_0900_hold_8h Mon 138     0.279       0.081    2.338     0.522   1.403        0.861            0.428             0.435     -9.228     8.824
buy_0900_hold_8h Tue 139    -0.006       0.103    1.724     0.525  -0.043       -0.027            0.504             0.453     -6.880     5.580
buy_0900_hold_8h Wed 139     0.143       0.037    1.818     0.511   0.930        0.569            0.446             0.424     -3.866     8.102
buy_0900_hold_8h Thu 139    -0.442       0.012    1.908     0.511  -2.728       -1.669            0.453             0.381     -9.792     3.558
buy_0900_hold_8h Fri 139    -0.071       0.072    1.955     0.518  -0.430       -0.263            0.439             0.489     -7.289     4.438
buy_0900_hold_8h Sat 139     0.020       0.052    1.124     0.561   0.205        0.125            0.094             0.101     -8.576     3.291
buy_0900_hold_8h Sun 139    -0.091      -0.025    1.332     0.482  -0.805       -0.493            0.237             0.165     -4.756     9.577
buy_0900_hold_8h ALL 972    -0.024       0.033    1.789     0.519  -0.424       -0.260            0.371             0.350     -9.792     9.577

conditional on the latest known ETF flow print at entry:
         session flow_bucket   n  mean_pct  hit_rate  t_stat
buy_0900_hold_8h      inflow 565     0.027     0.526   0.373
buy_0900_hold_8h  none/stale  23    -0.104     0.522  -0.336
buy_0900_hold_8h     outflow 384    -0.095     0.508  -0.965


## buy_1000_hold_8h  (10:00 -> 18:00 ET; bot's post-open trade with the 8h horizon)

         session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
buy_1000_hold_8h Mon 138     0.330       0.257    1.744     0.580   2.224        1.365            0.362             0.449     -4.149     6.431
buy_1000_hold_8h Tue 139     0.100       0.164    1.691     0.525   0.701        0.429            0.446             0.439     -8.175     5.068
buy_1000_hold_8h Wed 139     0.237       0.102    1.713     0.532   1.630        0.997            0.374             0.417     -3.801     7.040
buy_1000_hold_8h Thu 139    -0.170      -0.058    1.828     0.482  -1.096       -0.671            0.403             0.353     -8.241     6.597
buy_1000_hold_8h Fri 139    -0.087       0.112    1.831     0.518  -0.559       -0.342            0.410             0.439     -6.725     5.292
buy_1000_hold_8h Sat 139     0.024       0.102    1.021     0.568   0.276        0.169            0.086             0.108     -7.019     4.079
buy_1000_hold_8h Sun 139    -0.005       0.006    1.361     0.504  -0.039       -0.024            0.245             0.137     -4.191    10.222
buy_1000_hold_8h ALL 972     0.061       0.089    1.625     0.530   1.173        0.718            0.332             0.334     -8.241    10.222

conditional on the latest known ETF flow print at entry:
         session flow_bucket   n  mean_pct  hit_rate  t_stat
buy_1000_hold_8h      inflow 565     0.103     0.538   1.538
buy_1000_hold_8h  none/stale  23     0.077     0.478   0.222
buy_1000_hold_8h     outflow 384    -0.002     0.521  -0.024


## buy_0900_hold_24h  (09:00 -> 09:00 ET, next day; pre-open, hold a day)

          session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
buy_0900_hold_24h Mon 138    -0.055      -0.156    3.033     0.478  -0.212       -0.130            0.558             0.529    -12.278    10.987
buy_0900_hold_24h Tue 139     0.288       0.223    2.546     0.532   1.334        0.816            0.576             0.604     -5.501     7.796
buy_0900_hold_24h Wed 139     0.244       0.225    2.500     0.561   1.153        0.705            0.576             0.583     -7.194    10.346
buy_0900_hold_24h Thu 139    -0.235       0.178    2.604     0.532  -1.065       -0.651            0.525             0.532     -8.476     7.115
buy_0900_hold_24h Fri 139    -0.080       0.178    1.997     0.561  -0.470       -0.288            0.460             0.532     -7.819     5.521
buy_0900_hold_24h Sat 139     0.131       0.242    1.294     0.576   1.193        0.729            0.223             0.273     -5.453     4.255
buy_0900_hold_24h Sun 139     0.087       0.073    2.857     0.511   0.358        0.219            0.504             0.576    -20.337     8.811
buy_0900_hold_24h ALL 972     0.055       0.134    2.463     0.536   0.690        0.422            0.489             0.519    -20.337    10.987

conditional on the latest known ETF flow print at entry:
          session flow_bucket   n  mean_pct  hit_rate  t_stat
buy_0900_hold_24h      inflow 565     0.186     0.545   1.851
buy_0900_hold_24h  none/stale  23    -0.706     0.391  -1.268
buy_0900_hold_24h     outflow 384    -0.094     0.531  -0.721


## buy_1000_hold_24h  (10:00 -> 10:00 ET, next day; post-open, hold a day)

          session dow   n  mean_pct  median_pct  std_pct  hit_rate  t_stat  sharpe_long  liq_if_long_40x  liq_if_short_40x  worst_pct  best_pct
buy_1000_hold_24h Mon 138    -0.107      -0.280    2.744     0.449  -0.459       -0.282            0.529             0.580     -8.351     9.779
buy_1000_hold_24h Tue 139     0.293       0.231    2.440     0.554   1.417        0.866            0.561             0.597     -6.020     7.387
buy_1000_hold_24h Wed 139     0.144      -0.004    2.550     0.496   0.664        0.406            0.511             0.626     -6.925     9.798
buy_1000_hold_24h Thu 139    -0.053       0.162    2.691     0.511  -0.231       -0.141            0.532             0.561     -7.111     7.518
buy_1000_hold_24h Fri 139    -0.148       0.038    1.829     0.525  -0.957       -0.585            0.475             0.504     -7.987     4.985
buy_1000_hold_24h Sat 139     0.093       0.247    1.340     0.619   0.816        0.499            0.245             0.324     -4.793     4.061
buy_1000_hold_24h Sun 139     0.161       0.152    2.588     0.540   0.735        0.450            0.518             0.561    -15.678     6.424
buy_1000_hold_24h ALL 972     0.055       0.134    2.360     0.528   0.724        0.443            0.481             0.536    -15.678     9.798

conditional on the latest known ETF flow print at entry:
          session flow_bucket   n  mean_pct  hit_rate  t_stat
buy_1000_hold_24h      inflow 565     0.193     0.527   1.995
buy_1000_hold_24h  none/stale  23    -0.597     0.435  -0.887
buy_1000_hold_24h     outflow 384    -0.110     0.534  -0.910
