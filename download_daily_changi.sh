for y in `seq 1950 1980`
do
    for m in `seq 1 12`
    do
        wget -c 'http://www.weather.gov.sg/files/dailydata/DAILYDATA_S24_'`printf '%d%02d' $y $m`'.csv'
    done
done
