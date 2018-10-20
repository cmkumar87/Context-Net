declare -a CORPUS=("rprog-003" "calc1-003" "compilers-004" "smac-001" "maththink-004" "bioelectricity-002" "gametheory2-001" "musicproduction-006" "medicalneuro-002" "comparch-002" "bioinfomethods1-001")

#declare -a CORPUS=("rprog-003")

for i in "${CORPUS[@]}"
do
    /diskA/anaconda3/bin/python /diskA/muthu/Transact-Net/model/hierarchical_lstm.py -c "$i" -d 300 -ct 1 >"$i".300.1.txt 2>&1  
done
   
#for i in "${CORPUS[@]}"
#do
#    /diskA/anaconda3/bin/python /diskA/muthu/Transact-Net/model/hierarchical_lstm.py -c "$i" -d 50 -ct 999 >"$i".50.999.txt 2>&1
#done
#
#for i in "${CORPUS[@]}"
#do
#    /diskA/anaconda3/bin/python /diskA/muthu/Transact-Net/model/hierarchical_lstm.py -c "$i" -d 300 -ct 999>"$i".300.999.txt 2>&1 
#done
#
#for i in "${CORPUS[@]}"
#do
#    /diskA/anaconda3/bin/python /diskA/muthu/Transact-Net/model/hierarchical_lstm.py -c "$i" -d 50 -ct 1 >"$i".300.999.txt 2>&1 
#done

