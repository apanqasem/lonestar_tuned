#!/bin/bash

for (( o=1; o<4; o++ ))
do
#optimization
 reg=16

 for (( r=0; r<4; r++ ))
 do
 #registar flags
  blk=32

  for (( b=0; b<6; b++))
  do
  #thd/blk
   sed -i -E "s/THDS_NUM\t+[0-9]+/THDS_NUM\t\t\t$blk/g" comm.h
   sed -i -E "s/BLKS_NUM\t+[0-9]+/BLKS_NUM\t\t\t$blk/g" comm.h

   echo compiling...
   opt=O$o reg=$reg make -B &> /dev/null
   echo

   for g in $(ls /home/alh216/UnwUnd)
   do
   #graph folder
    echo -n $g,O$o,$reg,$blk,$blk >> results.csv
    echo $g, O$o optimization, $reg register flags, $blk threads per block, $blk num of blocks >> power.txt
    echo $g, O$o optimization, $reg register flags, $blk threads per block, $blk num of blocks
    echo

    for (( i=1; i<6; i++))
    do
    #runs
     echo starting run $i
     echo -n , >> results.csv
     nvprof -u ms --system-profiling on --log-file prof.log ./enterprise.bin $(find /home/alh216/UnwUnd/$g -type f -name '*' | grep beg)  $(find /home/alh216/UnwUnd/$g -type f -name '*' | grep csr) | tail -1 | echo -n $(grep -o -E [0-9]+.[0-9]+) >> results.csv
     cat prof.log | grep -m 1 Power >> power.txt
    done

    echo GPU $(cat prof.log | grep -m 1 Temperature)
    echo sleeping...
    sleep 5
    echo >> results.csv
    echo >> power.txt
    echo
   done

   blk=$(($blk * 2))
  done

  reg=$(($reg * 2))

  if [ $reg -eq 128 ]
  then
   reg=$(($reg * 4))
  fi
 done
done
