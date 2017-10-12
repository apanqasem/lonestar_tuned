#!/bin/bash
num=128

sed -i -E "s/THDS_NUM\t+[0-9]+/THDS_NUM\t\t\t$num/g" comm.h

cat comm.h
