#!/bin/bash

ONESAMP2COAL_MINALLELEFREQUENCY=0.05
mutationRate="0.000000012"
rangeNe=100,500
theta=0.000048,0.0048
microsatsOrSNPs=s
NeVals="00100"
numPOP="00256"

outputSampleSizes=(10 20 30)
locis=(1000 2000 4000 8000)

for outputSampleSize in "${outputSampleSizes[@]}"; do
  for loci in "${locis[@]}"; do
      ./refactor -t1 -rC -b$NeVals -d1 -u$mutationRate -v${theta} -$microsatsOrSNPs -l$loci -i$outputSampleSize -o1 -f$ONESAMP2COAL_MINALLELEFREQUENCY -p > "./data/genePop${outputSampleSize}x${loci}"
  done
done