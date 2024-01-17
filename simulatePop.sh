#!/bin/bash

ONESAMP2COAL_MINALLELEFREQUENCY=0.05
mutationRate="0.000000012"
rangeNe=100,500
theta=0.000048,0.0048
microsatsOrSNPs=s
NeVals="00200"
numPOP="00256"

outputSampleSizes=(40 50 100)
locis=(40 80 160)

for outputSampleSize in "${outputSampleSizes[@]}"; do
  for loci in "${locis[@]}"; do
      ./refactor -t1 -rC -b$NeVals -d1 -u$mutationRate -v${theta} -$microsatsOrSNPs -l$loci -i$outputSampleSize -o1 -f$ONESAMP2COAL_MINALLELEFREQUENCY -p > "./exampleData/genePop${outputSampleSize}x${loci}"
  done
done
