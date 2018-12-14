#!/bin/bash
for i in {1..4}
do
	julia clang.jl $i 4 &
done
wait
echo "All done"
