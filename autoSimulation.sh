#!/bin/bash

echo "Running full simulation"

echo "Compiling code"

make

echo "Generating data"

python pythonFiles/makeData.py TestData 100

echo "Running Simulation"

./bin/CUDAIntegrate TestData OutputData 10 0.01

echo "Generating Animation"

python pythonFiles/animateOutput.py OutputData -o Animations/ExampleAnim.mp4

echo "Cleaning Up"

rm TestData
# rm OutputData

echo "Complete"
