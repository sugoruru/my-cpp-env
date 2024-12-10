@echo off
set FILE=./main.cpp
if not "%1"=="" set FILE=%1

@g++ -DDEVELOPMENT %FILE% -std=c++17 -I ../../
@cat ./input.txt | a.exe > ./output.txt
