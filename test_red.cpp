#include<iostream>

#define DTYPE long

int main (int argc, char *argv[])
{
  long Elements = std::atol(argv[1]) * 1024L;
  DTYPE  *Data = new DTYPE[Elements];

#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements])
  for (long I = 0 ; I < Elements; I++)
    Data[I] = I - (Elements / 2);

  DTYPE Sum = 0;
#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements]) reduction(+:Sum)
  for (long I = 0 ; I < Elements; I++)
    Sum += Data[I];

  std::cout<< "Computed :" << Sum << " Correct: " << (-Elements/2) << "\n";

  delete [] Data;
  return 0;
}
