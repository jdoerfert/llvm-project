#include<iostream>

#define DTYPE long

int main (int argc, char *argv[])
{
  long Elements = std::atol(argv[1]) * 1024L;
  DTYPE  *Data = new DTYPE[Elements];

#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements])
  for (long I = 0 ; I < Elements; I++)
    Data[I] = I - (Elements / 2);

  DTYPE Sum1 = 0, Sum2 = 3;
#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements]) reduction(+:Sum1, Sum2)
  for (long I = 0; I < Elements; I++) {
    Sum1 += Data[I];
    Sum2 += Data[I];
  }

  std::cout << "Computed :" << Sum1 << " Correct: " << (-Elements / 2) << "\n";
  std::cout << "Computed :" << Sum2 << " Correct: " << (-Elements / 2) + 3
            << "\n";

  delete [] Data;
  return 0;
}
