#ifndef VM_CHOICES_H
#define VM_CHOICES_H

#include "logging.h"

#include <bit>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

namespace __ig {

struct ChoiceTrace {
  uint32_t MaxRecordedChoices = 0;
  uint32_t CurrentChoice = 0;

  std::bitset<128> Decisions;
  std::bitset<128> ChoicesToMake;

  ChoiceTrace(uint32_t I, uint32_t LastChoice)
      : Decisions(I), ChoicesToMake(LastChoice) {
    ChoicesToMake.flip();
  }

  bool addBooleanChoice(uint32_t ChoiceNo) {
    if (ChoiceNo >= Decisions.size()) {
      fprintf(stderr, "Run out of choice space!\n");
      __builtin_trap();
    }

    MaxRecordedChoices = std::max(MaxRecordedChoices, ChoiceNo);
    if (ChoiceNo > CurrentChoice)
      ChoicesToMake.set(ChoiceNo);
    return Decisions[ChoiceNo];
  }
};

struct ChoiceManager {
  std::vector<ChoiceTrace *> Choices;

  ChoiceManager(uint32_t NumThreads) : LastChoice(NumThreads - 1) {}
  int32_t LastChoice;

  ChoiceTrace *initializeChoices(uint32_t I) {
    auto *CT = new ChoiceTrace(I, LastChoice);
    INPUTGEN_DEBUG({
      std::cerr << "INITIAL CHOICES: " << CT->Decisions << "\n";
      std::cerr << "CHOICES TO MAKE: " << CT->ChoicesToMake << "\n";
    });
    Choices.push_back(CT);
    return CT;
  }

  bool returnChoices(uint32_t I) {
    ChoiceTrace *CT = Choices[I];
    uint32_t ChoiceToFlip = -1u;
    for (int32_t I = CT->MaxRecordedChoices; I >= 0; --I) {
      if (CT->ChoicesToMake[I]) {
        ChoiceToFlip = I;
        break;
      }
    }

    if (ChoiceToFlip == -1u)
      return false;
    INPUTGEN_DEBUG({ printf("Flip %u\n", ChoiceToFlip); });
    CT->Decisions.flip(ChoiceToFlip);
    CT->ChoicesToMake.flip(ChoiceToFlip);
    for (int32_t I = ChoiceToFlip + 1, E = CT->ChoicesToMake.size(); I < E; ++I)
      CT->ChoicesToMake.set(I);
    CT->CurrentChoice = ChoiceToFlip;
    INPUTGEN_DEBUG({ std::cout << "CHOICES: " << CT->Decisions << "\n"; });
    return true;
  }
};

} // namespace __ig

#endif
