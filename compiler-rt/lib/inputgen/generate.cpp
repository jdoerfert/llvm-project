#include "common.h"
#include "defer.h"
#include "logging.h"
#include "timer.h"
#include "vm_choices.h"
#include "vm_obj.h"

#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <string>
#include <string_view>
#include <thread>

// We are depending on this = false to be embedded as an initial value in the
// global of the binary so that we have it set to false before we get the call
// to __ig_pre_module. It seems to work in the current state (but may break if
// optimizations are off? Or does it work because the memory is initialized to
// 0 by default?)
//
// In the current inputgen we assume we instrument for generation only a
// single module so we should get exactly one call to init, so in theory we
// don't need it.
namespace __ig {
bool GM = false;
DeferGlobalConstruction<ObjectManager, GM> ThreadOM;
uint32_t OriginalSeed;
} // namespace __ig

using namespace __ig;

struct SharedState {
  SharedState(uint32_t NumThreads, std::span<uint32_t> Seeds)
      : CM(NumThreads), Counter(0), NumThreads(NumThreads), Seeds(Seeds) {}

  ChoiceManager CM;
  std::mt19937 Generator;

  void signalDone() {
    fprintf(stderr, "done\n");
    std::unique_lock Lock(Mutex);
    fprintf(stderr, "done %u, %u\n", Counter, NumThreads);
    if (++Counter == NumThreads)
      exit(0);
    FinishedCV.wait(Lock);
    fprintf(stderr, "never\n");
  }
  uint32_t getSeed(uint32_t Idx) { return Seeds[Idx]; }

  uint32_t Counter, NumThreads;
  std::mutex Mutex;

  std::condition_variable FinishedCV;
  std::span<uint32_t> Seeds;
};

static uint32_t ThreadID = 0;

struct GenerationThread {
  SharedState &SS;
  uint32_t ID, I, E, EntryNo;

  GenerationThread(SharedState &SS, uint32_t I, uint32_t E, uint32_t EntryNo)
      : SS(SS), ID(ThreadID++), I(I), E(E), EntryNo(EntryNo) {}

  static void start(SharedState *SS, std::string_view OutputName, uint32_t I,
                    uint32_t E, uint32_t EntryNo) {
    fprintf(stderr, "thread generating %u inputs\n", E - I);
    auto *GT = new GenerationThread(*SS, I, E, EntryNo);
    auto *ChoiceTrace = SS->CM.initializeChoices(GT->ID);
    ThreadOM->init(ChoiceTrace, OutputName,
                   std::bind(&GenerationThread::stopGeneration, GT,
                             std::placeholders::_1));
    GT->startGeneration();
  }

  void startGeneration() {
    void *Obj;
    {
      Timer T("init " + std::to_string(I));
      assert(I < E);
      ThreadOM->setSeed(SS.getSeed(I));
      Obj = ThreadOM->getEntryObj();
    }
    {
      Timer T("genr " + std::to_string(I));
      __ig_entry(EntryNo, Obj);
    }
    stopGeneration(0);
  }

  void stopGeneration(uint32_t ExitCode) {
    {
      Timer T("save " + std::to_string(I));
      ThreadOM->saveInput(EntryNo, I, ExitCode);
    }

    if (++I < E) {
      fprintf(stderr, "reset and restart %u of %u\n", I, E);
      if (SS.CM.returnChoices(ID)) {
        ThreadOM->reset();
        startGeneration();
      }
      fprintf(stderr, "No more choices to explore!\n");
    }

    // Done
    SS.signalDone();
  }
};

int main(int argc, char **argv) {

  uint32_t Seed = 42;
  int32_t EntryNo = 0;
  uint32_t NumInputs = 1;
  uint32_t FirstInput = 0;
  uint32_t NumThreads = 1;
  if (argc > 1)
    EntryNo = std::atoi(argv[1]);
  if (argc > 2)
    NumInputs = std::atoi(argv[2]);
  // TODO Threading disabled for now. We need to make sure each thread gets its
  // own deep copy of ThreadOM (i.e. also reallocate and copy over any dynamic
  // memory in there)
  if (false && argc > 3)
    NumThreads = std::bit_floor((uint32_t)std::atoi(argv[3]));
  if (argc > 4)
    FirstInput = std::atoi(argv[4]);
  if (argc > 5)
    Seed = std::atoi(argv[5]);

  OriginalSeed = Seed;

  if (EntryNo == -1) {
    printNumAvailableFunctions();
    printAvailableFunctions();
    exit(static_cast<int>(ExitStatus::WrongUsage));
  }

  if (static_cast<uint32_t>(EntryNo) >= __ig_num_entry_points) {
    ERR("Entry {} is out of bounds, {} available\n", EntryNo,
        __ig_num_entry_points);
    exit(static_cast<int>(ExitStatus::EntryNoOutOfBounds));
  }

  NumThreads = std::min(NumThreads, NumInputs);

  uint32_t NumInputsPerThread = NumInputs / NumThreads;
  INFO("Generating {} inputs for entry {} with {} threads, starting with {}; "
       "Seed: {}\n",
       NumInputs, EntryNo, NumThreads, FirstInput, Seed);

  std::mt19937 Generator(Seed);
  std::vector<uint32_t> Seeds;
  for (uint32_t I = 0; I < NumInputs; ++I)
    Seeds.push_back(Generator());
  std::span<uint32_t> SeedsSpan{Seeds.data() - FirstInput,
                                FirstInput + NumInputs};

  SharedState SS(NumThreads, SeedsSpan);
  for (uint32_t I = FirstInput; I < NumInputs + FirstInput - NumInputsPerThread;
       I += NumInputsPerThread) {
    new std::thread(GenerationThread::start, &SS, argv[0], I,
                    I + NumInputsPerThread, EntryNo);
  }
  GenerationThread::start(&SS, argv[0],
                          NumInputs + FirstInput - NumInputsPerThread,
                          NumInputs + FirstInput, EntryNo);

  fprintf(stderr, "No Inputs to process\n");
  return static_cast<int>(ExitStatus::NoInputs);
}
