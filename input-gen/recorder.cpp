#include "common.h"
#include "timer.h"
#include "vm_choices.h"
#include "vm_obj.h"
#include "logging.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <string>
#include <string_view>
#include <thread>
#include <condition_variable>

extern "C" uint32_t __ig_num_entry_points;
extern "C" void __ig_entry(uint32_t, void *);

using namespace __ig;

thread_local ObjectManager ThreadOM;

struct SharedState {
  SharedState(uint32_t NumThreads, std::vector<uint32_t> &Seeds)
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
  std::vector<uint32_t> &Seeds;
};

static uint32_t ThreadID = 0;

struct GenerationThread {
  SharedState &SS;
  uint32_t ID, I, E, EntryNo;

  GenerationThread(SharedState &SS, uint32_t I, uint32_t E, uint32_t EntryNo)
      : SS(SS), ID(ThreadID++), I(I), E(E), EntryNo(EntryNo) {}

  static void start(SharedState *SS, std::string_view OutputName, uint32_t I,
                    uint32_t E, uint32_t EntryNo) {
    printf("thread generating %u inputs\n", E - I);
    auto *GT = new GenerationThread(*SS, I, E, EntryNo);
    auto *ChoiceTrace = SS->CM.initializeChoices(GT->ID);
    ThreadOM.init(ChoiceTrace, OutputName,
                  std::bind(&GenerationThread::stopGeneration, GT,
                            std::placeholders::_1));
    GT->startGeneration();
  }

  void startGeneration() {
    void *Obj;
    {
      Timer T("init " + std::to_string(I));
      assert(I < E);
      ThreadOM.setSeed(SS.getSeed(I));
      Obj = ThreadOM.getObj(I);
    }
    {
      Timer T("rec  " + std::to_string(I));
      __ig_entry(EntryNo, Obj);
    }
    stopGeneration(0);
  }

  void stopGeneration(uint32_t ExitCode) {
    {
      Timer T("save " + std::to_string(I));
      ThreadOM.saveInput(I, ExitCode);
    }

    if (++I < E) {
      printf("reset and restart %u of %u\n", I, E);
      if (SS.CM.returnChoices(ID)) {
        ThreadOM.reset();
        startGeneration();
      }
      printf("No more choices to explore!\n");
    }

    // Done
    SS.signalDone();
  }
};

int main(int argc, char **argv) {

  uint32_t Seed = 42;
  uint32_t EntryNo = 0;
  uint32_t NumInputs = 1;
  uint32_t FirstInput = 0;
  uint32_t NumThreads = 1;
  if (argc > 1)
    EntryNo = std::atoi(argv[1]);
  if (argc > 2)
    NumInputs = std::atoi(argv[2]);
  if (argc > 3)
    NumThreads = std::bit_floor((uint32_t)std::atoi(argv[3]));
  if (argc > 4)
    FirstInput = std::atoi(argv[4]);
  if (argc > 5)
    Seed = std::atoi(argv[5]);

  if (EntryNo >= __ig_num_entry_points) {
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

  SharedState SS(NumThreads, Seeds);
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
