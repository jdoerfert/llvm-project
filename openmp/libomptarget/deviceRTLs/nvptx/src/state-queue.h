//===--- state-queue.h --- OpenMP target state queue ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a queue to hand out OpenMP state objects to teams of one
// or more kernels.
//
// Reference:
// Thomas R.W. Scogland and Wu-chun Feng. 2015.
// Design and Evaluation of Scalable Concurrent Queues for Many-Core
// Architectures. International Conference on Performance Engineering.
//
//===----------------------------------------------------------------------===//

#ifndef STATE_QUEUE_H
#define STATE_QUEUE_H

#include "target_impl.h"

#include <stdint.h>

template <typename ElementType, uint32_t SIZE> class omptarget_state_queue {
  ElementType Elements[SIZE];
  ElementType *ElementQueue[SIZE];
  uint32_t Head;
  uint32_t Tail;
  uint32_t Ids[SIZE];

  static const uint32_t MAX_ID = (1u << 31) / SIZE / 2;
  INLINE uint32_t enqueueTicket();
  INLINE uint32_t dequeueTicket();
  INLINE static uint32_t getID(uint32_t Ticket);
  INLINE bool isServing(uint32_t Slot, uint32_t ID);
  INLINE void pushElement(uint32_t Slot, ElementType *element);
  INLINE ElementType *popElement(uint32_t Slot);
  INLINE void doneServing(uint32_t Slot, uint32_t ID);

public:
  INLINE void enqueue(ElementType *element);
  INLINE ElementType *dequeue();
};

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t omptarget_state_queue<ElementType, SIZE>::enqueueTicket() {
  return __kmpc_impl_atomic_add(&Tail, 1);
}

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t omptarget_state_queue<ElementType, SIZE>::dequeueTicket() {
  return __kmpc_impl_atomic_add(&Head, 1);
}

template <typename ElementType, uint32_t SIZE>
INLINE uint32_t
omptarget_state_queue<ElementType, SIZE>::getID(uint32_t Ticket) {
  return (Ticket / SIZE) * 2;
}

template <typename ElementType, uint32_t SIZE>
INLINE bool omptarget_state_queue<ElementType, SIZE>::isServing(uint32_t Slot,
                                                               uint32_t ID) {
  return __kmpc_impl_atomic_add(&Ids[Slot], 0) == ID;
}

template <typename ElementType, uint32_t SIZE>
INLINE void
omptarget_state_queue<ElementType, SIZE>::pushElement(uint32_t Slot,
                                                     ElementType *element) {
  __kmpc_impl_atomic_exchange(&ElementQueue[Slot], element);
}

template <typename ElementType, uint32_t SIZE>
INLINE ElementType *
omptarget_state_queue<ElementType, SIZE>::popElement(uint32_t Slot) {
  return (ElementType *)__kmpc_impl_atomic_add(&ElementQueue[Slot], 0);
}

template <typename ElementType, uint32_t SIZE>
INLINE void omptarget_state_queue<ElementType, SIZE>::doneServing(uint32_t Slot,
                                                                 uint32_t ID) {
  __kmpc_impl_atomic_exchange(&Ids[Slot], (ID + 1) % MAX_ID);
}

template <typename ElementType, uint32_t SIZE>
INLINE void
omptarget_state_queue<ElementType, SIZE>::enqueue(ElementType *element) {
  uint32_t Ticket = enqueueTicket();
  uint32_t Slot = Ticket % SIZE;
  uint32_t ID = getID(Ticket) + 1;
  while (!isServing(Slot, ID))
    ;
  pushElement(Slot, element);
  doneServing(Slot, ID);
}

template <typename ElementType, uint32_t SIZE>
INLINE ElementType *omptarget_state_queue<ElementType, SIZE>::dequeue() {
  uint32_t Ticket = dequeueTicket();
  uint32_t Slot = Ticket % SIZE;
  uint32_t ID = getID(Ticket);
  while (!isServing(Slot, ID))
    ;
  ElementType *element = popElement(Slot);
  // This is to populate the queue because of the lack of GPU constructors.
  if (element == 0)
    element = &Elements[Slot];
  doneServing(Slot, ID);
  return element;
}

#endif // STATE_QUEUE_H
