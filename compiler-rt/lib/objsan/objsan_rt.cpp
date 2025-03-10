#include "include/obj_encoding.h"

__attribute__((
    visibility("default"))) __objsan::SmallObjectsTy __objsan::SmallObjects;

__attribute__((
    visibility("default"))) __objsan::LargeObjectsTy __objsan::LargeObjects;

__attribute__((
    visibility("default"))) __objsan::FixedObjectsTy __objsan::FixedObjects;

extern "C" {}
