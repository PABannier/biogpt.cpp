#pragma once

#include <string>
#include <vector>

struct biogpt_vocab;

struct biogpt_hparams;

enum biogpt_ftype {
    BIOGPT_FTYPE_ALL_F32     = 0,
    BIOGPT_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q8_0 = 3,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q5_0 = 4,  // except 1d tensors
};


//
// Model
//

struct biogpt_layer_decoder;

struct biogpt_model;
