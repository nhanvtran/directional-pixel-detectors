#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 14
#define N_LAYER_2 128
#define N_LAYER_6 3

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<5,1> weight2_t;
typedef ap_fixed<5,1> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<18,8> dense1_lineartable_t;
typedef ap_fixed<16,6,AP_RND_CONV,AP_SAT> layer3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef ap_ufixed<10,0,AP_RND_CONV,AP_SAT> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<5,1> weight6_t;
typedef ap_fixed<5,1> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<18,8> dense2_lineartable_t;
typedef ap_fixed<16,6,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<16,6> softmax_default_t;
typedef ap_fixed<18,8> softmaxexp_table_t;
typedef ap_fixed<18,4> softmaxinv_table_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> softmax_table_t;

#endif
