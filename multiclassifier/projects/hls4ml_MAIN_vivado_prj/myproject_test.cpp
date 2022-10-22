//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert bram

#define CHECKPOINT 1

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

template <class T, int N>
unsigned argmax(T args[N]) {
  unsigned max_idx = 0;
  T max = -31;

  for (int i = N-1; i >= 0; i--) {
    if(args[i] > max) {
      max = args[i];
      max_idx = i;
    }
  }
  return max_idx;
}

template <int N>
unsigned argmax_vector(std::vector<float> args) {
  unsigned max_idx = 0;
  float max = std::numeric_limits<float>::min();
  for (unsigned i = 0; i < N; i++) {
    if(args[i] > max) {
      max = args[i];
      max_idx = i;
    }
  }
  return max_idx;
}


int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;
  unsigned errors = 0;
  unsigned outputs = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "INFO: Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      input_t input_1[N_INPUT_1_1];
      nnet::copy_data<float, input_t, 0, N_INPUT_1_1>(in, input_1);
      result_t layer8_out[N_LAYER_6];

      //hls-fpga-machine-learning insert top-level-function
      myproject(input_1,layer8_out);

      if (e % CHECKPOINT == 0) {
        std::cout << "INFO: Predictions: ";
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_6; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "INFO: Quantized predictions: ";
        //hls-fpga-machine-learning insert quantized
        nnet::print_result<result_t, N_LAYER_6>(layer8_out, std::cout, true);
      }
      e++;

      unsigned expected_prediction = argmax_vector<N_LAYER_6>(pr);
      unsigned model_prediction = argmax<result_t, N_LAYER_6>(layer8_out);
      std::cout << "INFO: Expected prediction (argmax): " << expected_prediction << std::endl;
      std::cout << "INFO: Model prediction (argmax): " << model_prediction << std::endl;
      if (model_prediction != expected_prediction) errors++;
      outputs++;
      std::cout << "INFO:" << std::endl;
      //hls-fpga-machine-learning insert tb-output
      nnet::print_result<result_t, N_LAYER_6>(layer8_out, fout);
      if (e > 5) break;
    }
    float error_rate = float(errors) * 100 / outputs;
    std::cout << "INFO: Total errors: " << errors << " (" << outputs << ")" << std::endl;
    std::cout << "INFO: Error rate: " << error_rate << "%" << std::endl;
    std::cout << "INFO: Accuracy: " << 100 - error_rate << "%" << std::endl;

    fin.close();
    fpr.close();
  } else {
    std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;

    //hls-fpga-machine-learning insert zero
    input_t input_1[N_INPUT_1_1];
    nnet::fill_zero<input_t, N_INPUT_1_1>(input_1);
    result_t layer8_out[N_LAYER_6];

    //hls-fpga-machine-learning insert top-level-function
    myproject(input_1,layer8_out);

    //hls-fpga-machine-learning insert output
    nnet::print_result<result_t, N_LAYER_6>(layer8_out, std::cout, true);

    //hls-fpga-machine-learning insert tb-output
    nnet::print_result<result_t, N_LAYER_6>(layer8_out, fout);

  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
