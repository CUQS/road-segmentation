/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "general_post.h"

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using namespace std;

namespace {
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// size of output tensor vector should be 1.
const uint32_t kOutputTensorSize = 1;

// output image prefix
const string kOutputFilePrefix = "out_";

// output image tensor shape
const static std::vector<uint32_t> kDimImageOutput = {117124, 2};

// opencv draw label params.
const string kFileSperator = "/";

}

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

HIAI_StatusT GeneralPost::Init(
  const hiai::AIConfig &config,
  const vector<hiai::AIModelDescription> &model_desc) {
  // do noting
  return HIAI_OK;
}

bool GeneralPost::SendSentinel() {
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string",
                        static_pointer_cast<void>(sentinel_msg));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}

HIAI_StatusT GeneralPost::ModelPostProcess(const shared_ptr<EngineTrans> &result) {

  vector<Output> outputs = result->inference_res;
  
  if (outputs.size() != kOutputTensorSize) {
    ERROR_LOG("Detection output size does not match.");
    return HIAI_ERROR;
  }

  /* -----------------------------------------------------
   * image output
   * -----------------------------------------------------
  **/
  cout << "start get outputs" << endl;
  float *img_output = reinterpret_cast<float *>(outputs[0].data.get());
  cout << "convert outputs" << endl;
  Tensor<float> tensor_imgoutput;
  bool ret = true;
  ret = tensor_imgoutput.FromArray(img_output, kDimImageOutput);
  if (!ret) {
    ERROR_LOG("Failed to resolve tensor from array.");
    return HIAI_ERROR;
  }
  cout << "get outputs" << endl;
  /* -----------------------------------------------------
   * image output
   * -----------------------------------------------------
  **/

  cv::Mat mat = cv::imread(result->image_info.path, CV_LOAD_IMAGE_UNCHANGED);
  stringstream sstream;
  /* -----------------------------------------------------
   * image output
   * -----------------------------------------------------
  **/
  cout << "start mat change!!" << endl;
  cv::Vec3b pVec3b;
  for (int i = 0; i < 188; i++) {
    for (int j = 0; j < 623; j++) {
      float resultValue = tensor_imgoutput(i*623+j, 0)*255.0;
      cv::Vec3b pNow = mat.at<cv::Vec3b>(i, j);
      pVec3b[0] = (int) (0.4*resultValue+0.6*pNow[0]);
      pVec3b[1] = pNow[1];
      pVec3b[2] = (int) (0.4*(255.0-resultValue)+0.6*pNow[2]);
      if (pVec3b[0]>255) pVec3b[0]=255;
      if (pVec3b[1]>255) pVec3b[1]=255;
      if (pVec3b[2]>255) pVec3b[2]=255;
      mat.at<cv::Vec3b>(i, j) = pVec3b;
    }
  }
  cout << "mat changed!!" << endl;
  /* -----------------------------------------------------
   * image output
   * -----------------------------------------------------
  **/

  int pos = result->image_info.path.find_last_of(kFileSperator);
  string file_name(result->image_info.path.substr(pos + 1));
  bool save_ret(true);
  sstream.str("");
  sstream << result->console_params.output_path << kFileSperator
          << kOutputFilePrefix << file_name;
  string output_path = sstream.str();
  save_ret = cv::imwrite(output_path, mat);
  if (!save_ret) {
    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.",
              result->image_info.path.c_str());
    return HIAI_ERROR;
  }
  
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // check arg0
  if (arg0 == nullptr) {
    ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
    return HIAI_ERROR;
  }

  // just send to callback function when finished
  shared_ptr<EngineTrans> result = static_pointer_cast<EngineTrans>(arg0);
  if (result->is_finished) {
    if (SendSentinel()) {
      return HIAI_OK;
    }
    ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
    ERROR_LOG("Please stop this process manually.");
    return HIAI_ERROR;
  }

  // inference failed
  if (result->err_msg.error) {
    ERROR_LOG("%s", result->err_msg.err_msg.c_str());
    return HIAI_ERROR;
  }

  // arrange result
  return ModelPostProcess(result);
}
