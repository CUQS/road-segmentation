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

// namespace
namespace {
  // callback port (engine port begin with 0)
  const uint32_t kSendDataPort = 0;

  // sleep interval when queue full (unit:microseconds)
  const __useconds_t kSleepInterval = 200000;

  // size of output tensor vector should be 1.
  const uint32_t kOutputTensorSize = 1;

  // output image prefix
  const string kOutputFilePrefix = "out_";

  // output image tensor shape  623*188
  const static std::vector<uint32_t> kDimImageOutput = {117124, 2};

  const string kFileSperator = "/";
}

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

HIAI_StatusT GeneralPost::Init(
  const hiai::AIConfig &config,
  const vector<hiai::AIModelDescription> &model_desc) {
  addrLen = sizeof(struct sockaddr_in);
  serverAddr.sin_family = PF_INET;

  for (int index = 0; index < config.items_size(); ++index) {
    const ::hiai::AIConfigItem& item = config.items(index);
    string name = item.name();
    string value = item.value();

    if (name == "serverIP") {
      serverAddr.sin_addr.s_addr = inet_addr(value.data());
      cout << "--post-- serverIP: " << value.data() << endl;
    } else if (name == "serverPort") {
      int serverPort = atoi(value.data());
      serverAddr.sin_port = htons(serverPort);
      cout << "--post-- serverPort: " << serverPort << endl;
    } else {
      HIAI_ENGINE_LOG("unused config name: %s", name.c_str());
    }
  }
  if ((sokt = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    cout << "--post-- socket() failed" << endl;
    return HIAI_ERROR;
  }
  if (connect(sokt, (sockaddr*)&serverAddr, addrLen) < 0) {
    cout << "--post-- connect() failed!" << endl;
    cout << "--post-- close socket" << endl;
    close(sokt);
  }
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

HIAI_StatusT GeneralPost::ModelPostProcessCap(const shared_ptr<EngineTrans> &result) {

  vector<Output> outputs = result->inference_res;
  
  if (outputs.size() != kOutputTensorSize) {
    ERROR_LOG("Detection output size does not match.");
    return HIAI_ERROR;
  }
  // cout << "--post-- start get outputs" << endl;
  float *img_output = reinterpret_cast<float *>(outputs[0].data.get());
  // cout << "--post-- convert outputs" << endl;
  Tensor<float> tensor_imgoutput;
  bool ret = true;
  ret = tensor_imgoutput.FromArray(img_output, kDimImageOutput);
  if (!ret) {
    ERROR_LOG("Failed to resolve tensor from array.");
    return HIAI_ERROR;
  }
  // cout << "--post-- get outputs" << endl;
  // cout << "--post-- unsigned char to mat" << endl;
  uint8_t* pdata = result->image_info.data.get();
  cv::Mat yuvImg;
  yuvImg.create(result->image_info.height*3/2, result->image_info.width, CV_8UC1);
  memcpy(yuvImg.data, pdata, result->image_info.size);
  cv::Mat mat;
  cv::cvtColor(yuvImg, mat, CV_YUV2RGB_NV21);
  // crop image
  cv::Rect rect(0,172,1246,376);
  cv::Mat imageCrop = mat(rect);
  // resize iamge
  cv::resize(imageCrop, imageCrop, cv::Size(623, 188));
  stringstream sstream;

  // cout << "--post-- start mat change" << endl;
  cv::Vec3b pVec3b;
  for (int i = 0; i < 188; i++) {
    for (int j = 0; j < 623; j++) {
      float resultValue = tensor_imgoutput(i*623+j, 0)*255.0;
      cv::Vec3b pNow = imageCrop.at<cv::Vec3b>(i, j);
      pVec3b[0] = (int) (0.4*resultValue+0.6*pNow[0]);
      pVec3b[1] = pNow[1];
      pVec3b[2] = (int) (0.4*(255.0-resultValue)+0.6*pNow[2]);
      if (pVec3b[0]>255) pVec3b[0]=255;
      if (pVec3b[1]>255) pVec3b[1]=255;
      if (pVec3b[2]>255) pVec3b[2]=255;
      imageCrop.at<cv::Vec3b>(i, j) = pVec3b;
    }
  }
  // cout << "--post-- mat changed!!" << endl;
  int bytes = 0;
  int image_size = imageCrop.total() * imageCrop.elemSize();
  // cout << "--post-- send image to server, image_size: " << image_size << endl;
  if ((bytes = send(sokt, imageCrop.data, image_size, 0)) < 0){
    close(sokt);
    cout << "bytes = " << bytes << endl;
  }
  return HIAI_OK;
}

HIAI_StatusT GeneralPost::ModelPostProcessPic(const shared_ptr<EngineTrans> &result) {

  vector<Output> outputs = result->inference_res;
  
  if (outputs.size() != kOutputTensorSize) {
    ERROR_LOG("Detection output size does not match.");
    return HIAI_ERROR;
  }
  // cout << "start get outputs" << endl;
  float *img_output = reinterpret_cast<float *>(outputs[0].data.get());
  // cout << "convert outputs" << endl;
  Tensor<float> tensor_imgoutput;
  bool ret = true;
  ret = tensor_imgoutput.FromArray(img_output, kDimImageOutput);
  if (!ret) {
    ERROR_LOG("Failed to resolve tensor from array.");
    return HIAI_ERROR;
  }
  // cout << "get outputs" << endl;

  cv::Mat mat = cv::imread(result->image_info.path, CV_LOAD_IMAGE_UNCHANGED);
  stringstream sstream;

  // cout << "start mat change!!" << endl;
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
  // cout << "mat changed!!" << endl;
  int bytes = 0;
  int image_size = mat.total() * mat.elemSize();
  // cout << "--post-- send image to server, image_size: " << image_size << endl;
  if ((bytes = send(sokt, mat.data, image_size, 0)) < 0){
    close(sokt);
    cout << "bytes = " << bytes << endl;
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
    cout << "--post-- finished" << endl;
    close(sokt);
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
  if (result->image_info.mode==0) {
    return ModelPostProcessCap(result);
  }
  else {
    return ModelPostProcessPic(result);
  }
  return HIAI_ERROR;
}
