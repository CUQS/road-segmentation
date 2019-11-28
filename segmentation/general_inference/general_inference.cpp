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

#include "general_inference.h"

#include <vector>
#include <sstream>

#include "hiaiengine/log.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using hiai::ImageData;
using namespace std;
using namespace ascend::utils;

namespace {

// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// level for call DVPP
const int32_t kDvppToJpegLevel = 100;

// call dvpp success
const uint32_t kDvppProcSuccess = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// length of image info array
const uint32_t kImageInfoLength = 3;
}

// register custom data type
HIAI_REGISTER_DATA_TYPE("Output", Output);
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

GeneralInference::GeneralInference() {
  ai_model_manager_ = nullptr;
}

HIAI_StatusT GeneralInference::Init(
    const hiai::AIConfig& config,
    const vector<hiai::AIModelDescription>& model_desc) {
  return HIAI_OK;
}

bool GeneralInference::PreProcess(const shared_ptr<EngineTrans> &image_handle,
                                  ImageData<u_int8_t> &resized_image) {
  // call ez_dvpp to resize image
  DvppBasicVpcPara resize_para;
  resize_para.input_image_type = INPUT_YUV420_SEMI_PLANNER_UV;

  // get original image size and set to resize parameter
  int32_t width = image_handle->image_info.width;
  int32_t height = image_handle->image_info.height;

  // set source resolution ratio
  resize_para.src_resolution.width = width;
  resize_para.src_resolution.height = height;

  // crop parameters, only resize, no need crop, so set original image size
  // set crop left-top point (need even number)
  resize_para.crop_left = 0;
  resize_para.crop_up = 0;
  // set crop right-bottom point (need odd number)
  uint32_t crop_right = ((width >> 1) << 1) - 1;
  uint32_t crop_down = ((height >> 1) << 1) - 1;
  resize_para.crop_right = crop_right;
  resize_para.crop_down = crop_down;

  // set destination resolution ratio (need even number)
  uint32_t dst_width = ((image_handle->console_params.model_width) >> 1) << 1;
  uint32_t dst_height = ((image_handle->console_params.model_height) >> 1) << 1;
  resize_para.dest_resolution.width = dst_width;
  resize_para.dest_resolution.height = dst_height;

  // set input image align or not
  resize_para.is_input_align = true;

  // call
  DvppProcess dvpp_resize_img(resize_para);
  DvppVpcOutput dvpp_output;
  int ret = dvpp_resize_img.DvppBasicVpcProc(
      image_handle->image_info.data.get(), image_handle->image_info.size,
      &dvpp_output);
  if (ret != kDvppOperationOk) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call ez_dvpp failed, failed to resize image.");
    return false;
  }

  // call success, set data and size
  resized_image.data.reset(dvpp_output.buffer, default_delete<u_int8_t[]>());
  resized_image.size = dvpp_output.size;
  resized_image.width = dst_width;
  resized_image.height = dst_height;
  image_handle->image_info.width = dst_width;
  image_handle->image_info.height = dst_height;
  return true;
}

HIAI_StatusT GeneralInference::ConvertImage(const std::shared_ptr<EngineTrans> &image_handle) {
  uint32_t width = image_handle->image_info.width;
  uint32_t height = image_handle->image_info.height;
  uint32_t img_size = image_handle->image_info.size;
  // parameter
  ascend::utils::DvppToJpgPara dvpp_to_jpeg_para;
  dvpp_to_jpeg_para.format = JPGENC_FORMAT_NV12;
  dvpp_to_jpeg_para.level = kDvppToJpegLevel;
  dvpp_to_jpeg_para.resolution.height = height;
  dvpp_to_jpeg_para.resolution.width = width;
  ascend::utils::DvppProcess dvpp_to_jpeg(dvpp_to_jpeg_para);
  // call DVPP
  ascend::utils::DvppOutput dvpp_output;
  int32_t ret = dvpp_to_jpeg.DvppOperationProc(reinterpret_cast<char*>(image_handle->image_info.data.get()),
                                                img_size, &dvpp_output);

  // failed, no need to send to presenter
  if (ret != kDvppProcSuccess) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "Failed to convert YUV420SP to JPEG, skip it.");
    return HIAI_ERROR;
  }
  image_handle->image_info.data.reset(dvpp_output.buffer, default_delete<uint8_t[]>());
  image_handle->image_info.size = dvpp_output.size;
  return HIAI_OK;
}

bool GeneralInference::SendToEngine(
    const shared_ptr<EngineTrans> &image_handle) {
  // can not discard when queue full
  HIAI_StatusT hiai_ret;
  do {
    hiai_ret = SendData(kSendDataPort, "EngineTrans",
                        static_pointer_cast<void>(image_handle));
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

void GeneralInference::SendError(const std::string &err_msg,
                                 std::shared_ptr<EngineTrans> &image_handle) {
  image_handle->err_msg.error = true;
  image_handle->err_msg.err_msg = err_msg;
  if (!SendToEngine(image_handle)) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "SendData err_msg failed");
  }
}

HIAI_IMPL_ENGINE_PROCESS("general_inference",
    GeneralInference, INPUT_SIZE) {
  HIAI_StatusT ret = HIAI_OK;

  // arg0 is empty
  if (arg0 == nullptr) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "arg0 is empty.");
    return HIAI_ERROR;
  }

  // just send data when finished
  shared_ptr<EngineTrans> image_handle = static_pointer_cast<EngineTrans>(arg0);
  if (image_handle->is_finished) {
    cout << "image_handle is finished" << endl;
    if (SendToEngine(image_handle)) {
      return HIAI_OK;
    }
    SendError("Failed to send finish data. Reason: Inference SendData failed.",
              image_handle);
    return HIAI_ERROR;
  }

  // resize image
  cout << "--inference-- resize image" << endl;
  ImageData<u_int8_t> resized_image;
  if (!PreProcess(image_handle, resized_image)) {
    string err_msg = "Failed to deal file=" + image_handle->image_info.path
        + ". Reason: resize image failed.";
    SendError(err_msg, image_handle);
    return HIAI_ERROR;
  }

  // convert original image to JPEG
  cout << "--inference-- convert to JPEG" << endl;
  HIAI_StatusT convert_ret = ConvertImage(image_handle);
  if (convert_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                  "Convert YUV Image to Jpeg failed!");
    return HIAI_ERROR;
  }

  // send result
  cout << "--inference-- send to post engine" << endl;
  if (!SendToEngine(image_handle)) {
    string err_msg = "Failed to deal file=" + image_handle->image_info.path
        + ". Reason: Inference SendData failed.";
    SendError(err_msg, image_handle);
    return HIAI_ERROR;
  }
  return HIAI_OK;
}
