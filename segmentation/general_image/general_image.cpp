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

#include "general_image.h"

#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using namespace std;

namespace {
// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// get stat success
const int kStatSuccess = 0;
// image file path split character
const string kImagePathSeparator = ",";
// path separator
const string kPathSeparator = "/";

}

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

HIAI_StatusT GeneralImage::Init(
    const hiai::AIConfig& config,
    const vector<hiai::AIModelDescription>& model_desc) {
  // do noting
  return HIAI_OK;
}

bool GeneralImage::ArrangeImageInfo(shared_ptr<EngineTrans> &image_handle,
                                    const string &image_path) {
  // read image using OPENCV
  cv::Mat mat = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  if (mat.empty()) {
    ERROR_LOG("Failed to deal file=%s. Reason: read image failed.",
              image_path.c_str());
    return false;
  }

  // set property
  image_handle->image_info.path = image_path;
  image_handle->image_info.width = mat.cols;
  image_handle->image_info.height = mat.rows;

  // set image data
  cout << "mat.total(): " << mat.total() << endl;
  uint32_t size = mat.total() * mat.channels();
  u_int8_t *image_buf_ptr = new (nothrow) u_int8_t[size];
  if (image_buf_ptr == nullptr) {
    HIAI_ENGINE_LOG("new image buffer failed, size=%d!", size);
    ERROR_LOG("Failed to deal file=%s. Reason: new image buffer failed.",
              image_path.c_str());
    return false;
  }
  cout << "copy mat from image" << endl;
  error_t mem_ret = memcpy_s(image_buf_ptr, size, mat.ptr<u_int8_t>(),
                             mat.total() * mat.channels());
  if (mem_ret != EOK) {
    cout << "copy mat from image failed" << endl;
    delete[] image_buf_ptr;
    ERROR_LOG("Failed to deal file=%s. Reason: memcpy_s failed.",
              image_path.c_str());
    image_buf_ptr = nullptr;
    return false;
  }

  image_handle->image_info.size = size;
  image_handle->image_info.data.reset(image_buf_ptr,
                                      default_delete<u_int8_t[]>());
  return true;
}

bool GeneralImage::SendToEngine(const shared_ptr<EngineTrans> &image_handle) {
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

HIAI_IMPL_ENGINE_PROCESS("general_image",
    GeneralImage, INPUT_SIZE) {
      
  string path = "test.png";
  // Step3: send every image to inference engine
  shared_ptr<EngineTrans> image_handle = nullptr;
  MAKE_SHARED_NO_THROW(image_handle, EngineTrans);
  if (image_handle == nullptr) {
    ERROR_LOG("Failed to deal file=%s. Reason: new EngineTrans failed.",
              path.c_str());
    return HIAI_ERROR;
  }
  // arrange image information, if failed, skip this image
  if (!ArrangeImageInfo(image_handle, path)) {
    return HIAI_ERROR;
  }

  // send data to inference engine
  image_handle->console_params.input_path = "test.png";
  image_handle->console_params.model_height = 188;
  image_handle->console_params.model_width = 623;
  image_handle->console_params.output_path = "./";
  if (!SendToEngine(image_handle)) {
    ERROR_LOG("Failed to deal file=%s. Reason: send data failed.",
              path.c_str());
    return HIAI_ERROR;
  }


  // Step4: send finished data
  shared_ptr<EngineTrans> image_handle2 = nullptr;
  MAKE_SHARED_NO_THROW(image_handle2, EngineTrans);
  if (image_handle2 == nullptr) {
    ERROR_LOG("Failed to send finish data. Reason: new EngineTrans failed.");
    ERROR_LOG("Please stop this process manually.");
    return HIAI_ERROR;
  }

  image_handle2->is_finished = true;

  if (SendToEngine(image_handle2)) {
    return HIAI_OK;
  }
  ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
  ERROR_LOG("Please stop this process manually.");
  return HIAI_ERROR;
}
