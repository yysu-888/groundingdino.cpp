#pragma once
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>

#include <chrono>
#include <string>
#include "bert_tokenizer.hpp"
#include "string_utility.hpp"

using namespace cv;
using namespace std;
using namespace Ort;

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) printf("%s:%lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

inline std::string to_byte_string(const std::wstring& input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(input);
}

struct Object {
    cv::Rect box;
    string text;
    float prob;
};

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

template<class T>
void print_shape(std::vector<T>& v,std::string name){
    std::cout<<name<<" shape:";
    for(auto& x:v) std::cout<<x<<" ";
    std::cout<<std::endl;
}

class GroundingDINO {
public:
    GroundingDINO(string modelpath, float box_threshold, float text_threshold, bool with_logits);
    vector<Object> detect(Mat& srcimg, string text_prompt);

private:
    void img_process(Mat& img);
    void pre_process(Mat& srcimg, string text_prompt);
    void infer();
    vector<Object> post_process();

    const float mean[3] = {0.485, 0.456, 0.406};
    const float std[3]  = {0.229, 0.224, 0.225};
    const int size[2]   = {600, 400};

    std::shared_ptr<bert_tokenizer::BertTokenizer> tokenizer;

    std::vector<float> input_img;
    std::vector<std::vector<int64>> input_ids;
    std::vector<std::vector<uint8_t>> attention_mask;
    std::vector<std::vector<int64>> token_type_ids;
    std::vector<std::vector<uint8_t>> text_self_attention_masks;
    std::vector<std::vector<int64>> position_ids;

    Env env;
    Ort::Session* ort_session;
    SessionOptions sessionOptions;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    vector<Value> ort_outputs;

    int srch, srcw;
    float box_threshold;
    float text_threshold;
    bool with_logits;
    const int max_text_len        = 256;
    const char* specical_texts[4] = {"[CLS]", "[SEP]", ".", "?"};
    vector<int64> specical_tokens = {101, 102, 1012, 1029};
    const char* input_names[6]    = {"img", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"};
    const char* output_names[2]   = {"logits", "boxes"};
};

GroundingDINO::GroundingDINO(string modelpath, float box_threshold, float text_threshold, bool with_logits)
    : box_threshold(box_threshold), text_threshold(text_threshold), with_logits(with_logits) {
    std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end());

    env            = Env(ORT_LOGGING_LEVEL_ERROR, "GroundingDINO");
    Ort::SessionOptions sessionOptions;

    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    ort_session = new Session(env, to_byte_string(widestr).c_str(), sessionOptions);
    tokenizer   = std::make_shared<bert_tokenizer::BertTokenizer>();
}

void GroundingDINO::img_process(Mat& img) {
    cv::Mat rgbimg;
    cv::cvtColor(img, rgbimg, COLOR_BGR2RGB);
    cv::resize(rgbimg, rgbimg, cv::Size(this->size[0], this->size[1]));
    vector<cv::Mat> rgbChannels(3);
    cv::split(rgbimg, rgbChannels);
    for (int c = 0; c < 3; c++) {
        rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / (255.0 * std[c]), (-mean[c]) / std[c]);
    }

    const int image_area = this->size[0] * this->size[1];
    this->input_img.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_img.data(), (float*)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_img.data() + image_area, (float*)rgbChannels[1].data, single_chn_size);
    memcpy(this->input_img.data() + image_area * 2, (float*)rgbChannels[2].data, single_chn_size);
}

void GroundingDINO::pre_process(Mat& srcimg, string text_prompt) {
    img_process(srcimg);
    srch = srcimg.rows, srcw = srcimg.cols;

    std::transform(text_prompt.begin(), text_prompt.end(), text_prompt.begin(), ::tolower);
    string caption = strip(text_prompt);
    if (endswith(caption, ".") == 0) {
        caption += " .";
    }

    this->input_ids.resize(1);
    this->attention_mask.resize(1);
    this->token_type_ids.resize(1);

    std::vector<int64> ids = tokenizer->encode(caption, 0, true, false);

    int len_ids   = ids.size();
    int trunc_len = len_ids <= this->max_text_len ? len_ids : this->max_text_len;
    input_ids[0].resize(trunc_len);
    token_type_ids[0].resize(trunc_len);
    attention_mask[0].resize(trunc_len);
    for (int i = 0; i < trunc_len; i++) {
        input_ids[0][i]      = ids[i];
        token_type_ids[0][i] = 0;
        attention_mask[0][i] = ids[i] > 0 ? 1 : 0;
    }
    const int num_token = input_ids[0].size();
    vector<int> idxs;
    for (int i = 0; i < num_token; i++) {
        for (int j = 0; j < this->specical_tokens.size(); j++) {
            if (input_ids[0][i] == this->specical_tokens[j]) {
                idxs.push_back(i);
            }
        }
    }

    len_ids   = idxs.size();
    trunc_len = num_token <= this->max_text_len ? num_token : this->max_text_len;
    text_self_attention_masks.resize(1);
    text_self_attention_masks[0].resize(trunc_len * trunc_len);
    position_ids.resize(1);
    position_ids[0].resize(trunc_len);
    for (int i = 0; i < trunc_len; i++) {
        for (int j = 0; j < trunc_len; j++) {
            text_self_attention_masks[0][i * trunc_len + j] = (i == j ? 1 : 0);
        }
        position_ids[0][i] = 0;
    }
    int previous_col = 0;
    for (int i = 0; i < len_ids; i++) {
        const int col = idxs[i];
        if (col == 0 || col == num_token - 1) {
            text_self_attention_masks[0][col * trunc_len + col] = true;
            position_ids[0][col]                                = 0;
        } else {
            for (int j = previous_col + 1; j <= col; j++) {
                for (int k = previous_col + 1; k <= col; k++) {
                    text_self_attention_masks[0][j * trunc_len + k] = true;
                }
                position_ids[0][j] = j - previous_col - 1;
            }
        }
        previous_col = col;
    }
}

void GroundingDINO::infer() {
    const int seq_len                          = input_ids[0].size();
    std::vector<int64_t> input_img_shape       = {1, 3, this->size[1], this->size[0]};
    std::vector<int64_t> input_ids_shape       = {1, seq_len};
    std::vector<int64_t> text_token_mask_shape = {1, seq_len, seq_len};

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back((Ort::Value::CreateTensor<float>(memory_info_handler, input_img.data(), input_img.size(), input_img_shape.data(), input_img_shape.size())));

    inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, input_ids[0].data(), input_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));

    inputTensors.push_back((Ort::Value::CreateTensor<bool>(memory_info_handler, reinterpret_cast<bool*>(attention_mask[0].data()), attention_mask[0].size(), input_ids_shape.data(), input_ids_shape.size())));

    inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, position_ids[0].data(), position_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));

    inputTensors.push_back((Ort::Value::CreateTensor<int64>(memory_info_handler, token_type_ids[0].data(), token_type_ids[0].size(), input_ids_shape.data(), input_ids_shape.size())));

    inputTensors.push_back((Ort::Value::CreateTensor<bool>(memory_info_handler, reinterpret_cast<bool*>(text_self_attention_masks[0].data()), text_self_attention_masks[0].size(), text_token_mask_shape.data(), text_token_mask_shape.size())));

    Ort::RunOptions runOptions;
    ort_outputs = this->ort_session->Run(runOptions, this->input_names, inputTensors.data(), inputTensors.size(), this->output_names, 2);
}

vector<Object> GroundingDINO::detect(Mat& srcimg, string text_prompt) {
    pre_process(srcimg, text_prompt);
    infer();
    vector<Object> res = post_process();
    return res;
}

vector<Object> GroundingDINO::post_process() {
    const float* ptr_logits           = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> logits_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* ptr_boxes            = ort_outputs[1].GetTensorMutableData<float>();
    std::vector<int64_t> boxes_shape  = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const int outh                    = logits_shape[1];
    const int outw                    = logits_shape[2];
    vector<int> filt_inds;
    vector<float> scores;

    // print_shape<int64_t>(boxes_shape,"ptr_boxes");
    // print_shape<int64_t>(logits_shape,"logits_shape");
    for (int i = 0; i < outh; i++) {
        float max_data = 0;
        for (int j = 0; j < outw; j++) {
            float x = sigmoid(ptr_logits[i * outw + j]);
            if (max_data < x) {
                max_data = x;
            }
        }
        if (max_data > this->box_threshold) {
            filt_inds.push_back(i);
            scores.push_back(max_data);
        }
    }

    std::vector<Object> objects;
    for (int i = 0; i < filt_inds.size(); i++) {
        const int ind      = filt_inds[i];
        const int left_idx = 0, right_idx = 255;
        for (int j = left_idx + 1; j < right_idx; j++) {
            float x = sigmoid(ptr_logits[ind * outw + j]);
            if (x > this->text_threshold) {
                const int64 token_id = input_ids[0][j];
                std::vector<size_t> v{(size_t)token_id};
                Object obj;
                obj.text = tokenizer->decode(v);
                obj.prob = scores[i];
                int xmin = int((ptr_boxes[ind * 4] - ptr_boxes[ind * 4 + 2] * 0.5) * srcw);
                int ymin = int((ptr_boxes[ind * 4 + 1] - ptr_boxes[ind * 4 + 3] * 0.5) * srch);

                int w   = int(ptr_boxes[ind * 4 + 2] * srcw);
                int h   = int(ptr_boxes[ind * 4 + 3] * srch);
                obj.box = Rect(xmin, ymin, w, h);
                objects.push_back(obj);

                break;
            }
        }
    }
    return objects;
}
