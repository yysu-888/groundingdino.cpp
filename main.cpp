#include <opencv2/opencv.hpp>
#include "grounding_dino.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("usge ./grounding.cpp model_path img_path,prompt(for cat.dog.)\n");
        return 0;
    }
    std::string model_path  = argv[1];
    std::string img_path    = argv[2];
    std::string text_prompt = argv[3];

    float box_threshold  = 0.5;
    float text_threshold = 0.25;

    cv::Mat srcimg = cv::imread(img_path);
    GroundingDINO net(model_path, box_threshold, text_threshold, true);

    TICK(grounding_dino)
    vector<Object> objects = net.detect(srcimg, text_prompt);
    TOCK(grounding_dino)

    for (size_t i = 0; i < objects.size(); i++) {
        cv::rectangle(srcimg, objects[i].box, cv::Scalar(0, 0, 255), 2);
        string label = format("%.2f", objects[i].prob);
        label        = objects[i].text + ":" + label;
        cv::putText(srcimg, label, cv::Point(objects[i].box.x, objects[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255));
    }

    imshow("GroundingDINO", srcimg);

    waitKey(0);
    cv::imwrite("demo.jpg",srcimg);
    destroyAllWindows();

    return 0;
}