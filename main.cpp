#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using nlohmann::json;

vector<Point> bhContoursCenter(const vector<vector<Point>>& contours,bool centerOfMass,int contourIdx=-1)
{
    vector<Point> result;
    if (contourIdx > -1)
    {
        if (centerOfMass)
        {
            Moments m = moments(contours[contourIdx],true);
            result.push_back( Point(m.m10/m.m00, m.m01/m.m00));
        }
        else
        {
            Rect rct = boundingRect(contours[contourIdx]);
            result.push_back( Point(rct.x + rct.width / 2 ,rct.y + rct.height / 2));
        }
    }
    else
    {
        if (centerOfMass)
        {
            for (int i=0; i < contours.size();i++)
            {
                Moments m = moments(contours[i],true);
                result.push_back( Point(m.m10/m.m00, m.m01/m.m00));
            }
        }
        else
        {
            for (int i=0; i < contours.size(); i++)
            {
                Rect rct = boundingRect(contours[i]);
                result.push_back(Point(rct.x + rct.width / 2 ,rct.y + rct.height / 2));
            }
        }
    }
    return result;
}

vector<Point> bhFindLocalMaximum(InputArray _src,int neighbor=2){
    Mat src = _src.getMat();

    Mat peak_img = src.clone();
    dilate(peak_img,peak_img,Mat(),Point(-1,-1),neighbor);
    peak_img = peak_img - src;

    Mat flat_img ;
    erode(src,flat_img,Mat(),Point(-1,-1),neighbor);
    flat_img = src - flat_img;

    threshold(peak_img,peak_img,0,255,THRESH_BINARY);
    threshold(flat_img,flat_img,0,255,THRESH_BINARY);
    bitwise_not(flat_img,flat_img);

    peak_img.setTo(Scalar::all(255),flat_img);
    bitwise_not(peak_img,peak_img);

    vector<vector<Point>> contours;
    findContours(peak_img,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

    return bhContoursCenter(contours,false);
}

Mat Binary2Color(Mat& bwImg)
{
    Mat colorImg;
    vector<Mat> channels;
    channels.push_back(bwImg);
    channels.push_back(bwImg);
    channels.push_back(bwImg);
    merge(channels, colorImg);

    return colorImg;
}

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.second < right.second;
    }
};

vector<Point> getGTPts(json gt, string& frame_name, int frame_idx){
    vector<Point> gtPts;
    auto num_obj = gt[frame_idx][frame_name].size();
    for (int i=0; i<num_obj; i++){
        double xTemp = gt[frame_idx][frame_name][i]["center"][0];
        double yTemp = gt[frame_idx][frame_name][i]["center"][1];
        gtPts.push_back(Point(xTemp, yTemp));
    }
    return gtPts;
}

void GetMetrics(vector<Point>& vecPt, vector<Point>& gtPts, double min_dist)
{
    vector<Point> tempPts;
    map<int, int> idx_map;
    vector<double> distances;
    auto num_obj = gtPts.size();
    auto num_pred_obj = vecPt.size();
    cout << num_obj << "   " << num_pred_obj << endl;

    double min_dist_tmp(10e10);
    int min_idx(-1);
    vector<int> gt_usage(num_obj);
    int FP(0);
    int TP(0);
    int FN(0);
    for (int i=0; i<num_pred_obj; i++){
        min_dist_tmp = 10e10;
        min_idx = -1;
        for (int j=0; j<num_obj; j++){
            double temp_dist = norm(vecPt[i] - gtPts[j]);
            if (temp_dist < min_dist){
                if (temp_dist < min_dist_tmp){
                    min_dist_tmp = temp_dist;
                    min_idx = j;
                }
            }
        }
        if (min_idx>-1) {
            idx_map[i] = min_idx;
            gt_usage[min_idx] = 1;
            TP++;
        }
        else
            FP++;
        distances.emplace_back(min_dist_tmp);
    }
    for (int i=0; i<num_obj; i++){
        if (gt_usage[i] == 0)
            FN++;
    }
    cout << TP << endl << FP << endl << FN << endl;
    double precision = (TP+0.0)/(TP+FP);
    double recall = (TP+0.0)/(TP+FN);
    double f1 = 2 * precision * recall / (precision + recall);
    cout << "Dist threshold is - " << min_dist << endl;
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1: " << f1 << endl;

    int cnt = 0;
    double dist_sum = 0;
    for (double distance : distances){
        if (distance < min_dist){
            dist_sum += distance;
            cnt++;
        }
    }
    cout << "Mean distance is - " << dist_sum/cnt << endl;
}

void meanAveragePrecision(vector<Point>& vecPt, json gt, string frame_name, int frame_idx)
{
    vector<Point> gtPts = getGTPts(gt, frame_name, frame_idx);
    vector<Point> tempPts;
    map<int, int> idx_map;
    vector<pair<int, double>> idx_dist;
    auto num_obj = gtPts.size();
    auto num_pred_obj = vecPt.size();
    cout << num_obj << "   " << num_pred_obj << endl;

    double min_dist_tmp(10e10);
    int min_idx(-1);
    vector<int> gt_usage(num_obj);
    for (int i=0; i<num_pred_obj; i++){
        min_dist_tmp = 10e10;
        min_idx = -1;
        for (int j=0; j<num_obj; j++){
            double temp_dist = norm(vecPt[i] - gtPts[j]);
            if (temp_dist < min_dist_tmp){
                min_dist_tmp = temp_dist;
                min_idx = j;
            }
        }
        idx_map[i] = min_idx;
        idx_dist.emplace_back(make_pair(min_idx, min_dist_tmp));
        gt_usage[min_idx] = 1;
    }
    sort(idx_dist.begin(), idx_dist.end(), sort_pred());
}

vector<Point> getCorrectPts(vector<Point> raw_pts){
    vector<Point> correctPts;
    for(int i=0; i<raw_pts.size(); i++) {
        if ((raw_pts[i].x < 0) or (raw_pts[i].y < 0)){
            continue;
        }
        correctPts.push_back(raw_pts[i]);
    }
    return correctPts;
}

pair<Mat, vector<Point>> GenHeatmap(int n = 30, int width = 1280, int height = 1024){
    Mat heatmap(height, width, CV_32F);
    vector<Point> gt;

    RNG rng(getCPUTickCount());
    int temp_x, temp_y;
    double temp_sigma, ampl = 1.0, gauss_x, gauss_y, gauss_v;
    float gauss_v_f;
    for (int i = 0; i < n; i++){
        temp_y = rng.uniform(400, height-400);
        temp_x = rng.uniform(400, width-400);
        temp_sigma = rng.uniform(5.0, 7.0);
        gt.emplace_back(Point(temp_x, temp_y));

        for (int r=0; r < height; r++){
            for (int c=0; c < width; c++){
                if (norm(Point(r-temp_y, c-temp_x)) > 15)
                    continue;
                gauss_x = ((c - temp_x)*(double(c) - temp_x)) / (2.0 * temp_sigma*temp_sigma);
                gauss_y = ((r - temp_y)*(double(r) - temp_y)) / (2.0 * temp_sigma*temp_sigma);
                gauss_v_f = float(ampl * exp(-(gauss_x+gauss_y + 0.0)));
                heatmap.at<float>(r, c) = MAX(gauss_v_f, heatmap.at<float>(r, c));
            }
        }
    }
    normalize(heatmap, heatmap, 0.0, 1.0, NORM_MINMAX);
    cv::Mat noise = Mat(heatmap.size(),CV_32F);
    cv::randn(noise, 0, 0.000205);
    heatmap = heatmap + noise;
    //normalize(heatmap, heatmap, 0.0, 1.0, NORM_MINMAX);
    double min_m, max_m;
    cv::minMaxLoc(heatmap, &min_m, &max_m);

    heatmap.convertTo(heatmap, CV_8U, 255.0/(max_m-min_m), -255.0*min_m/(max_m-min_m));
    return make_pair(heatmap, gt);

}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}


int main() {
    std::string heatmap_path = "gauses/masked_gaussed_dafni600.png";
    ifstream ifs("gauses/gt.json");
    json gt = json::parse(ifs);
    cv::Mat heatmap;
    vector<Point> gtPts;
    pair<Mat, vector<Point>> gen_data= GenHeatmap();
    heatmap = gen_data.first;
    gtPts = gen_data.second;

    cv::Mat heatmap_img = cv::imread(heatmap_path, cv::IMREAD_GRAYSCALE);
    if(heatmap.empty())
    {
        std::cout << "Could not read the image: " << heatmap_path << std::endl;
        return 1;
    }

    vector<Point> vecPt = bhFindLocalMaximum(heatmap, 1);
    vector<Point> correctPts = getCorrectPts(vecPt);

    Mat resultImg = Binary2Color(heatmap);
    for(int i=0; i<correctPts.size(); i++) {
        circle(resultImg, correctPts[i], 2, Scalar(0, 0, 255), 2);
    }
    GetMetrics(correctPts, gtPts, 1.1);

    imshow("Local maximum image", resultImg);
    imshow("Source image", heatmap);

    imwrite("result_dafni1.png", resultImg);

    waitKey(0);
    return 0;
}
