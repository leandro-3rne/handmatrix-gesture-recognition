// ======================================================================================
// PROJECT: HAND MATRIX - STANDARD NN
// PURPOSE: Live gesture recognition using a self-trained MLP (NeuralNet.h)
// ======================================================================================

#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <ctime>
#include "NeuralNet.h"

namespace fs = std::filesystem;

// --- CONFIGURATION ---
const int IMG_SIZE = 32;
const int INPUT_NEURONS = IMG_SIZE * IMG_SIZE;
const int NUM_CLASSES = 8;
const std::string TRAIN_DATA_PATH = "training_data_final";

const std::vector<std::string> CLASS_NAMES = {
    "FIST", "PEACE", "HAND", "MIDDLE",
    "UP", "DOWN", "ROCK", "CHILL"
};

// --- VISUALIZATION (MATRIX DOTS) ---
struct MatrixDisplay {
    std::vector<std::string> getPattern(int classId) {
        if (classId == 0) return { "..........", ".XXXXXXX..", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", ".XXXXXXX..", "..........", ".........." }; // FIST
        if (classId == 1) return { "X.......X.", "XX.....XX.", ".XX...XX..", ".XX...XX..", "..XX.XX...", "..XXXXX...", "...XXX....", "...XXX....", "....X.....", ".........." }; // PEACE
        if (classId == 2) return { ".X.X.X.X.X", ".X.X.X.X.X", ".X.X.X.X.X", ".XXXXXXXXX", ".XXXXXXXXX", ".XXXXXXXXX", ".XXXXXXXXX", "..XXXXXXX.", "..........", ".........." }; // HAND
        if (classId == 3) return { "....XX....", "....XX....", "....XX....", "....XX....", ".XXXXXXX..", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", ".XXXXXXX..", ".........." }; // MIDDLE
        if (classId == 4) return { "....XXX...", "....XXX...", ".....XXX..", ".....XXX..", ".XXXXXXX..", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", ".XXXXXXX..", ".........." }; // UP
        if (classId == 5) return { ".XXXXXXX..", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", ".XXXXXXX..", ".....XXX..", ".....XXX..", "....XXX...", "....XXX...", ".........." }; // DOWN
        if (classId == 6) return { ".XX....XX.", ".XX....XX.", ".XX....XX.", ".XX....XX.", ".XXXXXXXX.", ".XXXXXXXX.", ".XXXXXXXX.", ".XXXXXXXX.", "..XXXXXX..", ".........." }; // ROCK
        if (classId == 7) return { ".........X", "........XX", ".......XXX", "....XXXXX.", "...XXXXXX.", "...XXXXXX.", ".XXXXX....", "XXX.......", "XX........", ".........." }; // CHILL
        return {};
    }

    void drawIcon(cv::Mat& canvas, int classId) {
        if (classId == -1) return;

        std::vector<std::string> pattern = getPattern(classId);
        if (pattern.empty()) return;

        int gridSize = 10;
        int spacing = 30;
        int startX = (canvas.cols - ((gridSize - 1) * spacing)) / 2;
        int startY = (canvas.rows - ((gridSize - 1) * spacing)) / 2;

        cv::Scalar mainColor = (classId == 3) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Scalar glowColor = (classId == 3) ? cv::Scalar(0, 0, 150) : cv::Scalar(0, 150, 0);

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                if (pattern[y][x] == 'X') {
                    cv::Point center(startX + x * spacing, startY + y * spacing);
                    cv::circle(canvas, center, 10, mainColor, -1);
                    cv::circle(canvas, center, 14, glowColor, 2);
                }
            }
        }
    }
};

// --- DATA LOADING UTILS ---
struct TrainingSample {
    Eigen::VectorXd image;
    Eigen::VectorXd label;
};

std::vector<TrainingSample> loadData(std::string rootFolder) {
    std::vector<TrainingSample> data;
    if (!fs::exists(rootFolder)) {
        std::cerr << "WARNING: Data folder '" << rootFolder << "' not found!" << std::endl;
        return data;
    }

    std::cout << "--- Loading Training Data ---" << std::endl;
    for (const auto& entry : fs::directory_iterator(rootFolder)) {
        if (entry.is_directory()) {
            std::string folderName = entry.path().filename().string();
            // Parse Class ID from folder name (e.g. "0_fist")
            if (!isdigit(folderName[0])) continue;

            int labelId = folderName[0] - '0';
            if (labelId < 0 || labelId >= NUM_CLASSES) continue;

            int count = 0;
            for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                cv::Mat img = cv::imread(imgEntry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                Eigen::VectorXd inputVec(INPUT_NEURONS);
                for (int i = 0; i < INPUT_NEURONS; i++) inputVec(i) = img.data[i] / 255.0;

                Eigen::VectorXd targetVec(NUM_CLASSES);
                targetVec.setZero();
                targetVec(labelId) = 1.0;

                data.push_back({inputVec, targetVec});
                count++;
            }
            std::cout << "Class " << labelId << " (" << folderName << "): " << count << " images" << std::endl;
        }
    }
    return data;
}

// --- MAIN APPLICATION ---
int main() {
    std::srand((unsigned int)time(0));

    // 1. Setup Neural Network
    NeuralNet brain(INPUT_NEURONS, 256, 64, NUM_CLASSES);

    // 2. Training Loop
    std::vector<TrainingSample> trainingSet = loadData(TRAIN_DATA_PATH);
    if (!trainingSet.empty()) {
        std::cout << "Starting Training (10,000 iterations)..." << std::endl;
        for (int i = 0; i < 10000; i++) {
            int idx = rand() % trainingSet.size();
            brain.train(trainingSet[idx].image, trainingSet[idx].label);
            if (i % 2000 == 0) std::cout << "Progress: " << (i/100) << "%" << std::endl;
        }
        std::cout << "Training complete." << std::endl;
    } else {
        std::cout << "WARNING: No data found. Network remains untrained." << std::endl;
    }

    // 3. Initialize Camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::namedWindow("HandMatrix NN", cv::WINDOW_NORMAL);

    cv::Mat frame, roi, edges, brainInput;
    cv::Mat background;
    bool bgCaptured = false;

    std::vector<int> history;
    MatrixDisplay matrixDisplay;

    // 4. Main Loop
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);

        cv::Mat finalDisplay = frame.clone();
        cv::Mat cleanRoi;
        cv::Rect scanBox(350, 100, 250, 250);

        if (scanBox.x + scanBox.width < frame.cols && scanBox.y + scanBox.height < frame.rows) {
            cleanRoi = frame(scanBox).clone();
            roi = frame(scanBox);

            // --- IMAGE PREPROCESSING ---
            cv::Mat processingImg;

            if (bgCaptured) {
                // Background Subtraction
                cv::Mat diff;
                cv::absdiff(cleanRoi, background, diff);
                cv::Mat mask;
                cv::cvtColor(diff, diff, cv::COLOR_BGR2GRAY);
                cv::threshold(diff, mask, 30, 255, cv::THRESH_BINARY);
                cv::Mat cleanHand = cv::Mat::zeros(cleanRoi.size(), cleanRoi.type());
                cleanRoi.copyTo(cleanHand, mask);
                processingImg = cleanHand;
            } else {
                processingImg = cleanRoi;
            }

            // Edge Detection
            cv::Mat hsv;
            cv::cvtColor(processingImg, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            cv::Mat saturation = channels[1];
            cv::equalizeHist(saturation, saturation);

            cv::Canny(saturation, edges, 50, 150);
            cv::GaussianBlur(edges, edges, cv::Size(5, 5), 0);
            cv::Canny(edges, edges, 50, 150);

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::dilate(edges, edges, kernel);

            cv::resize(edges, brainInput, cv::Size(IMG_SIZE, IMG_SIZE));

            // --- NN PREDICTION ---
            int stableClass = -1;
            std::string confidenceText = "0%";

            if (bgCaptured) {
                 int activePixels = cv::countNonZero(brainInput);
                 if (activePixels > 20) {
                     Eigen::VectorXd vec(INPUT_NEURONS);
                     for(int i=0; i<INPUT_NEURONS; i++) vec(i) = brainInput.data[i] / 255.0;

                     Eigen::VectorXd result = brain.forward(vec);

                     int currentClass = -1;
                     double maxProb = 0.0;
                     for(int k=0; k<NUM_CLASSES; k++) {
                         if(result(k) > maxProb) {
                             maxProb = result(k);
                             currentClass = k;
                         }
                     }

                     // Confidence Threshold (50%)
                     if (maxProb < 0.50) currentClass = -1;

                     history.push_back(currentClass);
                     int conf = (int)(maxProb * 100);
                     confidenceText = std::to_string(conf) + "%";
                 }
                 else {
                     history.clear();
                     history.push_back(-1);
                 }

                 // Smooth results over time
                 if (history.size() > 10) history.erase(history.begin());

                 int votes[9] = {0};
                 for(int v : history) if (v != -1) votes[v]++;

                 int maxVotes = 0;
                 for(int k=0; k<NUM_CLASSES; k++) {
                     if(votes[k] > maxVotes) {
                         maxVotes = votes[k];
                         stableClass = k;
                     }
                 }
                 if (maxVotes < 5) stableClass = -1;
            }

            // --- UI & VISUALIZATION ---
            if (bgCaptured) {
                cv::Mat overlay = cv::Mat::zeros(finalDisplay.size(), finalDisplay.type());
                matrixDisplay.drawIcon(overlay, stableClass);
                cv::add(finalDisplay, overlay, finalDisplay);
            }

            cv::Scalar boxColor = bgCaptured ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(finalDisplay, scanBox, boxColor, 2);

            // Debug View (Top Left)
            if (bgCaptured) {
                cv::Mat debugView;
                cv::cvtColor(brainInput, debugView, cv::COLOR_GRAY2BGR);
                cv::resize(debugView, debugView, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
                cv::Rect debugRect(20, 20, 100, 100);
                debugView.copyTo(finalDisplay(debugRect));
                cv::rectangle(finalDisplay, debugRect, cv::Scalar(0, 255, 255), 1);
                cv::putText(finalDisplay, "NN INPUT", cv::Point(20, 15), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 255, 255), 1);
            }

            // Status Text
            int yText = 150;
            cv::putText(finalDisplay, "DETECTED CLASS:", cv::Point(20, yText), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1);

            std::string className = (stableClass != -1) ? CLASS_NAMES[stableClass] : "WAITING...";
            cv::Scalar classColor = (stableClass != -1) ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 100);

            cv::putText(finalDisplay, className, cv::Point(20, yText + 30), cv::FONT_HERSHEY_PLAIN, 2.5, classColor, 2);

            if(stableClass != -1) {
                cv::putText(finalDisplay, "Confidence: " + confidenceText, cv::Point(20, yText + 55), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(200, 200, 200), 1);
            }

            // Instructions
            if (!bgCaptured) {
                 cv::putText(finalDisplay, "PRESS 'B' TO LOCK BACKGROUND", cv::Point(20, finalDisplay.rows - 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255), 2);
            } else {
                 cv::putText(finalDisplay, "NN RUNNING... (ESC: EXIT)", cv::Point(20, finalDisplay.rows - 40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1);
                 cv::putText(finalDisplay, "PRESS 'B' TO RESET BACKGROUND", cv::Point(20, finalDisplay.rows - 15), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
            }
        }

        cv::imshow("HandMatrix NN", finalDisplay);
        char key = (char)cv::waitKey(10);
        if (key == 27) break; // ESC

        // Background Reset
        if (key == 'b' && !cleanRoi.empty()) {
            background = cleanRoi.clone();
            bgCaptured = true;
            std::cout << "Background reset!" << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}