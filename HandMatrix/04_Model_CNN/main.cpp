// ======================================================================================
// PROJECT: HAND MATRIX - CNN INFERENCE (FINAL FIX)
// PURPOSE: Live gesture recognition using a pre-trained ONNX model (TensorFlow/Keras)
// ======================================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

// --- CONFIGURATION ---
const int IMG_SIZE = 32;
const std::string MODEL_PATH = "hand_cnn.onnx";

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

// --- MAIN APPLICATION ---
int main() {
    // 1. Load the ONNX Model
    std::cout << "Loading model: " << MODEL_PATH << "..." << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(MODEL_PATH);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "ERROR: Could not load model! " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully!" << std::endl;

    // 2. Initialize Camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cv::namedWindow("HandMatrix CNN", cv::WINDOW_NORMAL);

    cv::Mat frame, roi, edges, brainInput;
    cv::Mat background;
    bool bgCaptured = false;

    std::vector<int> history;
    MatrixDisplay matrixDisplay;

    // 3. Main Loop
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
            cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            cv::resize(edges, brainInput, cv::Size(IMG_SIZE, IMG_SIZE));

            // --- CNN PREDICTION ---
            int stableClass = -1;
            std::string confidenceText = "0%";

            if (bgCaptured && !brainInput.empty()) {
                int activePixels = cv::countNonZero(brainInput);

                if (activePixels > 20) {
                    try {
                        // Ensure grayscale input
                        cv::Mat grayInput;
                        if (brainInput.channels() == 3) {
                            cv::cvtColor(brainInput, grayInput, cv::COLOR_BGR2GRAY);
                        } else {
                            grayInput = brainInput.clone();
                        }

                        // Prepare Input Blob (Standardize to 0-1)
                        cv::Mat blob = cv::dnn::blobFromImage(grayInput, 1.0/255.0, cv::Size(IMG_SIZE, IMG_SIZE), cv::Scalar(), false, false);

                        // Reshape to NHWC format (Batch, Height, Width, Channels) for TensorFlow compatibility
                        cv::Mat blobNHWC = blob.reshape(1, {1, IMG_SIZE, IMG_SIZE, 1});

                        net.setInput(blobNHWC);
                        cv::Mat output = net.forward();

                        // Process Output
                        cv::Point classIdPoint;
                        double confidence;
                        cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);

                        int currentClass = classIdPoint.x;

                        // Confidence Threshold (40%)
                        if (confidence < 0.40) {
                            currentClass = -1;
                        }

                        history.push_back(currentClass);
                        int confInt = (int)(confidence * 100);
                        confidenceText = std::to_string(confInt) + "%";

                    } catch (cv::Exception& e) {
                        std::cerr << "Inference Warning: " << e.what() << std::endl;
                    }
                } else {
                    history.clear();
                    history.push_back(-1);
                }

                // Smooth results over time
                if (history.size() > 10) history.erase(history.begin());

                int votes[9] = {0};
                for(int v : history) if (v != -1) votes[v]++;

                int maxVotes = 0;
                for(int k=0; k<8; k++) {
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
            if (bgCaptured && !brainInput.empty()) {
                cv::Mat debugView;
                cv::cvtColor(brainInput, debugView, cv::COLOR_GRAY2BGR);
                cv::resize(debugView, debugView, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
                cv::Rect debugRect(20, 20, 100, 100);
                debugView.copyTo(finalDisplay(debugRect));
                cv::rectangle(finalDisplay, debugRect, cv::Scalar(0, 255, 255), 1);
                cv::putText(finalDisplay, "CNN INPUT", cv::Point(20, 15), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 255, 255), 1);
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
                 cv::putText(finalDisplay, "CNN RUNNING... (ESC: EXIT)", cv::Point(20, finalDisplay.rows - 40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 1);
                 cv::putText(finalDisplay, "PRESS 'B' TO RESET BACKGROUND", cv::Point(20, finalDisplay.rows - 15), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
            }
        }

        cv::imshow("HandMatrix CNN", finalDisplay);
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