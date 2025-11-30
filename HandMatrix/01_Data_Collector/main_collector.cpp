// ======================================================================================
// PROJEKT: HAND MATRIX - DATA COLLECTOR
// ZWECK:   Spezialisiertes Tool zur Aufnahme von Trainingsdaten in Unterordnern
// ======================================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// --- KONFIGURATION ---
const int IMG_SIZE = 32;
const std::string BASE_DIR = "training_data_raw"; // Hier landen die neuen Bilder

// Die Klassen-Namen auf ENGLISCH (Wichtig für Konsistenz!)
const std::vector<std::string> CLASS_NAMES = {
    "0_fist", "1_peace", "2_hand", "3_middle",
    "4_up", "5_down", "6_rock", "7_chill"
};

// --- HILFSFUNKTIONEN ---

// Erstellt die Ordnerstruktur
void initDirectories() {
    if (!fs::exists(BASE_DIR)) {
        fs::create_directory(BASE_DIR);
        std::cout << "[INIT] Hauptordner erstellt: " << BASE_DIR << std::endl;
    }
    for (const auto& name : CLASS_NAMES) {
        std::string path = BASE_DIR + "/" + name;
        if (!fs::exists(path)) {
            fs::create_directory(path);
            std::cout << "[INIT] Unterordner erstellt: " << path << std::endl;
        }
    }
}

// Speichert das Bild
void saveImage(const cv::Mat& img, int classId) {
    if (classId < 0 || classId >= CLASS_NAMES.size()) return;

    // Zeitstempel für eindeutigen Dateinamen
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string filename = BASE_DIR + "/" + CLASS_NAMES[classId] + "/img_" + std::to_string(timestamp) + ".png";

    cv::imwrite(filename, img);
    std::cout << "[SAVE] " << CLASS_NAMES[classId] << " -> " << filename << std::endl;
}

// --- MAIN PROGRAMM ---
int main() {
    initDirectories();

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "FEHLER: Keine Kamera gefunden!" << std::endl;
        return -1;
    }

    cv::namedWindow("Data Collector", cv::WINDOW_NORMAL);

    cv::Mat frame, cleanRoi, edges, saveInput;
    cv::Mat background;
    bool bgCaptured = false;
    int currentClassId = 0; // Startet bei Fist (0)

    std::cout << "--- DATA COLLECTOR GESTARTET ---" << std::endl;
    std::cout << "1. Halte Hand aus dem Rahmen." << std::endl;
    std::cout << "2. Druecke 'b' fuer Hintergrund-Reset." << std::endl;
    std::cout << "3. Waehle Klasse (0-7) und halte LEERTASTE zum Speichern." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);

        cv::Mat finalDisplay = frame.clone();

        // ROI Definition
        cv::Rect scanBox(350, 100, 250, 250);

        if (scanBox.x + scanBox.width < frame.cols && scanBox.y + scanBox.height < frame.rows) {
            cleanRoi = frame(scanBox).clone();

            // --- BILDVERARBEITUNG PIPELINE ---
            cv::Mat processingImg;

            if (bgCaptured) {
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

            // Fill Trick / Double Canny
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

            // Resize auf Zielgröße (32x32)
            cv::resize(edges, saveInput, cv::Size(IMG_SIZE, IMG_SIZE));

            // --- GUI ELEMENTE ---
            cv::Scalar boxColor = bgCaptured ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::rectangle(finalDisplay, scanBox, boxColor, 2);

            // Live-Vorschau des Inputs (oben links im Bild)
            if (bgCaptured) {
                cv::Mat debugView;
                cv::cvtColor(saveInput, debugView, cv::COLOR_GRAY2BGR);
                cv::resize(debugView, debugView, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
                debugView.copyTo(finalDisplay(cv::Rect(20, 20, 100, 100)));
                cv::rectangle(finalDisplay, cv::Rect(20, 20, 100, 100), cv::Scalar(0, 255, 255), 1);
            }
        }

        // --- INFO OVERLAY ---
        int yStart = 150;
        cv::putText(finalDisplay, "CURRENT CLASS:", cv::Point(20, yStart), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 1);
        cv::putText(finalDisplay, CLASS_NAMES[currentClassId], cv::Point(20, yStart + 25), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 255), 2);

        // Legende (jetzt Englisch)
        int yLeg = yStart + 60;
        cv::putText(finalDisplay, "[0] Fist    [1] Peace", cv::Point(20, yLeg), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(200, 200, 200), 1);
        cv::putText(finalDisplay, "[2] Hand    [3] Middle", cv::Point(20, yLeg + 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(200, 200, 200), 1);
        cv::putText(finalDisplay, "[4] Up      [5] Down", cv::Point(20, yLeg + 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(200, 200, 200), 1);
        cv::putText(finalDisplay, "[6] Rock    [7] Chill", cv::Point(20, yLeg + 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(200, 200, 200), 1);

        // Anleitung
        if (!bgCaptured) {
             cv::putText(finalDisplay, "PRESS 'B' TO LOCK BACKGROUND", cv::Point(20, finalDisplay.rows - 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255), 2);
        } else {
             cv::putText(finalDisplay, "HOLD 'SPACE' TO RECORD", cv::Point(20, finalDisplay.rows - 40), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);
             cv::putText(finalDisplay, "PRESS 'B' TO RESET BACKGROUND", cv::Point(20, finalDisplay.rows - 15), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
        }

        cv::imshow("Data Collector", finalDisplay);

        // --- INPUT HANDLING ---
        char key = (char)cv::waitKey(10);
        if (key == 27) break; // ESC

        // Klasse wechseln (Tasten '0' bis '7')
        if (key >= '0' && key <= '7') {
            currentClassId = key - '0';
        }

        // Hintergrund setzen
        if (key == 'b' && !cleanRoi.empty()) {
            background = cleanRoi.clone();
            bgCaptured = true;
            std::cout << "Hintergrund gelernt!" << std::endl;
        }

        // Speichern
        if (key == ' ' && bgCaptured) {
            saveImage(saveInput, currentClassId);
            cv::circle(finalDisplay, cv::Point(350+20, 100+20), 10, cv::Scalar(0,0,255), -1);
            cv::imshow("Data Collector", finalDisplay);
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}