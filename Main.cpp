#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <mutex>
#include <thread>
#include <atomic>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

std::mutex mtx;  // Mutex for thread-safe access to patternCounts and totalSets
unordered_map<string, int> globalPatternCounts;  // Global pattern counts
int globalTotalSets = 0;  // Global total of unique patterns (blocks)
std::atomic<int> progressCounter(0);  // Atomic counter for progress tracking

std::atomic<int> visibilityProgressCounter(0);  // Atomic counter for visibility progress


// Helper function to check if a point is within the image boundaries
bool isWithinImage(int x, int y, const Mat& image) {
    return x >= 0 && y >= 0 && x < image.cols && y < image.rows;
}

// Function to calculate the longest straight line of visible white pixels for each white pixel
void calculateLongestLineSubset(
    const Mat& image,
    const vector<Point>& whitePixels,
    vector<int>& lineSizes,
    int startIdx,
    int endIdx,
    atomic<int>& progressCounter,
    int totalPixels
) {
    for (int idx = startIdx; idx < endIdx; ++idx) {
        const Point& pixel = whitePixels[idx];
        int maxLineSize = 1; // Include the pixel itself

        // Check in all 8 directions
        vector<Point> directions = {
            {1, 0}, {0, 1}, {-1, 0}, {0, -1}, // Horizontal and vertical
            {1, 1}, {-1, 1}, {-1, -1}, {1, -1} // Diagonals
        };

        for (const Point& dir : directions) {
            int lineSize = 1; // Include the current pixel
            for (int sign : {-1, 1}) { // Check in both forward and backward directions
                Point nextPixel = pixel;
                while (true) {
                    nextPixel += sign * dir;
                    if (!isWithinImage(nextPixel.x, nextPixel.y, image) ||
                        image.at<uchar>(nextPixel.y, nextPixel.x) != 255) {
                        break;
                    }
                    ++lineSize;
                }
            }
            maxLineSize = max(maxLineSize, lineSize - 1); // Avoid double-counting the pixel itself
        }

        lineSizes[idx] = maxLineSize;

        // Update progress
        int completed = progressCounter.fetch_add(1) + 1;
        int progress = (completed * 100) / totalPixels;
        cout << "\rLine Size Progress: " << progress << "% " << flush;
    }
}

// Function to create a heatmap based on the longest line sizes
Mat createLineHeatmap(const Mat& image, const vector<int>& lineSizes, const vector<Point>& whitePixels) {
    Mat heatmap = Mat::zeros(image.size(), CV_8UC3);

    // Find the min and max line sizes
    int minLineSize = *min_element(lineSizes.begin(), lineSizes.end());
    int maxLineSize = *max_element(lineSizes.begin(), lineSizes.end());

    for (size_t i = 0; i < whitePixels.size(); ++i) {
        Point pt = whitePixels[i];
        int lineSize = lineSizes[i];

        // Normalize the line size to [0, 255]
        int normalized = static_cast<int>(255.0 * (lineSize - minLineSize) / (maxLineSize - minLineSize));

        // Apply the rainbow colormap
        Vec3b color;
        applyColorMap(Mat(1, 1, CV_8UC1, Scalar(normalized)), Mat(1, 1, CV_8UC3, &color), COLORMAP_RAINBOW);
        heatmap.at<Vec3b>(pt.y, pt.x) = color;
    }

    return heatmap;
}


// Function to display a progress bar
void displayProgressBar(int completed, int total) {
    int barWidth = 50;
    double progress = static_cast<double>(completed) / total;
    int pos = static_cast<int>(barWidth * progress);

    cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "%" << flush;
}

// Bresenham's Line Algorithm to determine visibility
bool isVisible(const Mat& image, Point start, Point end) {
    int dx = abs(end.x - start.x);
    int dy = -abs(end.y - start.y);
    int sx = (start.x < end.x) ? 1 : -1;
    int sy = (start.y < end.y) ? 1 : -1;
    int err = dx + dy;

    int x = start.x;
    int y = start.y;

    while (true) {
        if (image.at<uchar>(y, x) == 0) {  // Black pixel blocks visibility
            return false;
        }
        if (x == end.x && y == end.y) {
            return true;
        }

        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

// Thread function to calculate visibility scores for a subset of white pixels
void calculateVisibilitySubset(const Mat& image, const vector<Point>& whitePixels, vector<atomic<int>>& visibilityScores, int startIdx, int endIdx, atomic<int>& progressCounter, int totalPixels) {
    try {
        vector<int> localScores(whitePixels.size(), 0);

        for (int idx = startIdx; idx < endIdx; ++idx) {
            const Point& pixel = whitePixels[idx];
            int visibleCount = 0;

            for (const Point& target : whitePixels) {
                if (isVisible(image, pixel, target)) {
                    ++visibleCount;
                }
            }

            localScores[idx] = visibleCount;

            // Update progress
            int completed = progressCounter.fetch_add(1) + 1;
            int progress = (completed * 100) / totalPixels;
            cout << "\rProgress: " << progress << "% " << flush;
        }

        // Thread-safe aggregation of local results into the global visibility scores
        for (int idx = startIdx; idx < endIdx; ++idx) {
            visibilityScores[idx] = localScores[idx];
        }
    }
    catch (const exception& e) {
        cerr << "Exception in visibility calculation: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown exception in visibility calculation!" << endl;
    }
}

struct PointHash {
    size_t operator()(const Point& point) const {
        return std::hash<int>()(point.x) ^ (std::hash<int>()(point.y) << 1);
    }
};

struct PointEqual {
    bool operator()(const Point& lhs, const Point& rhs) const {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
};

struct PointPairHash {
    size_t operator()(const pair<Point, Point>& pair) const {
        size_t h1 = std::hash<int>()(pair.first.x) ^ (std::hash<int>()(pair.first.y) << 1);
        size_t h2 = std::hash<int>()(pair.second.x) ^ (std::hash<int>()(pair.second.y) << 1);
        return h1 ^ (h2 << 1); // Combine the hashes
    }
};

struct PointPairEqual {
    bool operator()(const pair<Point, Point>& lhs, const pair<Point, Point>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

void calculateVisibilityAndLongestLineSubset(
    const Mat& image,
    const vector<Point>& whitePixels,
    vector<atomic<int>>& visibilityScores,
    vector<atomic<int>>& lineSizes,
    int startIdx,
    int endIdx,
    atomic<int>& progressCounter,
    int totalPixels
) {
    const int numDirections = 7200; // Number of directions
    vector<Point2f> directions;

    // Generate evenly spaced directions around a circle
    for (int i = 0; i < numDirections; ++i) {
        float angle = 2 * CV_PI * i / numDirections;
        directions.emplace_back(cos(angle), sin(angle));
    }

    for (int idx = startIdx; idx < endIdx; ++idx) {
        const Point& pixel = whitePixels[idx];
        int visibleCount = 0;
        int maxLineSize = 1; // Include the pixel itself

        // Visibility Calculation: Check every other white pixel for visibility
        //for (const Point& target : whitePixels) {
        //    if (pixel != target && isVisible(image, pixel, target)) {
        //        ++visibleCount;
        //    }
        //}

        // Longest Line Calculation: Check in all generated directions
        for (const Point2f& dir : directions) {
            int lineSize = 1; // Start with the current pixel
            for (int sign : {-1, 1}) { // Check in both directions
                Point2f nextPixel = pixel;
                while (true) {
                    nextPixel += sign * dir;

                    // Round to nearest integer for pixel coordinates
                    Point roundedPixel(cvRound(nextPixel.x), cvRound(nextPixel.y));

                    if (!isWithinImage(roundedPixel.x, roundedPixel.y, image) ||
                        image.at<uchar>(roundedPixel.y, roundedPixel.x) != 255) { // Black pixel blocks
                        break;
                    }
                    ++lineSize;
                }
            }
            maxLineSize = max(maxLineSize, lineSize - 1); // Adjust for double-counting
        }

        // Store the results
        visibilityScores[idx].store(visibleCount);
        lineSizes[idx].store(maxLineSize);

        // Update progress
        int completed = progressCounter.fetch_add(1) + 1;
        int progress = (completed * 100) / totalPixels;
        cout << "\rProgress: " << progress << "% " << flush;
    }
}



// Function to create a heatmap based on visibility scores
Mat createHeatmap(const Mat& image, const vector<atomic<int>>& visibilityScores, const vector<Point>& whitePixels) {
    Mat heatmap = Mat::zeros(image.size(), CV_8UC1); // Initialize a single-channel heatmap (grayscale)

    // Calculate min and max visibility, considering only white pixels
    int minScore = INT_MAX;
    int maxScore = INT_MIN;
    for (size_t i = 0; i < whitePixels.size(); ++i) {
        int value = visibilityScores[i].load();
        minScore = min(minScore, value);
        maxScore = max(maxScore, value);
    }

    // Avoid division by zero if all scores are equal
    if (minScore == maxScore) {
        maxScore = minScore + 1;
    }

    // Populate the heatmap with normalized and inverted scores for white pixels
    for (size_t i = 0; i < whitePixels.size(); ++i) {
        Point pt = whitePixels[i];
        int score = visibilityScores[i];

        // Normalize score between 0 and 255 and invert it
        int normalized = static_cast<int>(255.0 * (score - minScore) / (maxScore - minScore));
        int inverted = 255 - normalized; // Invert the value
        heatmap.at<uchar>(pt.y, pt.x) = inverted;
    }

    // Convert the grayscale heatmap to a colored heatmap using the Rainbow colormap
    Mat coloredHeatmap;
    applyColorMap(heatmap, coloredHeatmap, COLORMAP_RAINBOW);

    // Retain original black pixels in the final heatmap
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (image.at<uchar>(y, x) == 0) { // Preserve black pixels
                coloredHeatmap.at<Vec3b>(y, x) = Vec3b(0, 0, 0); // Set to black
            }
        }
    }

    return coloredHeatmap;
}



// Function to generate canonical form of a block pattern by considering rotations and mirrorings
string getCanonicalPattern(const Mat& block) {
    vector<Mat> transformations;
    transformations.push_back(block);
    Mat rotated;
    rotate(block, rotated, ROTATE_90_CLOCKWISE);
    transformations.push_back(rotated);
    rotate(rotated, rotated, ROTATE_90_CLOCKWISE);
    transformations.push_back(rotated);
    rotate(rotated, rotated, ROTATE_90_CLOCKWISE);
    transformations.push_back(rotated);

    for (int i = 0; i < 4; ++i) {
        Mat mirrored;
        flip(transformations[i], mirrored, 1);
        transformations.push_back(mirrored);
    }

    string canonicalPattern;
    for (const Mat& tform : transformations) {
        string pattern;
        for (int r = 0; r < tform.rows; ++r) {
            for (int c = 0; c < tform.cols; ++c) {
                pattern += to_string(tform.at<uchar>(r, c)) + "_";
            }
        }
        if (canonicalPattern.empty() || pattern < canonicalPattern) {
            canonicalPattern = pattern;
        }
    }

    return canonicalPattern;
}

// Thread function for calculating pattern counts for a specific region
void calculatePatternCounts(Mat& image, int blockRows, int blockCols, int startRow, int endRow, int totalRows) {
    unordered_map<string, int> localPatternCounts;
    int localTotalSets = 0;

    for (int i = startRow; i <= endRow - blockRows; ++i) {
        for (int j = 0; j <= image.cols - blockCols; ++j) {
            Mat block = image(Rect(j, i, blockCols, blockRows));
            string canonicalPattern = getCanonicalPattern(block);

            bool isAllBlack = (countNonZero(block == 0) == block.total());
            bool isAllWhite = (countNonZero(block == 255) == block.total());
            if (isAllBlack || isAllWhite) continue;

            localPatternCounts[canonicalPattern]++;
            localTotalSets++;
        }
        // Update and display progress
        int completed = progressCounter.fetch_add(1) + 1;
        int progress = (completed * 100) / totalRows;
        cout << "\rProgress: " << progress << "% " << flush;
    }

    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& [pattern, count] : localPatternCounts) {
        globalPatternCounts[pattern] += count;
    }
    globalTotalSets += localTotalSets;
}

// Function to calculate entropy from global pattern counts
double calculateEntropy() {
    double entropy = 0.0;
    for (const auto& [pattern, count] : globalPatternCounts) {
        double probability = static_cast<double>(count) / globalTotalSets;
        entropy -= probability * log2(probability);
    }
    return entropy;
}

// Function to find the outermost black pixel border
vector<Point> findBorderPolygon(const Mat& image) {
    vector<Point> blackPixels;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (image.at<uchar>(i, j) == 0) {
                blackPixels.emplace_back(j, i);
            }
        }
    }

    vector<Point> borderPolygon;
    if (!blackPixels.empty()) {
        convexHull(blackPixels, borderPolygon);
    }
    return borderPolygon;
}

// Function to calculate black and white pixel counts within the border polygon
pair<int, int> calculatePixelCountsInsideBorder(const Mat& image, const vector<Point>& borderPolygon) {
    if (borderPolygon.empty()) {
        cerr << "Border polygon is empty!" << endl;
        return { 0, 0 };
    }

    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    fillPoly(mask, vector<vector<Point>>{borderPolygon}, Scalar(255));

    int blackPixelCount = 0, whitePixelCount = 0;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (mask.at<uchar>(i, j) == 255) {
                if (image.at<uchar>(i, j) == 0) {
                    blackPixelCount++;
                }
                else {
                    whitePixelCount++;
                }
            }
        }
    }

    return { blackPixelCount, whitePixelCount };
}

// Function to generate an output image with the border polygon highlighted in red
Mat createOutputImage(const Mat& image, const vector<Point>& borderPolygon) {
    Mat outputImage;
    cvtColor(image, outputImage, COLOR_GRAY2BGR);

    polylines(outputImage, vector<vector<Point>>{borderPolygon}, true, Scalar(0, 0, 255), 2);

    Mat overlay;
    outputImage.copyTo(overlay);
    fillPoly(overlay, vector<vector<Point>>{borderPolygon}, Scalar(0, 0, 255));
    addWeighted(overlay, 0.3, outputImage, 0.7, 0, outputImage);

    return outputImage;
}

double calculateMaxEntropy(int imageWidth, int imageHeight, int blockRows, int blockCols) {
    int totalPatterns = (imageWidth - blockCols + 1) * (imageHeight - blockRows + 1);
    return log2(totalPatterns);
}

int main() {
    _putenv("OPENCV_IO_MAX_IMAGE_PIXELS=2500000000000");
    string folderPath;
    cout << "Enter the folder path containing the images: ";
    getline(cin, folderPath);

    int blockRows, blockCols;
    cout << "Enter the number of rows for the block size: ";
    cin >> blockRows;
    cout << "Enter the number of columns for the block size: ";
    cin >> blockCols;

    int numThreads;
    cout << "Enter the number of threads to use (0 for max available): ";
    cin >> numThreads;
    if (numThreads == 0) {
        numThreads = thread::hardware_concurrency();
    }

    ofstream outputFile("entropy_results.csv", ios::app);
    if (!outputFile.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    if (outputFile.tellp() == 0) {
        outputFile << "Image,Ratio of Black Pixels,Entropy,Normalized Entropy,"
            << "Black Pixels Inside Border,White Pixels Inside Border,"
            << "Total Black Pixels,Total White Pixels\n";
    }

    ofstream outputFile2("visibility_results.csv");
    if (!outputFile2.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    // Add the new header
    outputFile2 << "Image,Average Visibility,Average Longest Line\n";

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        try {
            if (entry.is_regular_file()) {
                Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);
                if (image.empty()) {
                    cerr << "Could not read image: " << entry.path() << endl;
                    continue;
                }

                vector<Point> borderPolygon = findBorderPolygon(image);
                auto [blackInside, whiteInside] = calculatePixelCountsInsideBorder(image, borderPolygon);

                double ratioBlackPixels = static_cast<double>(blackInside) / (blackInside + whiteInside);

                globalPatternCounts.clear();
                globalTotalSets = 0;
                progressCounter = 0;
                atomic<int> progressCounter2(0);

                vector<thread> threads;
                int rowsPerThread = image.rows / numThreads;
                for (int t = 0; t < numThreads; ++t) {
                    int startRow = t * rowsPerThread;
                    int endRow = (t == numThreads - 1) ? image.rows : startRow + rowsPerThread;
                    threads.emplace_back(calculatePatternCounts, ref(image), blockRows, blockCols, startRow, endRow, image.rows);
                }

                for (auto& th : threads) {
                    th.join();
                }

                double entropy = calculateEntropy();
                double maxEntropy = calculateMaxEntropy(image.cols, image.rows, blockRows, blockCols);
                double normalizedEntropy = (maxEntropy > 0) ? entropy / maxEntropy : 0.0;

                int totalBlackPixels = countNonZero(image == 0);
                int totalWhitePixels = image.total() - totalBlackPixels;

                cout << "\nImage: " << entry.path().filename() << endl;
                cout << "Ratio of Black Pixels (inside border): " << ratioBlackPixels << endl;
                cout << "Entropy: " << entropy << endl;
                cout << "Normalized Entropy: " << normalizedEntropy << endl;
                cout << "Black Pixels Inside Border: " << blackInside << endl;
                cout << "White Pixels Inside Border: " << whiteInside << endl;
                cout << "Total Black Pixels: " << totalBlackPixels << endl;
                cout << "Total White Pixels: " << totalWhitePixels << endl << endl;

                outputFile << entry.path().filename() << ","
                    << ratioBlackPixels << ","
                    << entropy << ","
                    << normalizedEntropy << ","
                    << blackInside << ","
                    << whiteInside << ","
                    << totalBlackPixels << ","
                    << totalWhitePixels << "\n";
                outputFile.flush();

                Mat outputImage = createOutputImage(image, borderPolygon);
                string outputImagePath = "output_" + entry.path().filename().string();
                imwrite(outputImagePath, outputImage);
                cout << "Output image saved as '" << outputImagePath << "'" << endl;

                vector<Point> whitePixels;
                for (int y = 0; y < image.rows; ++y) {
                    for (int x = 0; x < image.cols; ++x) {
                        if (image.at<uchar>(y, x) == 255) {
                            whitePixels.emplace_back(x, y);
                        }
                    }
                }

                cout << "Processing visibility for image: " << entry.path().filename() << endl;

                vector<atomic<int>> visibilityScores(whitePixels.size());
                std::vector<std::atomic<int>> lineSizes(whitePixels.size());
                int totalPixels = whitePixels.size();
                int pixelsPerThread = whitePixels.size() / numThreads;

                try {
                    for (int t = 0; t < numThreads; ++t) {
                        int startIdx = t * pixelsPerThread;
                        int endIdx = (t == numThreads - 1) ? whitePixels.size() : startIdx + pixelsPerThread;

                        threads.emplace_back(
                            calculateVisibilityAndLongestLineSubset,
                            cref(image), cref(whitePixels), ref(visibilityScores), ref(lineSizes),
                            startIdx, endIdx, ref(progressCounter2), totalPixels
                        );
                    }

                    for (auto& th : threads) {
                        if (th.joinable()) {
                            th.join();
                        }
                    }
                }
                catch (const exception& e) {
                    cerr << "Thread creation/joining error: " << e.what() << endl;
                }
                catch (...) {
                    cerr << "Unknown thread management error!" << endl;
                }

                double averageVisibility = accumulate(
                    visibilityScores.begin(), visibilityScores.end(), 0.0,
                    [](double sum, const atomic<int>& score) { return sum + score.load(); }
                ) / whitePixels.size();

                double averageLongestLine = accumulate(
                    lineSizes.begin(), lineSizes.end(), 0.0,
                    [](double sum, const atomic<int>& size) { return sum + size.load(); }
                ) / whitePixels.size();

                outputFile2 << entry.path().filename() << ","
                    << averageVisibility << ","
                    << averageLongestLine << "\n";
                outputFile2.flush();


                // Create heatmaps for visibility and line sizes
                Mat visibilityHeatmap = createHeatmap(image, visibilityScores, whitePixels);
                Mat lineSizeHeatmap = createHeatmap(image, lineSizes, whitePixels);

                string visibilityHeatmapPath = "visibility_heatmap_" + entry.path().filename().string();
                string lineSizeHeatmapPath = "line_size_heatmap_" + entry.path().filename().string();

                imwrite(visibilityHeatmapPath, visibilityHeatmap);
                imwrite(lineSizeHeatmapPath, lineSizeHeatmap);

                cout << "Heatmaps saved as '" << visibilityHeatmapPath << "' and '" << lineSizeHeatmapPath << "'" << endl;

            }
        }
        catch (const exception& e) {
            cerr << "Error processing image: " << entry.path().filename() << ". " << e.what() << endl;
        }
        catch (...) {
            cerr << "Unknown error processing image: " << entry.path().filename() << endl;
        }
    }

    outputFile.close();
    cout << "Results saved to entropy_results.csv" << endl;
    outputFile2.close();
    cout << "Results saved to visibility_results.csv" << endl;

    return 0;
}
