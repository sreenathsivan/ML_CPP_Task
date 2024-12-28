#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PROCESS_H
#define PROCESS_H
using namespace std;
template <typename T>
class process {
public:
    // Function to read CSV into a 2D vector
    std::vector<std::vector<T>> CSV2VEC(const std::string& vec_filepath)
    {
        std::vector<std::vector<T>> vector_2d;
        std::ifstream data_stream(vec_filepath);
        std::string line;
        
        if (!data_stream)
        {
            std::cout << "Error: Unable to open the file." << std::endl;
            return vector_2d;  // Return empty vector in case of error
        }
        int index1 = 0;
        while (std::getline(data_stream, line))
        {
            std::stringstream ss(line);
            std::string feature;
            int i = 0;
            
            std::vector<int> row_data;
            vector_2d.resize(index1 + 1);
            // Read indices and initialize dimensions
            while (std::getline(ss, feature, ','))
            {
              
               vector_2d[index1].resize(i + 1);
              //  vector_2d[0][i]=1.1;
                if (i <8) // First column: row index
                {   
                  
                        // cout<<std::stof(feature);
                        
                        vector_2d[index1][i] = static_cast<T>(std::stof(feature));
                    
                }
                 else
                {
                  //  cout<<i;
                    
                //     // vector_2d.resize(index1 + 1);
                    vector_2d[index1][i] = static_cast<T>(std::stoi(feature));
                }

                i++;
            }
             index1++;
        }

        data_stream.close();  // Close file
        return vector_2d;
    }

     
  vector<vector<T>> vector2DSlice(vector<vector<T>> vec, int startRow, int endRow, int startCol, int endCol)
  {

    std::vector<std::vector<T>> sliced;

    for (int i = startRow; i < endRow; ++i)
    {
      std::vector<T> row(vec[i].begin() + startCol, vec[i].begin() + endCol+1);
      sliced.push_back(row);
    }
    auto o = size(sliced);
    return sliced;
  }

  // Function to standardize the data
std::vector<std::vector<float>> fit_transform(const std::vector<std::vector<float>>& X) {
    int rows = X.size();
    int cols = X[0].size();
    
    std::vector<float> mean(cols, 0.0);
    std::vector<float> std(cols, 0.0);
    
    // Calculate the mean for each column
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            mean[j] += X[i][j];
        }
        mean[j] /= rows;
    }
    
    // Calculate the standard deviation for each column
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            std[j] += std::pow(X[i][j] - mean[j], 2);
        }
        std[j] = std::sqrt(std[j] / rows);
    }

    // Print standard deviations
    std::cout << "Standard deviations: ";
    for (int j = 0; j < cols; ++j) {
        std::cout << std[j] << " ";
    }
    std::cout << std::endl;

    // Standardize the data
    std::vector<std::vector<float>> X_scaled(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X_scaled[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }

    return X_scaled;
}

};

#endif