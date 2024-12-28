#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PROCESS_H
#define PROCESS_H

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

        while (std::getline(data_stream, line))
        {
            std::stringstream ss(line);
            std::string feature;
            int i = 0;
            int index1 = -1, index2 = -1;

            // Read indices and initialize dimensions
            while (std::getline(ss, feature, ','))
            {
                if (i == 0) // First column: row index
                {
                    index1 = std::stoi(feature);
                    if (index1 >= vector_2d.size())
                    {
                        vector_2d.resize(index1 + 1);
                    }
                }
                else if (i == 1) // Second column: column index
                {
                    index2 = std::stoi(feature);
                    if (index2 >= vector_2d[index1].size())
                    {
                        vector_2d[index1].resize(index2 + 1);
                    }
                }
                else if (i == 2) // Third column: value
                {
                    vector_2d[index1][index2] = static_cast<T>(std::stod(feature));
                }

                i++;
            }
        }

        data_stream.close();  // Close file
        return vector_2d;
    }
      template <typename T>
  vector<vector<T>> vector2DSlice(vector<vector<T>> vec, int startRow, int endRow, int startCol, int endCol)
  {

    std::vector<std::vector<T>> sliced;

    for (int i = startRow; i < endRow; ++i)
    {
      std::vector<T> row(vec[i].begin() + startCol, vec[i].begin() + endCol);
      sliced.push_back(row);
    }
    auto o = size(sliced);
    return sliced;
  }
};

#endif