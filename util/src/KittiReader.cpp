//
// Created by bobin on 17-6-9.
//

#include <glob.h>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <glog/logging.h>

namespace ygz {

    inline std::vector<std::string> glob_vector(const std::string &pat) {
        using namespace std;
        glob_t glob_result;
        glob(pat.c_str(), GLOB_TILDE, NULL, &glob_result);
        vector<string> ret;
        for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
            ret.push_back(string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        return ret;
    }

    bool LoadImgNameFromPath(const std::string &strPathLeft, const std::string &strPathRight,
                             const std::string &strPathTimes, std::vector<std::string> &vstrImageLeft,
                             std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps) {

        vstrImageLeft = glob_vector(strPathLeft + "*.png");
        std::sort(vstrImageLeft.begin(), vstrImageLeft.end());
        vstrImageRight = glob_vector(strPathRight + "*.png");
        std::sort(vstrImageRight.begin(), vstrImageRight.end());


        std::ifstream fTimes;
        fTimes.open(strPathTimes.c_str());
        if (!fTimes) {
            LOG(ERROR) << "cannot find timestamp file: " << strPathTimes << std::endl;
            return false;
        }

        while (!fTimes.eof()) {
            std::string s;
            getline(fTimes, s);
            if (!s.empty()) {
                std::stringstream ss;
                ss << s;
                double t;
                ss >> t;
                vTimeStamps.push_back(t);
            }
        }

    }

}