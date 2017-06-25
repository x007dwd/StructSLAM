//
// Created by bobin on 17-6-9.
//

#ifndef YGZ_STEREO_KITTIREADER_H
#define YGZ_STEREO_KITTIREADER_H

#include <vector>
#include <string>


namespace ygz {

    inline std::vector<std::string> glob_vector(const std::string &pat);

    bool LoadImgNameFromPath(const std::string &strPathLeft, const std::string &strPathRight,
                             const std::string &strPathTimes, std::vector<std::string> &vstrImageLeft,
                             std::vector<std::string> &vstrImageRight, std::vector<double> &vTimeStamps);

}

#endif //YGZ_STEREO_KITTIREADER_H
