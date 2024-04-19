/**
 * @file FuzzySystem.cpp
 * @author Manuel Lerchner
 * @date 18.04.24
 */

#include "FuzzySystem.h"

namespace autopas::fuzzy_logic {

    void FuzzySystem::addRule(const FuzzyRule &rule) { _rules.push_back(rule); };

    std::shared_ptr<FuzzySet> FuzzySystem::applyRules(const std::map<std::string, double> &data) const {
        std::vector<std::shared_ptr<FuzzySet>> consequences;

        std::transform(_rules.begin(), _rules.end(), std::back_inserter(consequences),
                       [&data](const FuzzyRule &rule) { return rule.apply(data); });

        auto result = consequences[0];
        for (size_t i = 1; i < consequences.size(); ++i) {
            result = result || consequences[i];
        }

        return result;
    }

    double FuzzySystem::predict(const std::map<std::string, double> &data, size_t numSamples) const {
        auto unionSet = applyRules(data);
        return unionSet->centroid(numSamples);
    }

}  // namespace autopas::fuzzy_logic