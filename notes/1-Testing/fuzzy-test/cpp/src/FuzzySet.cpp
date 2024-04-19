/**
 * @file FuzzySet.cpp
 * @author Manuel Lerchner
 * @date 17.04.24
 */

#include "FuzzySet.h"
#include <iostream>
#include <utility>

namespace autopas::fuzzy_logic {

    FuzzySet::FuzzySet(std::string linguisticTerm, const std::shared_ptr<MembershipFunction> &membershipFunction)
            : _linguisticTerm(std::move(linguisticTerm)), _membershipFunction(membershipFunction) {}

    FuzzySet::FuzzySet(std::string linguisticTerm,
                       const std::shared_ptr<BaseMembershipFunction> &baseMembershipFunction)
            : _linguisticTerm(std::move(linguisticTerm)), _baseMembershipFunction(baseMembershipFunction) {}

    FuzzySet::FuzzySet(std::string linguisticTerm, const std::shared_ptr<MembershipFunction> &membershipFunction,
                       const std::shared_ptr<CrispSet> &crispSet)
            : _linguisticTerm(std::move(linguisticTerm)), _membershipFunction(membershipFunction), _crispSet(crispSet) {}

    double FuzzySet::evaluate_membership(const std::map<std::string, double> &data) const {
        if (_baseMembershipFunction.has_value()) {
            // The current fuzzy set is a base set and has therefore a membership function which can be evaluated with a
            // single value.
            const auto crisp_dimensions = _crispSet->getDimensions();
            if (crisp_dimensions.size() != 1) {
            }
            const auto dimension_name = crisp_dimensions.begin()->first;
            auto v= (*_baseMembershipFunction)->operator()(data.at(dimension_name));
            return v;
        } else {
            // The current fuzzy set is a derived set and has needs to delegate the evaluation recursively to its base sets.
            return _membershipFunction->operator()(data);
        }
    }

    double FuzzySet::centroid(size_t numSamples) const {
        const auto crisp_dimensions = _crispSet->getDimensions();
        if (crisp_dimensions.size() != 1) {
        }

        const auto [dimensionName, range] = *crisp_dimensions.begin();
        const auto [minBoundary, maxBoundary] = range;

        // Uses the formula centroid_x = sum(x*y) / sum(y) to calculate the centroid of the fuzzy set numerically.
        double numerator = 0;
        double denominator = 0;
        for (double x = minBoundary; x <= maxBoundary; x += (maxBoundary - minBoundary) / (numSamples - 1)) {
            std::map<std::string, double> data = {{dimensionName, x}};
            const auto membership = evaluate_membership(data);
            numerator += x * membership;
            denominator += membership;
        }

        return denominator == 0 ? 0 : numerator / denominator;
    }

    std::shared_ptr<FuzzySet> FuzzySet::unionSet(const std::shared_ptr<FuzzySet> &lhs,
                                                 const std::shared_ptr<FuzzySet> &rhs) {
        const std::string newLinguisticTerm = lhs->_linguisticTerm + " or " + rhs->_linguisticTerm;
        const auto newCrispSet = (*lhs->_crispSet) * (*rhs->_crispSet);
        auto newMembershipFunction = std::make_shared<MembershipFunction>([lhs, rhs](auto data) {
            return std::max((lhs->evaluate_membership(data)), (rhs->evaluate_membership(data)));
        });
        return std::make_shared<FuzzySet>(newLinguisticTerm, newMembershipFunction, newCrispSet);
    }

    std::shared_ptr<FuzzySet> FuzzySet::intersectionSet(const std::shared_ptr<FuzzySet> &lhs,
                                                        const std::shared_ptr<FuzzySet> &rhs) {
        const std::string newLinguisticTerm = lhs->_linguisticTerm + " and " + rhs->_linguisticTerm;
        const auto newCrispSet = (*lhs->_crispSet) * (*rhs->_crispSet);
        auto newMembershipFunction = std::make_shared<MembershipFunction>([lhs, rhs](auto data) {
            return std::min((lhs->evaluate_membership(data)), (rhs->evaluate_membership(data)));
        });
        return std::make_shared<FuzzySet>(newLinguisticTerm, newMembershipFunction, newCrispSet);
    }

    std::shared_ptr<FuzzySet> FuzzySet::complementSet(const std::shared_ptr<FuzzySet> &lhs) {
        const std::string newLinguisticTerm = "not " + lhs->_linguisticTerm;
        const auto newCrispSet = lhs->_crispSet;
        auto newMembershipFunction = std::make_shared<MembershipFunction>([lhs](auto data) {
            return 1 - (lhs->evaluate_membership(data));
        });
        return std::make_shared<FuzzySet>(newLinguisticTerm, newMembershipFunction, newCrispSet);
    }

    const std::string &FuzzySet::getLinguisticTerm() const { return _linguisticTerm; }

    const std::shared_ptr<CrispSet> &FuzzySet::getCrispSet() const { return _crispSet; }

    void FuzzySet::setCrispSet(const std::shared_ptr<CrispSet> &crispSet) { _crispSet = crispSet; }


}  // namespace autopas::fuzzy_logic
