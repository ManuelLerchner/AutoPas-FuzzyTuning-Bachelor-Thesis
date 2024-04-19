/**
 * @file FuzzySet.h
 * @author Manuel Lerchner
 * @date 17.04.24
 */

#pragma once
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "CrispSet.h"

namespace autopas::fuzzy_logic {

/**
 * Used to represent a mathematical Fuzzy-Set.
 */
class FuzzySet {
 public:
  using MembershipFunction = std::function<double(std::map<std::string, double>)>;
  using BaseMembershipFunction = std::function<double(double)>;

  /**
   * Constructs a FuzzySet with the given linguistic term and membership function.
   * @param linguisticTerm
   * @param membershipFunction
   */
  FuzzySet(std::string linguisticTerm, const std::shared_ptr<MembershipFunction> &membershipFunction);

  /**
   * Constructs a FuzzySet with the given linguistic term and membership function.
   * @param linguisticTerm
   * @param membershipFunction
   */
  FuzzySet(std::string linguisticTerm, const std::shared_ptr<BaseMembershipFunction> &baseMembershipFunction);

  /**
   * Constructs a FuzzySet with the given linguistic term and crisp set.
   * @param linguisticTerm
   * @param membershipFunction
   * @param crispSet
   */
  FuzzySet(std::string linguisticTerm, const std::shared_ptr<MembershipFunction> &membershipFunction,
           const std::shared_ptr<CrispSet> &crispSet);

  /**
   * Evaluates the membership function of this FuzzySet at the given value.
   * @param data A map of the form {dimension_name: value}.
   * @return The membership value of the given value in this FuzzySet.
   */
  [[nodiscard]] double evaluate_membership(const std::map<std::string, double> &data) const;

  /**
   * Calculates the x-coordinate of the centroid of this FuzzySet.
   * @return The x-coordinate of the centroid of this FuzzySet.
   */
  [[nodiscard]] double centroid(size_t numSamples = 100) const;

  /**
   * Calculates the intersection of two FuzzySets.
   * @param rhs
   * @return A new FuzzySet, which is the intersection of this and rhs.
   */
  static std::shared_ptr<FuzzySet> unionSet(const std::shared_ptr<FuzzySet> &lhs,
                                            const std::shared_ptr<FuzzySet> &rhs);

  static std::shared_ptr<FuzzySet> intersectionSet(const std::shared_ptr<FuzzySet> &lhs,
                                                   const std::shared_ptr<FuzzySet> &rhs);

  static std::shared_ptr<FuzzySet> complementSet(const std::shared_ptr<FuzzySet> &lhs);

  /**
   * Returns the linguistic term of the FuzzySet.
   * @return The linguistic term of the FuzzySet.
   */
  [[nodiscard]] const std::string &getLinguisticTerm() const;

  /**
   * Returns the crisp set of the FuzzySet.
   * @return The crisp set of the FuzzySet.
   */
  [[nodiscard]] const std::shared_ptr<CrispSet> &getCrispSet() const;

  /**
   * Sets the crisp set of the FuzzySet.
   * @param crispSet
   */
  void setCrispSet(const std::shared_ptr<CrispSet> &crispSet);

 private:
  std::string _linguisticTerm;
  std::shared_ptr<MembershipFunction> _membershipFunction;
  std::optional<std::shared_ptr<BaseMembershipFunction>> _baseMembershipFunction;
  std::shared_ptr<CrispSet> _crispSet;
};

}  // namespace autopas::fuzzy_logic
