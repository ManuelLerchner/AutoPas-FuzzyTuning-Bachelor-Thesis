/**
 * @file FuzzyTuning.cpp
 * @author Manuel Lerchner
 * @date 17.04.24
 */

#include <iostream>

#include "src/CrispSet.h"
#include "src/FuzzyRule.h"
#include "src/FuzzySystem.h"
#include "src/LinguisticVariable.h"
#include "src/MembershipFunction.h"

int main() {
    using namespace autopas::fuzzy_logic;

    auto service = LinguisticVariable("service", std::pair(0, 10));
    auto food = LinguisticVariable("food", std::pair(0, 10));
    auto tip = LinguisticVariable("tip", std::pair(0, 30));

    service.addLinguisticTerm(makeGaussian("poor", 0, 1.5));
    service.addLinguisticTerm(makeGaussian("good", 5, 1.5));
    service.addLinguisticTerm(makeGaussian("excellent", 10, 1.5));

    food.addLinguisticTerm(makeTrapezoid("rancid", 0, 0, 1, 3));
    food.addLinguisticTerm(makeTrapezoid("delicious", 7, 9, 10, 10));

    tip.addLinguisticTerm(makeTriangle("cheap", 0, 5, 10));
    tip.addLinguisticTerm(makeTriangle("average", 10, 15, 20));
    tip.addLinguisticTerm(makeTriangle("generous", 20, 25, 30));


    auto fs = FuzzySystem();
    fs.addRule(FuzzyRule(service == "poor" || food == "rancid", tip == "cheap"));
    fs.addRule(FuzzyRule(service == "good", tip == "average"));
    fs.addRule(FuzzyRule(service == "excellent" || food == "delicious", tip == "generous"));


    auto x2 = fs.predict(
            {{"service", 1},
             {"food",    3}});

    std::cout << "Predicted tip: " << x2 << std::endl;

    return 0;
}
