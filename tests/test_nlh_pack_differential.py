# SPDX-License-Identifier: GPL-2.0-only
"""Offline guards for semantic versus scheduling-dependent frame properties."""
import copy
import unittest

from nlh_pack_differential import property_differences, validate_resources


def properties():
    return {'0': {'_NSSSigma': 25., '_NSSModelVersion': 5,
                  '_NSSResourceBytes': [0, 0, 0, 100, 50, 1000],
                  '_NSSResourcePeak': 200, '_NSSResourceLimit': 10000}}


class ResourcePropertyTests(unittest.TestCase):
    def test_concurrent_resource_changes_are_recorded_but_not_semantic(self):
        left = properties()
        right = copy.deepcopy(left)
        right['0']['_NSSResourceBytes'][3] = 200
        right['0']['_NSSResourcePeak'] = 400
        self.assertEqual(len(property_differences(left, right)), 2)
        self.assertEqual(property_differences(left, right, False), [])
        validate_resources(right['0'])

    def test_serial_ab_resources_remain_exact(self):
        left, right = properties(), properties()
        right['0']['_NSSResourcePeak'] += 1
        self.assertEqual(property_differences(left, right, True)[0]['property'], '_NSSResourcePeak')

    def test_sigma_model_and_budget_limit_remain_semantic(self):
        for key in ('_NSSSigma', '_NSSModelVersion', '_NSSResourceLimit'):
            left, right = properties(), properties()
            right['0'][key] += 1
            self.assertEqual(property_differences(left, right, False)[0]['property'], key)

    def test_resource_bounds_exclude_unowned_framework_references(self):
        validate_resources(properties()['0'])
        for key, value in (('_NSSResourcePeak', 10001), ('_NSSResourcePeak', 100),
                           ('_NSSResourceBytes', [0, 0, 0, -1, 50, 1000])):
            row = properties()['0']
            row[key] = value
            with self.assertRaises(AssertionError):
                validate_resources(row)


if __name__ == '__main__':
    unittest.main()
