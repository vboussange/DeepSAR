import unittest

import muscari


class PublicApiTest(unittest.TestCase):
    def test_all_exports_are_importable(self):
        for name in muscari.__all__:
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(muscari, name),
                    f"muscari.__all__ contains missing public symbol {name!r}",
                )


if __name__ == "__main__":
    unittest.main()
