import unittest

from five_gram_kn import FiveGramKNModel


class FiveGramKNModelTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            ["best", "places", "to", "visit", "in", "india"],
            ["best", "places", "to", "visit", "near", "me"],
            ["best", "places", "to", "visit", "in", "chennai"],
            ["best", "places", "to", "visit", "during", "summer"],
            ["best", "places", "to", "visit", "in", "delhi"],
            ["best", "places", "to", "travel", "in", "india"],
        ]
        self.model = FiveGramKNModel()
        self.model.fit(self.corpus)

    def test_predicts_high_probability_completion(self):
        suggestions = self.model.suggest(["best", "places", "to", "visit"], top_k=4)
        self.assertGreaterEqual(len(suggestions), 4)
        self.assertIn("in", [token for token, _ in suggestions])
        self.assertIn("near", [token for token, _ in suggestions])

    def test_handles_oov_tokens(self):
        suggestions = self.model.suggest(["best", "places", "to", "visit", "in", "mumbai"], top_k=3)
        self.assertTrue(suggestions)
        self.assertEqual(suggestions[0][0], "<UNK>")

    def test_probabilities_follow_backoff_for_short_context(self):
        prob = self.model.probability("india", ["best", "places", "to", "visit", "in"])
        self.assertGreater(prob, 0.0)

    def test_prefix_with_unknown_token_is_supported(self):
        suggestions = self.model.suggest(["best", "places", "to", "mystery"], top_k=3)
        self.assertTrue(suggestions)


if __name__ == "__main__":
    unittest.main()
