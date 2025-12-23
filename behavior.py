"""
Data class for storing behavior information
"""

class Behavior:
    """Represents a single harmful behavior and attack results"""
    
    def __init__(self, harmful_behavior, suffix, response, harmbench_eval="", beaver_cost_score=""):
        self.harmful_behavior = harmful_behavior
        self.suffix = suffix
        self.response = response
        self.harmbench_eval = harmbench_eval
        self.beaver_cost_score = beaver_cost_score
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "harmful_behavior": self.harmful_behavior,
            "suffix": self.suffix,
            "response": self.response,
            "harmbench_eval": self.harmbench_eval,
            "beaver_cost_score": self.beaver_cost_score
        }
