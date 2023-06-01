class Report:
    """Get GPTCache report including time and counts for different operations."""

    def __init__(self):
        self.op_pre = OpCounter()
        self.op_embedding = OpCounter()
        self.op_search = OpCounter()
        self.op_data = OpCounter()
        self.op_evaluation = OpCounter()
        self.op_post = OpCounter()
        self.op_llm = OpCounter()
        self.op_save = OpCounter()
        self.hint_cache_count = 0

    def pre(self, delta_time):
        """Pre-process counts and time.

        :param delta_time: additional runtime.
        """
        self.op_pre.total_time += delta_time
        self.op_pre.count += 1

    def embedding(self, delta_time):
        """Embedding counts and time.

        :param delta_time: additional runtime.
        """
        self.op_embedding.total_time += delta_time
        self.op_embedding.count += 1

    def search(self, delta_time):
        """Search counts and time.

        :param delta_time: additional runtime.
        """
        self.op_search.total_time += delta_time
        self.op_search.count += 1

    def data(self, delta_time):
        """Get data counts and time.

        :param delta_time: additional runtime.
        """

        self.op_data.total_time += delta_time
        self.op_data.count += 1

    def evaluation(self, delta_time):
        """Evaluation counts and time.

        :param delta_time: additional runtime.
        """
        self.op_evaluation.total_time += delta_time
        self.op_evaluation.count += 1

    def post(self, delta_time):
        """Post-process counts and time.

        :param delta_time: additional runtime.
        """
        self.op_post.total_time += delta_time
        self.op_post.count += 1

    def llm(self, delta_time):
        """LLM counts and time.

        :param delta_time: additional runtime.
        """
        self.op_llm.total_time += delta_time
        self.op_llm.count += 1

    def save(self, delta_time):
        """Save counts and time.

        :param delta_time: additional runtime.
        """
        self.op_save.total_time += delta_time
        self.op_save.count += 1

    def average_pre_time(self):
        """Average pre-process time."""
        return self.op_pre.average()

    def average_embedding_time(self):
        """Average embedding time."""
        return self.op_embedding.average()

    def average_search_time(self):
        """Average search time."""
        return self.op_search.average()

    def average_data_time(self):
        """Average data time."""
        return self.op_data.average()

    def average_evaluation_time(self):
        """Average evaluation time."""
        return self.op_evaluation.average()

    def average_post_time(self):
        """Average post-process time."""
        return self.op_post.average()

    def average_llm_time(self):
        """Average LLM time."""
        return self.op_llm.average()

    def average_save_time(self):
        """Average save time."""
        return self.op_save.average()

    def hint_cache(self):
        """hint cache count."""
        self.hint_cache_count += 1


class OpCounter:
    """Operation counter."""

    count = 0
    """Operation count."""
    total_time = 0
    """Total time."""

    def average(self):
        """Average time."""
        return round(self.total_time / self.count, 4) if self.count != 0 else 0
